# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention CP backend -- faithful port of vLLM-Ascend's GQA
Context-Parallel attention (prefill head/tail zigzag + DCP decode merge).

This is a CP-specific backend that subclasses the stock FlashAttention backend
and overrides only the CP pieces (``forward`` dispatch + ``do_kv_cache_update``
all-gather/restore). It is a faithful port of
``vllm_ascend/attention/context_parallel/attention_cp.py``, with NPU kernels
swapped for stock flash_attn / torch primitives (see the plan's substitution
table). Decode + mixed dispatch land in P3/P4.

Mixed-batch contract (ascend ``forward_impl`` :951): decode tokens are laid out
first (``query[:num_decode_tokens]``), prefill tokens after
(``query[num_decode_tokens:]``); the two sub-paths write disjoint output regions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch

from vllm.distributed.parallel_state import get_dcp_group, get_pcp_group
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    reshape_and_cache_flash,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.attention.cp import CPContext, cp_merge_decode_out_lse
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states


@dataclass
class FlashAttentionCPMetadata:
    """Per-step CP tensors (port of ascend ``AscendPCPMetadata``).

    Populated by :class:`FlashAttentionCPMetadataBuilder` from the runner-side
    :class:`~vllm.v1.worker.pcp_manager.PCPManager` output. ``None`` when PCP is
    inactive for this step (no prefill requests).
    """

    num_decode_tokens: int = 0
    num_prefills: int = 0
    # CP topology carried for the decode merge (populated by the builder/runner).
    interleave: int = 1
    dcp_comm_backend: str = "ag_rs"
    # Per-rank prefill-Q length (= 2*chunk_len) per prefill request.
    prefill_query_lens: torch.Tensor | None = None
    # PCP index tensors (on device), all into the local-Q / all-gathered-KV.
    q_head_idx: torch.Tensor | None = None
    q_tail_idx: torch.Tensor | None = None
    kv_with_q_head_nomask_idx: torch.Tensor | None = None
    kv_with_q_head_mask_idx: torch.Tensor | None = None
    kv_with_q_tail_nomask_idx: torch.Tensor | None = None
    kv_with_q_tail_mask_idx: torch.Tensor | None = None
    q_full_idx: torch.Tensor | None = None
    # Cumulative seqlens (Python lists; callers prepend 0 for cu_seqlens).
    attn_mask_seqlens: list = field(default_factory=list)
    head_attn_nomask_seqlens: list = field(default_factory=list)
    tail_attn_nomask_seqlens: list = field(default_factory=list)
    # all_gather+restore index for KV / hidden states (length = padded total).
    pcp_allgather_restore_idx: torch.Tensor | None = None
    num_actual_tokens_pcp_padded: int = 0


class FlashAttentionCPMetadataBuilder(FlashAttentionMetadataBuilder):
    """Builds :class:`FlashAttentionCPMetadata` from runner-side PCP state.

    The runner sets ``common_attn_metadata.pcp`` to the per-step PCP split
    (``_PCPStep``); this builder copies the index tensors onto the device,
    computes the prefill-attention index set via
    :func:`generate_prefill_metadata`, and attaches them to ``metadata.cp``.
    """

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        metadata.cp = self._build_cp_metadata(common_attn_metadata)
        return metadata

    def _build_cp_metadata(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> FlashAttentionCPMetadata:
        from vllm.distributed.parallel_state import get_pcp_group

        try:
            pcp_group = get_pcp_group()
            pcp_size = pcp_group.world_size
            pcp_rank = pcp_group.rank_in_group
        except AssertionError:
            pcp_size = 1
            pcp_rank = 0

        step: Any = common_attn_metadata.pcp
        if pcp_size <= 1 or step is None:
            return FlashAttentionCPMetadata()

        device = common_attn_metadata.query_start_loc.device
        cp = FlashAttentionCPMetadata(
            num_decode_tokens=step.num_decode_tokens,
            num_prefills=step.num_prefills,
            interleave=self.cp_kv_cache_interleave_size,
            dcp_comm_backend=self.parallel_config.dcp_comm_backend,
        )
        if step.restore_idx is not None:
            cp.pcp_allgather_restore_idx = step.restore_idx.to(device)
        if step.num_prefills > 0 and step.prefill_query_lens is not None:
            from vllm.v1.worker.pcp_manager import generate_prefill_metadata

            prefill_q_lens = (
                step.prefill_query_lens.detach().cpu().numpy().astype(np.int32)
            )
            pm = generate_prefill_metadata(prefill_q_lens, pcp_size, pcp_rank)
            cp.q_head_idx = torch.as_tensor(pm.q_head_idx, device=device)
            cp.q_tail_idx = torch.as_tensor(pm.q_tail_idx, device=device)
            cp.kv_with_q_head_nomask_idx = torch.as_tensor(
                pm.kv_with_q_head_nomask_idx, device=device
            )
            cp.kv_with_q_head_mask_idx = torch.as_tensor(
                pm.kv_with_q_head_mask_idx, device=device
            )
            cp.kv_with_q_tail_nomask_idx = torch.as_tensor(
                pm.kv_with_q_tail_nomask_idx, device=device
            )
            cp.kv_with_q_tail_mask_idx = torch.as_tensor(
                pm.kv_with_q_tail_mask_idx, device=device
            )
            cp.q_full_idx = torch.as_tensor(pm.q_full_idx, device=device)
            cp.attn_mask_seqlens = list(pm.attn_mask_seqlens)
            cp.head_attn_nomask_seqlens = list(pm.head_attn_nomask_seqlens)
            cp.tail_attn_nomask_seqlens = list(pm.tail_attn_nomask_seqlens)
        return cp


class FlashAttentionCPBackend(FlashAttentionBackend):
    """FlashAttention backend whose impl/builder swap to the CP variants.

    Mirrors ascend's conditional ``get_impl_cls``/``get_builder_cls``
    (``attention_v1.py:85-98``); selection (when ``pcp_size>1``) is wired in P4.

    ``forward_includes_kv_cache_update = True`` because the PCP cache write
    (all-gather KV across PCP + restore) needs ``attn_metadata.cp`` -- which the
    separate ``do_kv_cache_update`` hook does not receive. Ascend's
    ``reshape_and_cache`` is likewise part of the attention forward.
    """

    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls():
        return FlashAttentionCPImpl

    @staticmethod
    def get_builder_cls():
        return FlashAttentionCPMetadataBuilder


class FlashAttentionCPImpl(FlashAttentionImpl):
    """CP attention impl. Subclasses stock FA; overrides forward + cache write.

    Implements the ascend ``forward_impl`` (:951-1059) split: decode tokens
    (leading ``num_decode_tokens``) -> :meth:`_forward_decode_pcp` (Mode-2 LSE
    merge), prefill tokens -> :meth:`_forward_prefill_cp` (head/tail zigzag).
    Runner-side token slicing + backend selection (P4) must be in place before
    this runs end-to-end.
    """

    supports_pcp: bool = True

    # ------------------------------------------------------------------
    # Forward: ascend forward_impl (:951-1059) -- split decode/prefill. Because
    # ``forward_includes_kv_cache_update`` is True on the backend, the KV cache
    # write (all-gather KV across PCP + restore + scatter via slot_mapping) is
    # done here, where ``attn_metadata.cp`` is in scope (ascend's reshape_and_cache
    # is likewise part of the attention forward).
    # ------------------------------------------------------------------
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pcp_group = get_pcp_group()
        if pcp_group.world_size <= 1:
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )

        cp = cast(FlashAttentionCPMetadata, attn_metadata.cp)
        num_decode_tokens = cp.num_decode_tokens
        if num_decode_tokens > 0:
            # Decode cache write: decode tokens are replicated across PCP ranks,
            # so every rank has the decode KV; writing it via the full
            # (natural) slot_mapping lands each token on its owner-rank slot
            # (PAD_ID skips the rest). Port of ascend reshape_and_cache (:816-824).
            key_cache, value_cache = kv_cache.unbind(1)
            reshape_and_cache_flash(
                key[:num_decode_tokens],
                value[:num_decode_tokens],
                key_cache,
                value_cache,
                attn_metadata.slot_mapping[:num_decode_tokens],
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )
            output[:num_decode_tokens] = self._forward_decode_pcp(
                query[:num_decode_tokens], key_cache, value_cache, attn_metadata, cp
            )

        if cp.num_prefills > 0:
            # Cache write: recover the full (natural-order) prefill KV via
            # all-gather + restore, then scatter-write via the full slot_mapping
            # (slot kernel PAD_ID skips non-local -> sharded cache). Port of
            # ascend reshape_and_cache (:826-856).
            key_cache, value_cache = kv_cache.unbind(1)
            per_rank_prefill_kv = query.shape[0] - num_decode_tokens
            kv = torch.cat(
                [
                    key[pcp_group.world_size * num_decode_tokens :],
                    value[pcp_group.world_size * num_decode_tokens :],
                ],
                dim=-1,
            )
            all_kv = pcp_group.all_gather(kv[:per_rank_prefill_kv].contiguous(), dim=0)
            assert cp.pcp_allgather_restore_idx is not None
            restore = cp.pcp_allgather_restore_idx[: all_kv.shape[0]].long()
            all_kv = torch.index_select(all_kv, 0, restore)
            full_key, full_value = all_kv.split(
                [self.head_size, self.head_size], dim=-1
            )
            reshape_and_cache_flash(
                full_key,
                full_value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

            prefill_out = self._forward_prefill_cp(
                query[num_decode_tokens:], full_key, full_value, attn_metadata
            )
            output[num_decode_tokens : num_decode_tokens + prefill_out.shape[0]] = (
                prefill_out
            )
        return output

    def _forward_decode_pcp(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        cp: FlashAttentionCPMetadata,
    ) -> torch.Tensor:
        """PCP decode (Mode-2 LSE merge).

        Port of ascend ``_forward_decode_pcp_dcp`` (:566-658) +
        ``_process_attn_out_lse``. The KV cache is sharded round-robin across
        the full CP group (``total_cp = pcp*dcp``), so each rank owns a disjoint
        slice of each request's KV. Each rank attends its (replicated) decode Q
        to its local KV shard, then the per-rank partial ``(out, lse)`` are
        merged across PCP (+DCP) via :func:`cp_merge_decode_out_lse`.

        Decode Q is not gathered across PCP (decode tokens are replicated); it is
        gathered across DCP heads when ``dcp>1``.
        """
        assert self.vllm_flash_attn_version is not None
        pcp_group = get_pcp_group()
        dcp_group = get_dcp_group()
        ctx = CPContext(
            pcp_size=pcp_group.world_size,
            pcp_rank=pcp_group.rank_in_group,
            dcp_size=dcp_group.world_size,
            dcp_rank=dcp_group.rank_in_group,
            pcp_group=pcp_group,
            dcp_group=dcp_group,
            interleave=cp.interleave,
            dcp_comm_backend=cp.dcp_comm_backend,
        )
        total_cp = pcp_group.world_size * dcp_group.world_size
        total_rank = (
            pcp_group.rank_in_group * dcp_group.world_size + dcp_group.rank_in_group
        )

        # Decode requests are the leading num_decode_tokens requests (each qlen=1,
        # so request count == token count). Their full seq_lens give the per-rank
        # local KV shard length (round-robin across total_cp).
        num_decode_tokens = query.shape[0]
        decode_seq_lens = attn_metadata.seq_lens[:num_decode_tokens]
        local_kv = get_dcp_local_seq_lens(
            decode_seq_lens, total_cp, total_rank, ctx.interleave
        )
        # Workspace bound without a GPU->CPU sync.
        max_local_kv = (
            (attn_metadata.max_seq_len + total_cp * ctx.interleave - 1)
            // (total_cp * ctx.interleave)
        ) * ctx.interleave

        q = query.contiguous()
        if dcp_group.world_size > 1:
            q = dcp_group.all_gather(q, dim=1)
        cu_q = attn_metadata.query_start_loc[: num_decode_tokens + 1]
        sw = list(self.sliding_window) if self.sliding_window is not None else None
        attn_out, softmax_lse = flash_attn_varlen_func(
            q=q,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_q,
            max_seqlen_q=1,
            seqused_k=local_kv,
            max_seqlen_k=max_local_kv,
            softmax_scale=self.scale,
            causal=False,
            alibi_slopes=self.alibi_slopes,
            window_size=sw,
            block_table=attn_metadata.block_table,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            fa_version=self.vllm_flash_attn_version,
            num_splits=attn_metadata.max_num_splits,
        )
        # FA returns lse as [H, B]; cp_merge wants [B, H]. Returns merged [B, H, D].
        merged, _ = cp_merge_decode_out_lse(
            ctx, attn_out, softmax_lse.transpose(0, 1), self.head_size
        )
        return merged

    def _forward_prefill_cp(
        self,
        query: torch.Tensor,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
    ) -> torch.Tensor:
        """Prefill head/tail attention against the all-gathered KV.

        Port of ascend ``_forward_prefill_cp`` (:485-564). Each rank holds its
        pre-sharded Q (head+tail chunks); the full KV was recovered by the caller
        (all_gather + restore). Per chunk, a causal-free ``nomask`` call over the
        prefix KV is LSE-merged with a causal ``mask`` call over the local chunk
        KV. Outputs are re-interleaved by ``q_full_idx``.
        """
        cp = cast(FlashAttentionCPMetadata, attn_metadata.cp)
        # Index tensors are populated by the builder when num_prefills>0 (the
        # caller gates this path); narrow them for the kernel calls below.
        assert cp.q_head_idx is not None
        assert cp.q_tail_idx is not None
        assert cp.kv_with_q_head_nomask_idx is not None
        assert cp.kv_with_q_head_mask_idx is not None
        assert cp.kv_with_q_tail_nomask_idx is not None
        assert cp.kv_with_q_tail_mask_idx is not None
        assert cp.q_full_idx is not None
        assert cp.pcp_allgather_restore_idx is not None
        pcp_rank = get_pcp_group().rank_in_group

        # Slice this rank's Q chunks and the corresponding KV (prefix=nomask,
        # own chunk=mask) via the precomputed indices.
        q_head = torch.index_select(query, 0, cp.q_head_idx.long())
        q_tail = torch.index_select(query, 0, cp.q_tail_idx.long())
        data_head = {
            "q": q_head,
            "k_nomask": (
                torch.index_select(full_key, 0, cp.kv_with_q_head_nomask_idx.long())
                if pcp_rank > 0
                else None
            ),
            "v_nomask": (
                torch.index_select(full_value, 0, cp.kv_with_q_head_nomask_idx.long())
                if pcp_rank > 0
                else None
            ),
            "k_mask": torch.index_select(
                full_key, 0, cp.kv_with_q_head_mask_idx.long()
            ),
            "v_mask": torch.index_select(
                full_value, 0, cp.kv_with_q_head_mask_idx.long()
            ),
        }
        data_tail = {
            "q": q_tail,
            "k_nomask": torch.index_select(
                full_key, 0, cp.kv_with_q_tail_nomask_idx.long()
            ),
            "v_nomask": torch.index_select(
                full_value, 0, cp.kv_with_q_tail_nomask_idx.long()
            ),
            "k_mask": torch.index_select(
                full_key, 0, cp.kv_with_q_tail_mask_idx.long()
            ),
            "v_mask": torch.index_select(
                full_value, 0, cp.kv_with_q_tail_mask_idx.long()
            ),
        }

        output_head, _ = self._attention_with_nomask_and_mask(
            data_head, cp, is_head=True
        )
        output_tail, _ = self._attention_with_nomask_and_mask(
            data_tail, cp, is_head=False
        )
        # Re-interleave head+tail outputs back into local-Q natural order.
        return torch.index_select(
            torch.cat([output_head, output_tail], dim=0), 0, cp.q_full_idx.long()
        )

    def _attention_with_nomask_and_mask(
        self,
        data: dict,
        cp: FlashAttentionCPMetadata,
        is_head: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """nomask (causal=False, prefix KV) + mask (causal=True, local chunk) + merge.

        Port of ascend ``_attention_with_nomask_and_mask`` (:421-483), mapping
        ``npu_fused_infer_attention_score(actual_seq_lengths/_kv=...)`` to
        ``flash_attn_varlen_func(cu_seqlens_q/k=...)``.
        """
        assert self.vllm_flash_attn_version is not None
        device = data["q"].device
        # cu_seqlens: prepend 0 to the cumulative lists.
        mask_cu = torch.tensor(
            [0, *cp.attn_mask_seqlens], device=device, dtype=torch.int32
        )
        nomask_cu = torch.tensor(
            [
                0,
                *(
                    cp.head_attn_nomask_seqlens
                    if is_head
                    else cp.tail_attn_nomask_seqlens
                ),
            ],
            device=device,
            dtype=torch.int32,
        )
        chunk_max = max(cp.attn_mask_seqlens) if cp.attn_mask_seqlens else 0
        nomask_max = (
            max(cp.head_attn_nomask_seqlens if is_head else cp.tail_attn_nomask_seqlens)
            or 0
        )
        sw = list(self.sliding_window) if self.sliding_window is not None else None

        attn_out_mask, lse_mask = flash_attn_varlen_func(
            q=data["q"],
            k=data["k_mask"],
            v=data["v_mask"],
            cu_seqlens_q=mask_cu,
            cu_seqlens_k=mask_cu,
            max_seqlen_q=chunk_max,
            max_seqlen_k=chunk_max,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            window_size=sw,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            fa_version=self.vllm_flash_attn_version,
        )
        if data["k_nomask"] is not None:
            attn_out_nomask, lse_nomask = flash_attn_varlen_func(
                q=data["q"],
                k=data["k_nomask"],
                v=data["v_nomask"],
                cu_seqlens_q=mask_cu,
                cu_seqlens_k=nomask_cu,
                max_seqlen_q=chunk_max,
                max_seqlen_k=nomask_max,
                softmax_scale=self.scale,
                causal=False,
                alibi_slopes=self.alibi_slopes,
                window_size=sw,
                softcap=self.logits_soft_cap,
                return_softmax_lse=True,
                fa_version=self.vllm_flash_attn_version,
            )
            merged = torch.empty_like(attn_out_mask)
            # prefix (nomask) is the "unpadded" KV-cache state; the local causal
            # chunk (mask) is the "suffix" new-KV state.
            merge_attn_states(
                merged,
                attn_out_nomask,
                lse_nomask,
                attn_out_mask,
                lse_mask,
            )
            return merged, None
        return attn_out_mask, lse_mask
