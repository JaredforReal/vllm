# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import MambaSpec


class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None
    # Chunk-kernel inputs for prefill
    prefill_query_start_loc: torch.Tensor | None = None
    prefill_state_indices: torch.Tensor | None = None
    prefill_has_initial_state: torch.Tensor | None = None

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


@triton.jit(do_not_specialize=["num_spec_decodes", "batch_size", "spec_token_size"])
def _stage_pure_spec_metadata_kernel(
    block_table_ptr,
    query_start_loc_ptr,
    num_accepted_tokens_ptr,
    out_state_indices_ptr,
    out_sequence_masks_ptr,
    out_query_start_loc_ptr,
    out_num_accepted_tokens_ptr,
    out_token_indx_ptr,
    block_table_stride_0,
    block_table_stride_1,
    block_table_num_cols,
    out_state_indices_stride_0,
    num_spec_decodes,
    batch_size,
    spec_token_size,
    NUM_STATE_SLOTS: tl.constexpr,
    BLOCK_STATE_SLOTS: tl.constexpr,
    NULL_STATE_ID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Stage a pure spec-decode batch into the CUDA-graph metadata buffers.

    Real requests occupy rows [0, num_spec_decodes) and padding trails them, so
    every request-indexed output is a prefix copy plus a constant fill, and the
    spec token index is the identity permutation.
    """
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    real = rows < num_spec_decodes
    in_batch = rows < batch_size

    slots = tl.arange(0, BLOCK_STATE_SLOTS)
    valid_slot = slots < NUM_STATE_SLOTS
    # A single-column table (no per-step state slots) broadcasts like copy_.
    cols = tl.minimum(slots, block_table_num_cols - 1)
    state_indices = tl.load(
        block_table_ptr
        + rows[:, None] * block_table_stride_0
        + cols[None, :] * block_table_stride_1,
        mask=real[:, None] & valid_slot[None, :],
        other=NULL_STATE_ID,
    )
    tl.store(
        out_state_indices_ptr
        + rows[:, None] * out_state_indices_stride_0
        + slots[None, :],
        state_indices,
        mask=in_batch[:, None] & valid_slot[None, :],
    )
    tl.store(out_sequence_masks_ptr + rows, real, mask=in_batch)

    query_start_loc = tl.load(
        query_start_loc_ptr + tl.minimum(rows, num_spec_decodes),
        mask=rows <= batch_size,
    )
    tl.store(out_query_start_loc_ptr + rows, query_start_loc, mask=rows <= batch_size)

    num_accepted_tokens = tl.load(num_accepted_tokens_ptr + rows, mask=real, other=1)
    tl.store(out_num_accepted_tokens_ptr + rows, num_accepted_tokens, mask=in_batch)

    tl.store(out_token_indx_ptr + rows, rows.to(tl.int32), mask=rows < spec_token_size)


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    kv_cache_spec: MambaSpec
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _resolve_gdn_prefill_backend,
        )

        self.gdn_prefill_backend: Literal["triton", "flashinfer", "cutedsl"]
        _, self.gdn_prefill_backend = _resolve_gdn_prefill_backend(vllm_config)

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode: bool = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        self.decode_cudagraph_max_bs: int = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_sequence_masks: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.bool,
            device=device,
        )
        self.spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

    def _stage_pure_spec_metadata(
        self,
        block_table_tensor: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        num_spec_decodes: int,
        batch_size: int,
        spec_token_size: int,
    ) -> None:
        """Fill the CUDA-graph metadata buffers for a pure spec-decode batch.

        Replaces the per-buffer copy_/fill_ sequence (and the device arange for
        the identity token permutation) with one kernel launch.
        """
        if not block_table_tensor.is_cuda:
            # CPU / non-Triton fallback with the same buffer contents.
            n = num_spec_decodes
            self.spec_state_indices_tensor[:n].copy_(
                block_table_tensor[:n, : self.num_spec + 1]
            )
            self.spec_state_indices_tensor[n:batch_size].fill_(NULL_BLOCK_ID)
            self.spec_sequence_masks[:n].fill_(True)
            self.spec_sequence_masks[n:batch_size].fill_(False)
            self.spec_query_start_loc[: n + 1].copy_(query_start_loc[: n + 1])
            self.spec_query_start_loc[n + 1 : batch_size + 1] = query_start_loc[n]
            self.num_accepted_tokens[:n].copy_(num_accepted_tokens[:n])
            self.num_accepted_tokens[n:batch_size].fill_(1)
            torch.arange(spec_token_size, out=self.spec_token_indx[:spec_token_size])
            return
        block = 128
        grid = (triton.cdiv(max(batch_size + 1, spec_token_size), block),)
        _stage_pure_spec_metadata_kernel[grid](
            block_table_tensor,
            query_start_loc,
            num_accepted_tokens,
            self.spec_state_indices_tensor,
            self.spec_sequence_masks,
            self.spec_query_start_loc,
            self.num_accepted_tokens,
            self.spec_token_indx,
            block_table_tensor.stride(0),
            block_table_tensor.stride(1),
            block_table_tensor.shape[1],
            self.spec_state_indices_tensor.stride(0),
            num_spec_decodes,
            batch_size,
            spec_token_size,
            NUM_STATE_SLOTS=self.num_spec + 1,
            BLOCK_STATE_SLOTS=triton.next_power_of_2(self.num_spec + 1),
            NULL_STATE_ID=NULL_BLOCK_ID,
            BLOCK=block,
            num_warps=4,
        )

    def _build_chunk_metadata(
        self,
        prefill_query_start_loc: torch.Tensor,
        prefill_query_start_loc_cpu: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

        if self.gdn_prefill_backend == "cutedsl":
            from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                prepare_metadata_cutedsl,
            )

            assert prefill_query_start_loc is not None
            assert prefill_query_start_loc_cpu is not None
            total_tokens = int(prefill_query_start_loc_cpu[-1].item())
            return prepare_metadata_cutedsl(
                prefill_query_start_loc,
                total_tokens,
                FLA_CHUNK_SIZE,
            )

        # Only prefill batches use FLA chunk ops.
        # Pre-compute on CPU and async-copy to GPU to avoid
        # GPU→CPU sync (.tolist()) in prepare_chunk_indices.
        from vllm.third_party.flash_linear_attention.ops.index import (
            prepare_chunk_indices,
            prepare_chunk_offsets,
        )

        assert prefill_query_start_loc_cpu is not None
        return (
            async_tensor_h2d(
                prepare_chunk_indices(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                device=device,
            ),
            async_tensor_h2d(
                prepare_chunk_offsets(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                device=device,
            ),
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

        spec_sequence_masks_cpu: torch.Tensor | None = None
        spec_sequence_masks_np: np.ndarray | None = None
        non_spec_req_indices: torch.Tensor | None = None
        pure_spec = False
        stage_pure_spec = False
        spec_token_size = 0
        if not self.use_spec_decode or num_decode_draft_tokens_cpu is None:
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            num_decode_draft_tokens_np = num_decode_draft_tokens_cpu.numpy()
            spec_sequence_masks_np = num_decode_draft_tokens_np >= 0
            num_spec_decodes = int(np.count_nonzero(spec_sequence_masks_np))
            if (
                num_spec_decodes == 0
                or num_decode_draft_tokens_np[spec_sequence_masks_np].sum() == 0
            ):
                num_spec_decodes = 0
                spec_sequence_masks = None
                spec_sequence_masks_np = None
            else:
                spec_sequence_masks_cpu = torch.from_numpy(spec_sequence_masks_np)
                # Materialized lazily: the pure spec-decode CUDA-graph path
                # generates the device mask inside the staging kernel.
                spec_sequence_masks = spec_sequence_masks_cpu

        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            assert spec_sequence_masks_cpu is not None
            assert spec_sequence_masks_np is not None
            non_spec_sequence_masks_cpu = ~spec_sequence_masks_cpu
            non_spec_sequence_masks_np = ~spec_sequence_masks_np
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            query_lens_np = query_lens_cpu.numpy()

            # Use CPU values to avoid CPU-GPU sync
            non_spec_query_lens_np = query_lens_np[non_spec_sequence_masks_np]
            num_decodes = int(np.count_nonzero(non_spec_query_lens_np == 1))
            # Exclude zero-length padded sequences from prefill count.
            num_zero_len = int(np.count_nonzero(non_spec_query_lens_np == 0))
            num_prefills = non_spec_query_lens_np.size - num_decodes - num_zero_len
            num_decode_tokens = num_decodes
            num_prefill_tokens = int(non_spec_query_lens_np.sum()) - num_decode_tokens
            num_spec_decode_tokens = (
                int(query_lens_np.sum()) - num_prefill_tokens - num_decode_tokens
            )

            # num_decodes and num_spec_decodes are mutually exclusive.
            # Reclassify non-spec decodes as prefills when spec decodes
            # exist — the prefill kernel handles 1-token sequences with
            # initial state correctly, producing identical results.
            if num_decodes > 0 and num_spec_decodes > 0:
                num_prefills += num_decodes
                num_prefill_tokens += num_decode_tokens
                num_decodes = 0
                num_decode_tokens = 0

            assert num_accepted_tokens is not None
            pure_spec = num_prefills == 0 and num_decodes == 0
            if pure_spec:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    int(query_start_loc_cpu[-1]),
                )
                # Every real request is a spec decode and padding trails them,
                # so the request-indexed metadata is a plain prefix slice and
                # the spec token order is the identity. The CUDA-graph path
                # below stages everything with a single kernel launch.
                stage_pure_spec = (
                    self.use_full_cuda_graph
                    and num_spec_decodes <= self.decode_cudagraph_max_bs
                    and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
                )
                if stage_pure_spec:
                    spec_token_indx = None
                    non_spec_token_indx = None
                else:
                    spec_sequence_masks = async_tensor_h2d(
                        spec_sequence_masks_cpu, device=query_start_loc.device
                    )
                    spec_token_indx = torch.arange(
                        spec_token_size,
                        dtype=torch.int32,
                        device=query_start_loc.device,
                    )
                    non_spec_token_indx = torch.empty(
                        0, dtype=torch.int32, device=query_start_loc.device
                    )
                spec_state_indices_tensor = block_table_tensor[
                    :num_spec_decodes, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = None
                # Padded sequences are always at the back, so the first
                # num_spec_decodes + 1 entries of query_start_loc already
                # contain the correct cumulative token counts.
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
                num_accepted_tokens = num_accepted_tokens[:num_spec_decodes]
            else:
                spec_sequence_masks = async_tensor_h2d(
                    spec_sequence_masks_cpu, device=query_start_loc.device
                )
                # Request selectors are computed on the host and copied once so
                # no device-side mask indexing (nonzero -> sync) is needed.
                spec_req_indices = async_tensor_h2d(
                    np.flatnonzero(spec_sequence_masks_np).astype(np.int32),
                    device=query_start_loc.device,
                )
                non_spec_req_indices = async_tensor_h2d(
                    np.flatnonzero(non_spec_sequence_masks_np).astype(np.int32),
                    device=query_start_loc.device,
                )
                query_lens = query_start_loc[1:] - query_start_loc[:-1]
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks,
                    query_lens,
                    output_size=int(query_start_loc_cpu[-1]),
                )
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = block_table_tensor[
                    :, : self.num_spec + 1
                ].index_select(0, spec_req_indices)
                non_spec_state_indices_tensor = block_table_tensor[:, 0].index_select(
                    0, non_spec_req_indices
                )

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens.index_select(0, spec_req_indices),
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens.index_select(0, non_spec_req_indices),
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[non_spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )
                num_accepted_tokens = num_accepted_tokens.index_select(
                    0, spec_req_indices
                )

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        prefill_query_start_loc: torch.Tensor | None = None
        prefill_state_indices: torch.Tensor | None = None
        prefill_has_initial_state: torch.Tensor | None = None
        if num_prefills > 0:
            # In a mixed non-spec batch, decodes are peeled off to the recurrent
            # kernel (decode-first front slice), so build chunk metadata from the
            # rebased prefill-only cu_seqlens; otherwise use the full non-spec one.
            # _forward_core keys off the same condition, so they agree.
            if spec_sequence_masks is None and num_decodes > 0:
                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                assert non_spec_state_indices_tensor is not None
                prefill_query_start_loc = (
                    non_spec_query_start_loc[num_decodes:] - num_decode_tokens
                )
                prefill_query_start_loc_cpu = (
                    non_spec_query_start_loc_cpu[num_decodes:] - num_decode_tokens
                )
                prefill_state_indices = non_spec_state_indices_tensor[num_decodes:]
            else:
                prefill_query_start_loc = non_spec_query_start_loc
                prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu
                prefill_state_indices = non_spec_state_indices_tensor

            chunk_indices, chunk_offsets = self._build_chunk_metadata(
                prefill_query_start_loc,
                prefill_query_start_loc_cpu,
                query_start_loc.device,
            )

        if num_prefills > 0:
            context_lens_tensor = m.compute_num_computed_tokens()
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks_cpu is not None:
                has_initial_state = has_initial_state.index_select(
                    0, non_spec_req_indices
                )
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
            if spec_sequence_masks is None and num_decodes > 0:
                prefill_has_initial_state = has_initial_state[num_decodes:]
            else:
                prefill_has_initial_state = has_initial_state
        else:
            has_initial_state = None

        # Function code counted on either presency non-spec decode or spec decode,
        # but not both.
        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        # Prepare per-request tensors for cudagraph. m.num_actual_tokens is
        # token-padded for FULL graph replay, but the GDN state/query/accepted
        # metadata below is indexed by request.
        batch_size = m.num_reqs

        if num_spec_decodes > 0 and pure_spec and stage_pure_spec:
            assert num_accepted_tokens is not None
            self._stage_pure_spec_metadata(
                block_table_tensor,
                query_start_loc,
                num_accepted_tokens,
                num_spec_decodes,
                batch_size,
                spec_token_size,
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_token_indx = self.spec_token_indx[:spec_token_size]
            non_spec_token_indx = self.non_spec_token_indx[:0]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_state_indices=prefill_state_indices,
            prefill_has_initial_state=prefill_has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        return attn_metadata

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = torch.diff(m.query_start_loc_cpu).sub_(1)
        assert num_decode_draft_tokens_cpu.shape == num_accepted_tokens.shape

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)
