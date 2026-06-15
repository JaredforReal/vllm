# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill Context Parallel (PCP) token slicing -- faithful port of vLLM-Ascend's
``vllm_ascend/worker/pcp_utils.py`` (GQA-only subset).

PCP splits each prefill request's tokens across PCP ranks in a head-tail
(DualChunkSwap / zigzag) style for balanced load, and replicates decode tokens.
This module contains the **pure CPU/numpy math** so it can be unit-tested without
GPU; the :class:`PCPManager` wrapper adds the per-step state the runner consumes.

Reference (ascend): ``update_tokens_for_pcp`` (~pcp_utils.py:557),
``get_current_rank_positions`` (~:604), ``pcp_allgather_restore_idx`` (~:639).
Hybrid-attention / MTP / chunked-context branches from ascend are intentionally
NOT ported (out of scope for GQA prefill+decode+mixed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cumsum_and_arange(
    num_tokens: np.ndarray,
    arange_np: np.ndarray,
    cumsum_dtype: np.dtype | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative sum and batched arange. Verbatim port of ascend's
    ``_get_cumsum_and_arange``.

    E.g. ``[2, 5, 3]`` -> ``([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])``.
    """
    cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
    total_num_tokens = int(cu_num_tokens[-1])
    cumsums_offsets: np.ndarray = np.repeat(cu_num_tokens - num_tokens, num_tokens)
    arange = arange_np[:total_num_tokens] - cumsums_offsets
    return cu_num_tokens, arange


@dataclass
class PCPSplitResult:
    """Output of :func:`update_tokens_for_pcp` for one PCP rank."""

    # Per-request token count THIS rank processes (post-split): decode reqs keep
    # their original count (replicated), prefill reqs are halved.
    pcp_tokens: np.ndarray
    # Flattened per-request *relative* positions for this rank's tokens (added to
    # ``num_computed_tokens[req]`` by the runner to get absolute positions).
    positions: np.ndarray
    # Padded token count per request (alignment to 2*pcp_world_size for prefill;
    # original*pcp_world_size for decode).
    num_padded_scheduled_tokens: np.ndarray
    # Pads added per request (padded - original).
    num_pcp_pads: np.ndarray
    # Mask (len = total padded tokens across all ranks) marking real vs padded
    # slots in the all-gather buffer.
    pcp_unpad_mask: np.ndarray
    # ``argsort`` of the concatenated per-rank global positions: after
    # ``all_gather`` concatenates rank chunks, ``index_select(buf, restore_idx)``
    # recovers natural token order.
    pcp_allgather_restore_idx: np.ndarray


@dataclass
class PCPPrefillMetadata:
    """Index set consumed by the PCP prefill attention (per rank).

    All index arrays index into:
      - ``q_*``: this rank's local prefill-Q tensor.
      - ``kv_*``: the all-gathered+restored full prefill-KV tensor.

    Port of ascend ``AscendPCPMetadata`` (common_cp.py:9-44), GQA prefill subset.
    """

    # Local-Q row indices of the head / tail chunks per prefill request.
    q_head_idx: np.ndarray
    q_tail_idx: np.ndarray
    # Full-KV indices: prefix before this rank's chunk (nomask) and the chunk
    # itself (causal mask), for the head and tail Q respectively.
    kv_with_q_head_nomask_idx: np.ndarray
    kv_with_q_head_mask_idx: np.ndarray
    kv_with_q_tail_nomask_idx: np.ndarray
    kv_with_q_tail_mask_idx: np.ndarray
    # Cumulative Q/KV lengths for flash_attn_varlen (cu_seqlens style, as a
    # Python list ending in the total -- callers prepend 0).
    attn_mask_seqlens: list  # cu_seqlens for the mask (causal) call, per req
    head_attn_nomask_seqlens: list  # cu_seqlens_k for the head nomask call
    tail_attn_nomask_seqlens: list  # cu_seqlens_k for the tail nomask call
    # argsort of cat(q_head_idx, q_tail_idx): re-interleaves [head_out, tail_out]
    # back into natural local-Q order.
    q_full_idx: np.ndarray


def update_tokens_for_pcp(
    num_scheduled_tokens: np.ndarray,
    pcp_world_size: int,
    pcp_world_rank: int,
    num_decode_reqs: int,
    num_decode_tokens: int,
    arange_np: np.ndarray,
) -> PCPSplitResult:
    """Head-tail (DualChunkSwap) PCP split for one rank.

    Faithful port of ascend ``PCPManager.update_tokens_for_pcp`` (pcp_utils.py:557).

    Args:
        num_scheduled_tokens: per-request scheduled token counts, ordered
            **decode requests first, then prefill requests** (ascend layout).
        pcp_world_size / pcp_world_rank: PCP topology.
        num_decode_reqs: number of leading decode requests in the array.
        num_decode_tokens: total tokens across those decode requests.
        arange_np: a large ``np.arange`` scratch buffer (ascend passes
            ``self.arange_np``).

    Returns:
        :class:`PCPSplitResult` for ``pcp_world_rank``.
    """
    num_reqs = int(num_scheduled_tokens.shape[0])

    # DualChunkSwap requires alignment to a multiple of (2 * pcp_world_size).
    num_padded_scheduled_tokens = np.ceil(
        num_scheduled_tokens / (2 * pcp_world_size)
    ).astype(np.int32) * (2 * pcp_world_size)
    # PCP does not split decode requests; duplicate their tokens across ranks.
    num_padded_scheduled_tokens[:num_decode_reqs] = (
        num_scheduled_tokens[:num_decode_reqs] * pcp_world_size
    )
    num_pcp_pads = num_padded_scheduled_tokens - num_scheduled_tokens

    cu_padded_tokens, pcp_padded_arange = cumsum_and_arange(
        num_padded_scheduled_tokens, arange_np
    )
    pcp_padded_tokens_length = int(pcp_padded_arange.shape[0])

    # Real (unpadded) token mask within the padded all-gather buffer.
    pcp_unpad_mask = pcp_padded_arange < np.repeat(
        num_scheduled_tokens, num_padded_scheduled_tokens
    )
    # Decode slots: only rank-0's copy is real, the rest are padding.
    if num_decode_tokens > 0:
        unpad_mask_decode = pcp_unpad_mask[: num_decode_tokens * pcp_world_size]
        unpad_mask_decode = unpad_mask_decode.reshape([-1, pcp_world_size])
        unpad_mask_decode[:, 0] = True
        unpad_mask_decode[:, 1:] = False

    pcp_tokens = num_padded_scheduled_tokens // pcp_world_size
    # Head/tail chunk size per request: prefill is split in two; decode keeps full.
    pcp_chunk_sizes = (pcp_tokens // 2).clip(min=1)
    pcp_chunk_sizes[:num_decode_reqs] = pcp_tokens[:num_decode_reqs]

    _, pcp_arange = cumsum_and_arange(pcp_tokens, arange_np)
    _, pcp_chunk_arange = cumsum_and_arange(pcp_chunk_sizes, arange_np)
    pcp_head_chunk_mask = pcp_arange < np.repeat(pcp_chunk_sizes, pcp_tokens)

    def get_current_rank_positions(
        positions_start_loc: int | np.ndarray, rank: int
    ) -> np.ndarray:
        positions: np.ndarray = np.zeros(len(pcp_head_chunk_mask), dtype=np.int32)
        head_start_loc = positions_start_loc + rank * pcp_chunk_sizes
        tail_start_loc = (
            positions_start_loc + (2 * pcp_world_size - rank - 1) * pcp_chunk_sizes
        )
        positions[pcp_head_chunk_mask] = pcp_chunk_arange + np.repeat(
            head_start_loc, pcp_chunk_sizes
        )
        positions[~pcp_head_chunk_mask] = (
            pcp_chunk_arange[num_decode_tokens:]
            + np.repeat(tail_start_loc, pcp_chunk_sizes)[num_decode_tokens:]
        )
        return positions

    # Per-rank model positions: head/tail chunks, relative (start_loc=0).
    positions = get_current_rank_positions(0, pcp_world_rank)
    # Decode positions are the natural arange regardless of PCP (replicated).
    if num_decode_reqs > 0:
        positions[:num_decode_tokens] = cumsum_and_arange(
            num_scheduled_tokens[:num_decode_reqs], arange_np
        )[1]

    # Restore index: global positions across all ranks, argsorted.
    padded_pos_start_loc = np.roll(cu_padded_tokens, 1)
    padded_pos_start_loc[0] = 0
    all_positions = np.concatenate(
        [
            get_current_rank_positions(padded_pos_start_loc, rank_i)
            for rank_i in range(pcp_world_size)
        ]
    )
    pcp_allgather_restore_idx = all_positions.argsort().astype(np.int32)

    return PCPSplitResult(
        pcp_tokens=pcp_tokens[:num_reqs],
        positions=positions,
        num_padded_scheduled_tokens=num_padded_scheduled_tokens,
        num_pcp_pads=num_pcp_pads,
        pcp_unpad_mask=pcp_unpad_mask[:pcp_padded_tokens_length],
        pcp_allgather_restore_idx=pcp_allgather_restore_idx,
    )


def generate_prefill_metadata(
    prefill_query_lens: np.ndarray,
    pcp_world_size: int,
    pcp_world_rank: int,
) -> PCPPrefillMetadata:
    """Build the prefill-attention index set for one PCP rank.

    Faithful port of ascend ``PCPManager.generate_pcp_metadata`` (pcp_utils.py
    :1115-1218), GQA prefill subset (no hybrid-attn / chunked-context / MTP).

    Args:
        prefill_query_lens: per-rank query length of each prefill request
            (``= 2 * chunk_len``; decode requests already excluded).
        pcp_world_size / pcp_world_rank: PCP topology.
    """
    q_head_idx: list[int] = []
    q_tail_idx: list[int] = []
    kv_with_q_head_nomask_idx: list[int] = []
    kv_with_q_head_mask_idx: list[int] = []
    kv_with_q_tail_nomask_idx: list[int] = []
    kv_with_q_tail_mask_idx: list[int] = []
    chunk_seqlens: list[int] = []
    kv_with_q_head_nomask_seqlens: list[int] = []
    kv_with_q_tail_nomask_seqlens: list[int] = []

    q_req_offset = 0
    kv_req_offset = 0
    q_head_chunk_id = pcp_world_rank
    q_tail_chunk_id = pcp_world_size * 2 - 1 - pcp_world_rank

    for seq_len in prefill_query_lens:
        chunk_len = int(seq_len) // 2
        chunk_seqlens.append(chunk_len)

        q_head_idx.extend(range(q_req_offset, q_req_offset + chunk_len))
        kv_with_q_head_nomask_idx.extend(
            range(kv_req_offset, kv_req_offset + chunk_len * q_head_chunk_id)
        )
        kv_with_q_head_mask_idx.extend(
            range(
                kv_req_offset + chunk_len * q_head_chunk_id,
                kv_req_offset + chunk_len * (q_head_chunk_id + 1),
            )
        )
        kv_with_q_head_nomask_seqlens.append(chunk_len * q_head_chunk_id)

        q_tail_idx.extend(range(q_req_offset + chunk_len, q_req_offset + chunk_len * 2))
        kv_with_q_tail_nomask_idx.extend(
            range(kv_req_offset, kv_req_offset + chunk_len * q_tail_chunk_id)
        )
        kv_with_q_tail_mask_idx.extend(
            range(
                kv_req_offset + chunk_len * q_tail_chunk_id,
                kv_req_offset + chunk_len * (q_tail_chunk_id + 1),
            )
        )
        kv_with_q_tail_nomask_seqlens.append(chunk_len * q_tail_chunk_id)

        q_req_offset += int(seq_len)
        kv_req_offset += int(seq_len) * pcp_world_size

    q_head_arr = np.array(q_head_idx, dtype=np.int32)
    q_tail_arr = np.array(q_tail_idx, dtype=np.int32)
    q_full_idx = (
        np.concatenate([q_head_arr, q_tail_arr])
        .astype(np.float32)
        .argsort()
        .astype(np.int32)
    )

    attn_mask_seqlens = np.cumsum(np.array(chunk_seqlens, dtype=np.int32)).tolist()
    head_attn_nomask_seqlens = np.cumsum(
        np.array(kv_with_q_head_nomask_seqlens, dtype=np.int32)
    ).tolist()
    tail_attn_nomask_seqlens = np.cumsum(
        np.array(kv_with_q_tail_nomask_seqlens, dtype=np.int32)
    ).tolist()

    return PCPPrefillMetadata(
        q_head_idx=q_head_arr,
        q_tail_idx=q_tail_arr,
        kv_with_q_head_nomask_idx=np.array(kv_with_q_head_nomask_idx, dtype=np.int32),
        kv_with_q_head_mask_idx=np.array(kv_with_q_head_mask_idx, dtype=np.int32),
        kv_with_q_tail_nomask_idx=np.array(kv_with_q_tail_nomask_idx, dtype=np.int32),
        kv_with_q_tail_mask_idx=np.array(kv_with_q_tail_mask_idx, dtype=np.int32),
        attn_mask_seqlens=attn_mask_seqlens,
        head_attn_nomask_seqlens=head_attn_nomask_seqlens,
        tail_attn_nomask_seqlens=tail_attn_nomask_seqlens,
        q_full_idx=q_full_idx,
    )


class PCPManager:
    """Per-step PCP state for the model runner (wrap of the pure math above).

    Stateless across steps aside from topology; the runner calls
    :meth:`split` per ``_prepare_inputs`` and consumes the :class:`PCPSplitResult`.
    GPU-buffer plumbing (``copy_to_gpu`` etc.) is added when wiring the runner
    hooks (P4); the math itself lives in :func:`update_tokens_for_pcp`.
    """

    def __init__(
        self,
        pcp_world_size: int,
        pcp_world_rank: int,
        dcp_world_size: int = 1,
        dcp_world_rank: int = 0,
        cp_kv_cache_interleave_size: int = 1,
    ) -> None:
        self.pcp_world_size = pcp_world_size
        self.pcp_world_rank = pcp_world_rank
        self.dcp_world_size = dcp_world_size
        self.dcp_world_rank = dcp_world_rank
        self.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size

    def split(
        self,
        num_scheduled_tokens: np.ndarray,
        num_decode_reqs: int,
        num_decode_tokens: int,
        arange_np: np.ndarray,
    ) -> PCPSplitResult:
        """Slice this rank's tokens for one scheduling step."""
        return update_tokens_for_pcp(
            num_scheduled_tokens=num_scheduled_tokens,
            pcp_world_size=self.pcp_world_size,
            pcp_world_rank=self.pcp_world_rank,
            num_decode_reqs=num_decode_reqs,
            num_decode_tokens=num_decode_tokens,
            arange_np=arange_np,
        )

    def generate_prefill_metadata(
        self,
        prefill_query_lens: np.ndarray,
    ) -> PCPPrefillMetadata:
        """Build the prefill-attention index set for this rank.

        ``prefill_query_lens`` is the per-rank query length of each **prefill**
        request (= ``2 * chunk_len``; decode requests excluded). Port of ascend
        ``PCPManager.generate_pcp_metadata`` (pcp_utils.py:1115-1218), GQA-only.
        """
        return generate_prefill_metadata(
            prefill_query_lens=prefill_query_lens,
            pcp_world_size=self.pcp_world_size,
            pcp_world_rank=self.pcp_world_rank,
        )
