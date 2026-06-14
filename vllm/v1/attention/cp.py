# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Context Parallel (PCP/DCP) shared infrastructure.

Single home for CP *layout* math and *recipe* helpers that are independent of
any attention backend. It deliberately contains:

* :class:`CPContext`    - frozen value object with static CP facts (groups,
  ranks, sizes, policy). Replaces the loose ``pcp_world_size`` /
  ``dcp_world_size`` attributes scattered across backends.
* :class:`CPBatchLayout` - per-batch, **layout-level** metadata. MUST NOT carry
  backend/kernel-specific index tensors (q_head_idx, kv_nomask_idx, ...); those
  are derived inside each backend from a ``CPBatchLayout``.
* layout construction (:func:`pcp_build_layout`) and recipe helpers
  (:func:`pcp_allgather_kv_restore`, :func:`pcp_logits_owner_rank`,
  :func:`pcp_gather_logits`).

Communication goes through ``vllm.distributed`` groups; numerical primitives
(``merge_attn_states``, ``dcp_a2a_lse_reduce``, ...) live in
``vllm.v1.attention.ops`` and are consumed, not re-implemented, here.

Scope note: the layout describes how the **prefill** sequence is zigzag-sharded
across PCP ranks. Decode is an inert, replicated-cache path (not described
here); a mixed batch composes the prefill layout on its prefill segment with
the normal decode path on its decode segment.

Design reference: ``learning_doc/RFC-vLLM-CP-Design-Final.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.parallel_state import GroupCoordinator


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class CPShardingPolicy:
    """How the sequence / KV ownership is partitioned across CP ranks.

    Only names live here; arithmetic is in :func:`pcp_build_layout` and the
    backend recipes. A model registers its policy via :class:`CPContext`; that
    is the only topology-specific knob.
    """

    NONE = "none"
    ZIGZAG = "zigzag"  # DualChunkSwap; dense prefill load balance
    CONTINUOUS = "continuous"  # hybrid / linear-friendly natural order
    ROUNDROBIN = "roundrobin"  # DCP KV interleave; sparse indexer
    HEADSPLIT = "headsplit"  # pure-linear: split heads, full seq


# ---------------------------------------------------------------------------
# CPContext
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CPContext:
    """Frozen value object with static CP runtime facts.

    A *carrier*, not an orchestrator: it does not mutate token scheduling,
    does not build backend-specific indices, does not own communication
    primitives, and does not dispatch backend strategy.
    """

    pcp_size: int
    pcp_rank: int
    dcp_size: int
    dcp_rank: int
    pcp_group: GroupCoordinator | None
    dcp_group: GroupCoordinator | None
    interleave: int  # == cp_kv_cache_interleave_size
    dcp_comm_backend: str  # "ag_rs" | "a2a"
    policy: str = CPShardingPolicy.ZIGZAG

    @property
    def total_cp_size(self) -> int:
        return self.pcp_size * self.dcp_size

    @property
    def total_cp_rank(self) -> int:
        return self.pcp_rank * self.dcp_size + self.dcp_rank

    @property
    def cp_active(self) -> bool:
        return self.total_cp_size > 1

    @property
    def pcp_active(self) -> bool:
        return self.pcp_size > 1

    @classmethod
    def build(cls, vllm_config: VllmConfig) -> CPContext:
        """Build from a VllmConfig, reading process groups with the same
        try/except fallbacks as ``AttentionImplBase.__new__``."""
        from vllm.distributed.parallel_state import get_dcp_group, get_pcp_group

        pc = vllm_config.parallel_config
        try:
            pcp_group = get_pcp_group()
            pcp_size = pcp_group.world_size
            pcp_rank = pcp_group.rank_in_group
        except AssertionError:
            # Groups may be uninitialized in unit tests.
            pcp_group = None
            pcp_size = pc.prefill_context_parallel_size
            pcp_rank = 0
        try:
            dcp_group = get_dcp_group()
            dcp_size = dcp_group.world_size
            dcp_rank = dcp_group.rank_in_group
        except AssertionError:
            dcp_group = None
            dcp_size = pc.decode_context_parallel_size
            dcp_rank = 0
        return cls(
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            pcp_group=pcp_group,
            dcp_group=dcp_group,
            interleave=pc.cp_kv_cache_interleave_size,
            dcp_comm_backend=pc.dcp_comm_backend,
            policy=CPShardingPolicy.ZIGZAG,
        )


# ---------------------------------------------------------------------------
# CPBatchLayout
# ---------------------------------------------------------------------------


@dataclass
class CPBatchLayout:
    """Per-batch, layout-level CP metadata for the **prefill** segment.

    Rule: **layout-level only**. If only one backend/kernel can interpret a
    tensor, it must NOT live here -- it is derived inside that backend.

    All index tensors index into the prefill token sub-buffer for this rank,
    unless noted. ``restore_idx`` indexes into the rank-concatenated
    all-gathered buffer.
    """

    active: bool
    policy: str
    pcp_size: int
    pcp_rank: int

    # Per-request padded token count this rank processes (head+tail chunks,
    # INCLUDING pad positions, so every rank gets an equal share for balanced
    # all-gather). = padded_len_i / pcp_size per request.
    local_query_lens_cpu: np.ndarray  # [num_reqs] int32
    # Cumulative offsets for this rank's local tokens ([num_reqs + 1]).
    local_query_start_loc_cpu: np.ndarray  # [num_reqs + 1] int32

    # Padded token count this rank processes (== sum local_query_lens, balanced
    # across ranks). Includes pad positions; use ``unpad_mask`` to drop them.
    num_local_tokens_padded: int

    # Total padded tokens across all ranks (== pcp_size * num_local_tokens_padded).
    # Length of the rank-concatenated all-gathered buffer.
    total_tokens_padded: int

    # Restore natural padded order from the rank-concatenated all-gathered
    # buffer: ``gathered.index_select(0, restore_idx)`` -> natural-order full
    # (padded) prefill. Length == total_tokens_padded.
    restore_idx_cpu: np.ndarray  # [total_tokens_padded] int32

    # Marks real (non-padded) tokens in the RESTORED natural padded buffer.
    # Apply after restore to drop pad rows: ``full_padded[unpad_mask]``.
    # Length == total_tokens_padded.
    unpad_mask_cpu: np.ndarray  # [total_tokens_padded] bool

    # Original (pre-split) per-request query lengths, for reference.
    query_lens_full_cpu: np.ndarray  # [num_reqs] int32


# ---------------------------------------------------------------------------
# Zigzag layout math (per request)
# ---------------------------------------------------------------------------


def _pad_to_multiple(x: int, m: int) -> int:
    if m <= 1:
        return x
    return ((x + m - 1) // m) * m


def pcp_zigzag_padded_len(query_len: int, pcp_size: int) -> int:
    """Padded per-request query length for a balanced zigzag split.

    Padded to a multiple of ``2 * pcp_size`` so all ranks receive an equal
    number of tokens (head + tail chunks of equal size).
    """
    return _pad_to_multiple(query_len, 2 * pcp_size)


def pcp_zigzag_rank_positions(
    query_len: int, pcp_size: int, pcp_rank: int
) -> tuple[np.ndarray, int]:
    """Global positions [0, padded_len) owned by ``pcp_rank`` for one request.

    Rank ``r`` owns the *head* chunk ``[r*c, (r+1)*c)`` and the *tail* chunk
    ``[(2P-1-r)*c, (2P-r)*c)`` where ``c = padded_len / (2*pcp_size)``.

    Returns:
        (sorted_positions, padded_len); positions are within the request's
        padded token range.
    """
    assert pcp_size >= 1 and 0 <= pcp_rank < pcp_size, (pcp_size, pcp_rank)
    padded_len = pcp_zigzag_padded_len(query_len, pcp_size)
    chunk = padded_len // (2 * pcp_size)
    head = np.arange(pcp_rank * chunk, (pcp_rank + 1) * chunk, dtype=np.int32)
    tail_start = (2 * pcp_size - 1 - pcp_rank) * chunk
    tail = np.arange(tail_start, tail_start + chunk, dtype=np.int32)
    positions = np.concatenate([head, tail])
    positions.sort()
    return positions, padded_len


def pcp_zigzag_owner_rank(position: int, padded_len: int, pcp_size: int) -> int:
    """Rank owning a (padded) position under zigzag.

    ``chunk = padded_len / (2*pcp_size)``; ``c = position // chunk``;
    ``owner = min(c, 2*pcp_size - 1 - c)``.
    """
    chunk = padded_len // (2 * pcp_size)
    if chunk == 0:
        return 0
    c = position // chunk
    return int(min(c, 2 * pcp_size - 1 - c))


def pcp_zigzag_real_chunks(
    q_len: int, pcp_size: int, pcp_rank: int
) -> list[tuple[int, int]]:
    """Real (non-padded) head+tail chunk ``(start, len)`` ranges within a
    request of ``q_len`` tokens owned by ``pcp_rank`` under zigzag.

    Unlike :func:`pcp_zigzag_rank_positions` (which returns padded positions
    for the balanced all-gather of the sequence-parallel mode), this returns
    only the *real* token ranges (clipped to ``[0, q_len)``). It is used by the
    attention-only PCP path, where each rank computes attention for its real
    Q-shard and outputs are assembled via all-reduce (no padding/balancing
    needed).

    ``chunk = ceil(q_len / (2*pcp_size))``; the ``2*pcp_size`` chunks of size
    ``chunk`` are assigned so rank ``r`` owns chunk ``r`` (head) and chunk
    ``(2*pcp_size-1-r)`` (tail). The returned ranges across all ranks partition
    ``[0, q_len)`` disjointly.
    """
    c = (q_len + 2 * pcp_size - 1) // (2 * pcp_size)
    if c == 0:
        return []
    chunks: list[tuple[int, int]] = []
    for j in (pcp_rank, 2 * pcp_size - 1 - pcp_rank):
        start = j * c
        end = min((j + 1) * c, q_len)
        if start < end:
            chunks.append((start, end - start))
    return chunks


# ---------------------------------------------------------------------------
# Batch layout construction
# ---------------------------------------------------------------------------


def pcp_build_layout(
    query_lens: np.ndarray,
    pcp_size: int,
    pcp_rank: int,
    policy: str = CPShardingPolicy.ZIGZAG,
) -> CPBatchLayout:
    """Build the per-batch :class:`CPBatchLayout` (prefill segment) for a rank.

    The layout is computed in per-request padded-position space and flattened
    assuming each request occupies ``padded_len`` contiguous slots in the
    prefill token buffer (request ``i`` at offset ``sum_{j<i} padded_len_j``).
    The caller (runner) pads buffers accordingly and uses ``unpad_mask`` to
    drop pad rows.

    Args:
        query_lens: [num_reqs] int32, prefill tokens scheduled per request.
        pcp_size, pcp_rank: PCP group topology.
        policy: only ZIGZAG is implemented for the first cut.
    """
    assert policy == CPShardingPolicy.ZIGZAG, f"policy {policy} not implemented"
    num_reqs = int(len(query_lens))
    active = pcp_size > 1

    padded_lens = np.empty(num_reqs, dtype=np.int32)
    req_offsets = np.empty(num_reqs, dtype=np.int32)
    running = 0
    for i in range(num_reqs):
        padded_lens[i] = pcp_zigzag_padded_len(int(query_lens[i]), pcp_size)
        req_offsets[i] = running
        running += int(padded_lens[i])
    total_tokens_padded = int(running)

    # This rank's kept (padded) token count per request = padded_len / pcp_size
    # (the head+tail chunks, balanced across ranks). Positions are recomputed
    # by the runner via :func:`pcp_zigzag_rank_positions`.
    local_query_lens = (padded_lens // pcp_size).astype(np.int32)
    local_query_start_loc = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(local_query_lens, out=local_query_start_loc[1:])
    num_local_tokens_padded = int(local_query_start_loc[-1])

    # unpad_mask over the RESTORED natural padded buffer: position t in request
    # i is real iff t < query_len_i.
    unpad_segments = [
        np.arange(int(padded_lens[i]), dtype=np.int32) < int(query_lens[i])
        for i in range(num_reqs)
    ]
    unpad_mask = np.concatenate(unpad_segments) if num_reqs else np.zeros(0, dtype=bool)

    restore_idx = _build_restore_idx(padded_lens, req_offsets, pcp_size)

    return CPBatchLayout(
        active=active,
        policy=policy,
        pcp_size=pcp_size,
        pcp_rank=pcp_rank,
        local_query_lens_cpu=local_query_lens,
        local_query_start_loc_cpu=local_query_start_loc,
        num_local_tokens_padded=num_local_tokens_padded,
        total_tokens_padded=total_tokens_padded,
        restore_idx_cpu=restore_idx,
        unpad_mask_cpu=unpad_mask,
        query_lens_full_cpu=np.asarray(query_lens, dtype=np.int32),
    )


def _build_restore_idx(
    padded_lens: np.ndarray,
    req_offsets: np.ndarray,
    pcp_size: int,
) -> np.ndarray:
    """Restore natural padded order from the rank-concatenated all-gathered
    buffer.

    The all-gathered buffer is ``[rank0_locals | rank1_locals | ...]``; within a
    rank, locals follow per-request order (request 0's head+tail chunks sorted,
    then request 1's, ...). For every *padded* position we record its owner
    rank; concatenating per-rank yields the gathered order.
    ``restore_idx[n]`` is the gathered slot of the n-th token in natural
    (increasing buffer-index) order, so
    ``gathered.index_select(0, restore_idx)`` recovers natural padded order.
    """
    if pcp_size <= 1 or len(padded_lens) == 0:
        return np.zeros(0, dtype=np.int32)
    per_rank_locals: list[list[int]] = [[] for _ in range(pcp_size)]
    for i in range(len(padded_lens)):
        padded_len = int(padded_lens[i])
        off = int(req_offsets[i])
        chunk = padded_len // (2 * pcp_size)
        for t in range(padded_len):  # all padded positions
            c = t // chunk if chunk > 0 else 0
            owner = int(min(c, 2 * pcp_size - 1 - c))
            per_rank_locals[owner].append(off + t)
    gathered = np.array(
        [idx for r in range(pcp_size) for idx in per_rank_locals[r]], dtype=np.int32
    )
    # gathered[order] is sorted ascending (natural buffer order); restore_idx
    # = order maps natural position n -> gathered slot order[n].
    return np.argsort(gathered, kind="stable").astype(np.int32)


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------


def pcp_allgather_kv_restore(
    ctx: CPContext,
    restore_idx: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-gather per-rank K/V across PCP and restore natural order.

    Mode-1 prefill primitive: each rank computed K/V for its zigzag shard;
    all-gather concatenates ranks along dim=0; ``restore_idx`` undoes the
    rank-major / head-tail interleaving to recover natural sequence order.

    Args:
        ctx: CP context (uses ``ctx.pcp_group``).
        restore_idx: [num_gathered_tokens] index into the gathered buffer
            (from :attr:`CPBatchLayout.restore_idx_cpu`).
        key, value: [num_local_tokens, num_kv_heads, head_size] this rank's
            shard K/V.

    Returns:
        (full_key, full_value) in natural order, each
        [num_gathered_tokens, num_kv_heads, head_size].
    """
    assert ctx.pcp_active, "pcp_allgather_kv_restore called with pcp_size == 1"
    assert ctx.pcp_group is not None
    kv = torch.cat([key, value], dim=-1)
    gathered = ctx.pcp_group.all_gather(kv, dim=0)
    full = gathered.index_select(0, restore_idx.to(gathered.device))
    head_size = key.shape[-1]
    return full[..., :head_size], full[..., head_size:]


def pcp_logits_owner_rank(
    query_lens: np.ndarray,
    pcp_size: int,
) -> np.ndarray:
    """Owner rank (per request) of each request's last prefill token.

    The last token of a prefill request lives on exactly one PCP rank. Returns
    ``[num_reqs]`` int32 owner ranks.
    """
    owners = np.zeros(len(query_lens), dtype=np.int32)
    if pcp_size <= 1:
        return owners
    for i, ql in enumerate(query_lens):
        padded_len = pcp_zigzag_padded_len(int(ql), pcp_size)
        owners[i] = pcp_zigzag_owner_rank(int(ql) - 1, padded_len, pcp_size)
    return owners


def pcp_gather_logits(
    ctx: CPContext,
    local_logits: torch.Tensor,
    owner_rank: torch.Tensor,
    num_requests: int,
) -> torch.Tensor:
    """Gather sharded prefill logits across PCP and pick the owner row per req.

    Each rank computes logits only for requests whose last token it owns (other
    rows should be zeroed). All-gather along dim=0, then select the owner
    rank's row for each request.

    Args:
        ctx: CP context.
        local_logits: [num_requests, vocab] this rank's rows.
        owner_rank: [num_requests] int owner rank per request.
        num_requests: ``== local_logits.shape[0]``.

    Returns:
        [num_requests, vocab] full logits.
    """
    if not ctx.pcp_active:
        return local_logits
    assert ctx.pcp_group is not None
    gathered = ctx.pcp_group.all_gather(local_logits, dim=0)
    # gathered: [pcp_size * num_requests, vocab] as [rank0_rows | rank1_rows].
    vocab = gathered.shape[-1]
    gathered = gathered.reshape(ctx.pcp_size, num_requests, vocab)
    owner_rank = owner_rank.to(gathered.device, non_blocking=True)
    req_idx = torch.arange(num_requests, device=gathered.device)
    return gathered[owner_rank, req_idx]


def cp_merge_decode_out_lse(
    ctx: CPContext,
    attn_out: torch.Tensor,
    softmax_lse: torch.Tensor,
    head_size: int,
) -> torch.Tensor:
    """Merge per-rank decode attention (out, lse) across PCP and DCP (Mode-2).

    Faithful port of vLLM-Ascend's ``_process_attn_out_lse`` +
    ``_npu_attention_update``, using a pure-torch N-way logsumexp merge
    (Ascend's ``_update_out_and_lse``) in place of the CANN
    ``npu_attention_update`` op.

    Flow::

        cat(out, lse) -> [B, H_total, D+1]
        if dcp>1: all_to_all over heads (DCP)
        if pcp>1: all_gather over seq   (PCP) -> [pcp*B, H_total, D+1]
        reshape [pcp, B, dcp, H, D+1] -> [N=pcp*dcp, B, H, D+1]
        N-way online-softmax merge      -> [B, H, D]

    Each rank computed attention of (its Q) against (its local KV shard); the
    per-rank partial (out, lse) are combined into the correct full output via
    logsumexp weighting.

    Args:
        ctx: CP context (uses ``pcp_group`` and ``dcp_group``).
        attn_out: ``[B, H_total, D]`` where ``H_total`` includes DCP-gathered
            heads when ``dcp>1``.
        softmax_lse: ``[B, H_total]`` or ``[B, H_total, 1]``.
        head_size: ``D``.

    Returns:
        ``[B, H, D]`` merged output, where ``H = H_total // dcp_size``.
    """
    import torch.distributed as dist

    pcp_size = ctx.pcp_size
    dcp_size = ctx.dcp_size
    out = attn_out.to(torch.float32)
    lse = softmax_lse.to(torch.float32)
    if lse.dim() == 2:
        lse = lse.unsqueeze(-1)
    attn_out_lse = torch.cat([out, lse], dim=-1)  # [B, H_total, D+1]

    if dcp_size > 1:
        assert ctx.dcp_group is not None  # dcp_size>1 implies the group exists
        # head-dim all-to-all across the DCP group: [B, H_total, D+1] ->
        # [H_total, D+1, B] -> a2a -> [H_total, D+1, B] -> [B, H_total, D+1].
        attn_out_lse = attn_out_lse.permute(1, 2, 0).contiguous()
        recv = torch.empty_like(attn_out_lse)
        dist.all_to_all_single(recv, attn_out_lse, group=ctx.dcp_group.device_group)
        attn_out_lse = recv.permute(2, 0, 1).contiguous()

    if pcp_size > 1:
        assert ctx.pcp_group is not None  # pcp_size>1 implies the group exists
        # seq-dim all-gather across the PCP group: -> [pcp*B, H_total, D+1].
        attn_out_lse = ctx.pcp_group.all_gather(attn_out_lse.contiguous(), dim=0)

    b_total, h_total, d_plus_1 = attn_out_lse.shape
    seq = b_total // pcp_size if pcp_size > 1 else b_total
    heads = h_total // dcp_size if dcp_size > 1 else h_total
    assert d_plus_1 == head_size + 1, (d_plus_1, head_size)

    # [pcp, seq, dcp, heads, D+1] -> [N=pcp*dcp, seq, heads, D+1].
    x = attn_out_lse.view(pcp_size, seq, dcp_size, heads, d_plus_1)
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, seq, heads, d_plus_1)
    out_flat = x[..., :head_size]  # [N, seq, heads, D]
    lse_flat = x[..., head_size:]  # [N, seq, heads, 1]

    # N-way online-softmax merge (Ascend _update_out_and_lse).
    lse_final = torch.logsumexp(lse_flat, dim=0)  # [seq, heads, 1]
    out_final = torch.sum(
        torch.exp(lse_flat - lse_final) * out_flat, dim=0
    )  # [seq, heads, D]
    return out_final.to(attn_out.dtype)
