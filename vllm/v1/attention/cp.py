# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Context Parallel (PCP/DCP) shared infrastructure.

The FlashAttention PCP path (:meth:`FlashAttentionImpl._forward_pcp`) consumes:

* :class:`CPContext` -- frozen value object carrying the static CP facts
  (sizes / ranks / process groups).
* :func:`cp_merge_decode_out_lse` -- cross-PCP (+DCP) LSE merge of the per-rank
  partial ``(out, lse)`` produced when each rank attends its query against its
  local slice of a ``total_cp = pcp*dcp``-sharded KV cache.

Communication goes through ``vllm.distributed`` groups; the N-way logsumexp
merge is a pure-torch port of vLLM-Ascend's ``_process_attn_out_lse`` +
``_update_out_and_lse``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator


@dataclass(frozen=True)
class CPContext:
    """Frozen value object with static CP runtime facts.

    A *carrier*, not an orchestrator: it carries the CP sizes/ranks/groups for
    the merge recipe; it does not dispatch strategy, build backend-specific
    indices, or own communication primitives.
    """

    pcp_size: int
    pcp_rank: int
    dcp_size: int
    dcp_rank: int
    pcp_group: GroupCoordinator | None
    dcp_group: GroupCoordinator | None
    interleave: int  # == cp_kv_cache_interleave_size
    dcp_comm_backend: str  # "ag_rs" | "a2a"

    @property
    def total_cp_size(self) -> int:
        return self.pcp_size * self.dcp_size

    @property
    def total_cp_rank(self) -> int:
        return self.pcp_rank * self.dcp_size + self.dcp_rank


def cp_merge_decode_out_lse(
    ctx: CPContext,
    attn_out: torch.Tensor,
    softmax_lse: torch.Tensor,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge per-rank decode/context attention ``(out, lse)`` across PCP and DCP.

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
    per-rank partial ``(out, lse)`` are combined into the correct full output via
    logsumexp weighting.

    Args:
        ctx: CP context (uses ``pcp_group`` and ``dcp_group``).
        attn_out: ``[B, H_total, D]`` where ``H_total`` includes DCP-gathered
            heads when ``dcp>1``.
        softmax_lse: ``[B, H_total]`` or ``[B, H_total, 1]``.
        head_size: ``D``.

    Returns:
        ``(out, lse)`` where ``out`` is ``[B, H, D]`` (``H = H_total // dcp_size``)
        and ``lse`` is the matching ``[B, H, 1]`` logsumexp. Both are needed by the
        caller to LSE-merge the context result with the local query-only result.
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
    return out_final.to(attn_out.dtype), lse_final
