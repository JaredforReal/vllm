# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped sparse-attention prefill metadata for DSA (DeepSeek V3.2 / GLM-5.x).

Consecutive prefill queries select heavily overlapping top-k key sets, while the
per-token sparse MLA kernel re-gathers every selected KV row for each token. This
module groups ``GROUP`` consecutive query tokens of one request into a "row" of
``GROUP * num_heads`` pseudo-heads that attend over the *union* of the group's
top-k indices; a per-head-group bitmask restricts each token's heads to its own
top-k so the result is exactly the per-token sparse attention. The union list is
built with a per-group bitmap over key positions (so it comes out sorted), and
the mask marks each key's rank in the union.

Layouts (per group ``g``):
  - ``union[g, :ulen[g]]``: sorted request-local key positions, ``-1`` padded to
    ``union_pad``.
  - ``mask[g, b, t*16:(t+1)*16]``: 128-bit mask for union positions
    ``[b*128, (b+1)*128)``, bit ``j`` set iff key ``union[g, b*128+j]`` is in
    token ``t``'s top-k. This matches FlashMLA's ``head_group_mask`` layout.
  - ``tok_map[g*GROUP + t]``: flat query index of token ``t`` (``-1`` padding).
"""

from dataclasses import dataclass

import numpy as np
import torch

from vllm.triton_utils import tl, tldevice, triton

GROUP = 8
UNION_BLOCK = 128
# Each token contributes at most `topk` keys; the union is padded to the
# worst case so no device->host sync is needed (FlashMLA only iterates
# ceil(ulen / 128) blocks per group, so padding costs memory, not compute).


@dataclass
class GroupedSparsePrefillMeta:
    tok_map: torch.Tensor  # [num_groups * GROUP] int32, -1 = padding
    req_of_group: torch.Tensor  # [num_groups] int32
    union: torch.Tensor  # [num_groups, union_pad] int32 (request-local key pos)
    ulen: torch.Tensor  # [num_groups] int32
    mask: torch.Tensor  # [num_groups, union_pad // 128, 128] uint8
    num_groups: int
    union_pad: int


@triton.jit
def _group_bitmap_kernel(
    topk_ptr,
    topk_stride,
    tok_map_ptr,
    bitmap_ptr,
    WORDS: tl.constexpr,
    G: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    g = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    one = tl.full([BLOCK], 1, tl.int32)
    for t in range(G):
        tok = tl.load(tok_map_ptr + g * G + t).to(tl.int64)
        if tok >= 0:
            for c in range(0, TOPK, BLOCK):
                idx = tl.load(
                    topk_ptr + tok * topk_stride + c + offs,
                    mask=(c + offs) < TOPK,
                    other=-1,
                )
                valid = idx >= 0
                word = tl.where(valid, idx // 32, 0)
                bit = tl.where(valid, idx % 32, 0)
                tl.atomic_or(
                    bitmap_ptr + g * WORDS + word, one << bit, mask=valid, sem="relaxed"
                )


@triton.jit
def _group_compact_kernel(
    bitmap_ptr,
    union_ptr,
    ulen_ptr,
    cum_ptr,
    WORDS: tl.constexpr,
    BLOCK_W: tl.constexpr,
    U_PAD: tl.constexpr,
):
    g = tl.program_id(0).to(tl.int64)
    w = tl.arange(0, BLOCK_W)
    wmask = w < WORDS
    words = tl.load(bitmap_ptr + g * WORDS + w, mask=wmask, other=0)
    cnt = tldevice.popc(words)
    excl = tl.cumsum(cnt, axis=0) - cnt
    tl.store(cum_ptr + g * WORDS + w, excl, mask=wmask)
    total = tl.sum(cnt, axis=0)
    tl.store(ulen_ptr + g, total)
    # Emit the set bits in ascending key order: rank = excl[word] + popc(lower bits).
    for b in tl.static_range(32):
        has = ((words >> b) & 1) == 1
        below = tldevice.popc(words & ((1 << b) - 1))
        tl.store(
            union_ptr + g * U_PAD + excl + below,
            w * 32 + b,
            mask=has & wmask,
        )
    for c in range(0, U_PAD, BLOCK_W):
        p = c + w
        tl.store(
            union_ptr + g * U_PAD + p,
            tl.full([BLOCK_W], -1, tl.int32),
            mask=(p >= total) & (p < U_PAD),
        )


@triton.jit
def _group_mask_kernel(
    topk_ptr,
    topk_stride,
    tok_map_ptr,
    bitmap_ptr,
    cum_ptr,
    mask_ptr,
    WORDS: tl.constexpr,
    G: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    NBLK: tl.constexpr,
):
    g = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1)
    tok = tl.load(tok_map_ptr + g * G + t).to(tl.int64)
    if tok >= 0:
        offs = tl.arange(0, BLOCK)
        one = tl.full([BLOCK], 1, tl.int32)
        for c in range(0, TOPK, BLOCK):
            idx = tl.load(
                topk_ptr + tok * topk_stride + c + offs,
                mask=(c + offs) < TOPK,
                other=-1,
            )
            valid = idx >= 0
            word = tl.where(valid, idx // 32, 0)
            bit = tl.where(valid, idx % 32, 0)
            wv = tl.load(bitmap_ptr + g * WORDS + word, mask=valid, other=0)
            below = tldevice.popc(wv & ((one << bit) - 1))
            rank = tl.load(cum_ptr + g * WORDS + word, mask=valid, other=0) + below
            blk = rank // 128
            within = rank % 128
            # mask words: [NG, NBLK, G, 4 x uint32] (little endian = 16 B per group)
            widx = ((g * NBLK + blk) * G + t) * 4 + within // 32
            tl.atomic_or(
                mask_ptr + widx, one << (within % 32), mask=valid, sem="relaxed"
            )


def _make_tok_map(query_lens: list[int], group: int) -> tuple[np.ndarray, np.ndarray]:
    """Flat query index per (group, slot), -1 padded, and the request of each group."""
    tok_maps: list[np.ndarray] = []
    req_of_group: list[np.ndarray] = []
    start = 0
    for r, q_len in enumerate(query_lens):
        num_groups = -(-q_len // group)
        m = np.arange(num_groups * group, dtype=np.int32)
        m = np.where(m < q_len, m + start, -1)
        tok_maps.append(m)
        req_of_group.append(np.full(num_groups, r, dtype=np.int32))
        start += q_len
    return np.concatenate(tok_maps), np.concatenate(req_of_group)


def build_grouped_sparse_prefill(
    topk: torch.Tensor,
    query_lens: list[int],
    max_seq_len: int,
    group: int = GROUP,
) -> GroupedSparsePrefillMeta:
    """Group prefill queries and build union index lists plus head-group masks.

    Args:
        topk: [num_prefill_tokens, topk] int32 request-local key positions,
            ``-1`` padded, tokens ordered request by request.
        query_lens: prefill query length per request (same order).
        max_seq_len: upper bound of key positions (+1) across the requests.
        group: consecutive tokens per group.
    """
    assert topk.dtype == torch.int32 and topk.dim() == 2
    device = topk.device
    num_topk = topk.shape[1]
    tok_map_np, req_np = _make_tok_map(query_lens, group)
    num_groups = len(req_np)
    tok_map = torch.from_numpy(tok_map_np).to(device, non_blocking=True)
    req_of_group = torch.from_numpy(req_np).to(device, non_blocking=True)
    union_pad = triton.cdiv(group * num_topk, UNION_BLOCK) * UNION_BLOCK
    words = triton.cdiv(max_seq_len, 32)
    nblk = union_pad // UNION_BLOCK

    bitmap = torch.zeros(num_groups, words, dtype=torch.int32, device=device)
    cum = torch.empty_like(bitmap)
    union = torch.empty(num_groups, union_pad, dtype=torch.int32, device=device)
    ulen = torch.empty(num_groups, dtype=torch.int32, device=device)
    mask32 = torch.zeros(num_groups, nblk, group, 4, dtype=torch.int32, device=device)

    block = min(2048, triton.next_power_of_2(num_topk))
    _group_bitmap_kernel[(num_groups,)](
        topk,
        topk.stride(0),
        tok_map,
        bitmap,
        WORDS=words,
        G=group,
        TOPK=num_topk,
        BLOCK=block,
        num_warps=4,
    )
    block_w = triton.next_power_of_2(words)
    _group_compact_kernel[(num_groups,)](
        bitmap,
        union,
        ulen,
        cum,
        WORDS=words,
        BLOCK_W=block_w,
        U_PAD=union_pad,
        num_warps=8,
    )
    _group_mask_kernel[(num_groups, group)](
        topk,
        topk.stride(0),
        tok_map,
        bitmap,
        cum,
        mask32,
        WORDS=words,
        G=group,
        TOPK=num_topk,
        BLOCK=block,
        NBLK=nblk,
        num_warps=4,
    )
    mask = mask32.view(torch.uint8).view(num_groups, nblk, group * 16)
    if group * 16 != 128:
        # FlashMLA expects 8 x 16-byte group masks per key block.
        full = torch.zeros(num_groups, nblk, 128, dtype=torch.uint8, device=device)
        full[..., : group * 16] = mask
        mask = full
    return GroupedSparsePrefillMeta(
        tok_map=tok_map,
        req_of_group=req_of_group,
        union=union,
        ulen=ulen,
        mask=mask,
        num_groups=num_groups,
        union_pad=union_pad,
    )


def grouped_sparse_prefill_reference(
    topk: torch.Tensor, query_lens: list[int], group: int = GROUP
) -> GroupedSparsePrefillMeta:
    """Pure-torch reference of :func:`build_grouped_sparse_prefill` (tests)."""
    device = topk.device
    tok_map_np, req_np = _make_tok_map(query_lens, group)
    tok_map_l = tok_map_np.tolist()
    num_groups = len(req_np)
    num_topk = topk.shape[1]
    union_pad = triton.cdiv(group * num_topk, UNION_BLOCK) * UNION_BLOCK
    nblk = union_pad // UNION_BLOCK
    union = torch.full((num_groups, union_pad), -1, dtype=torch.int32, device=device)
    ulen = torch.zeros(num_groups, dtype=torch.int32, device=device)
    mask = torch.zeros(num_groups, nblk, 128, dtype=torch.uint8, device=device)
    for g in range(num_groups):
        toks = [t for t in tok_map_l[g * group : (g + 1) * group] if t >= 0]
        keys = topk[toks].reshape(-1)
        u = torch.unique(keys[keys >= 0])
        union[g, : u.numel()] = u.int()
        ulen[g] = u.numel()
        pos_of = torch.full(
            (int(u.max().item()) + 1 if u.numel() else 1,),
            -1,
            dtype=torch.int64,
            device=device,
        )
        pos_of[u.long()] = torch.arange(u.numel(), device=device)
        for t, tok in enumerate(tok_map_l[g * group : (g + 1) * group]):
            if tok < 0:
                continue
            k = topk[tok].long()
            k = k[k >= 0]
            r = pos_of[k]
            flat = mask[g].view(-1)
            idx = (r // 128) * 128 + t * 16 + (r % 128) // 8
            flat.index_put_((idx,), (1 << (r % 8)).to(torch.uint8), accumulate=True)
    return GroupedSparsePrefillMeta(
        tok_map=torch.from_numpy(tok_map_np).to(device),
        req_of_group=torch.from_numpy(req_np).to(device),
        union=union,
        ulen=ulen,
        mask=mask,
        num_groups=num_groups,
        union_pad=union_pad,
    )
