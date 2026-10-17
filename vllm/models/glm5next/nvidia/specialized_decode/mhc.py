# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-kernel mHC sublayer boundary for decode batches.

Stock runs two kernels per boundary: a post-mix + split prenorm GEMV kernel and
a single-CTA kernel for the Sinkhorn, the collapse and the RMSNorm. Here each
token's hidden dim is split over ``SPLITS`` co-resident CTAs that

1. post-mix their column slice of the 4 residual streams (BF16 out),
2. publish partial 24-way prenorm GEMVs and sums of squares, then wait for all
   splits of their token (an acquire / release counter),
3. finish the mixes, run sigmoid / Sinkhorn redundantly, collapse their slice,
   publish its sum of squares and wait again for the RMSNorm.

Counters only grow, so CUDA graph replays need no reset. Batches are capped so
every CTA is resident at once (the waits would deadlock otherwise).
"""

import torch

from vllm.triton_utils import tl, triton

HC = 4
MIX = (2 + HC) * HC  # 24: pre 4, post 4, comb 16
SPLITS = 32
MAX_SPLITS = 64
MAX_TOKENS = 16


@triton.jit
def _wait(counter, splits):
    old = tl.atomic_add(counter, 1, sem="release", scope="gpu")
    target = (old // splits + 1) * splits
    seen = tl.atomic_add(counter, 0, sem="acquire", scope="gpu")
    while seen < target:
        seen = tl.atomic_add(counter, 0, sem="acquire", scope="gpu")


@triton.jit
def _boundary_kernel(
    x_ptr,
    res_ptr,
    post_ptr,
    comb_ptr,
    fn_ptr,
    scale_ptr,
    base_ptr,
    norm_w_ptr,
    res_out_ptr,
    post_out_ptr,
    comb_out_ptr,
    y_ptr,
    part_ptr,
    part2_ptr,
    counter_ptr,
    rms_eps,
    pre_eps,
    sinkhorn_eps,
    post_mult,
    norm_eps,
    H: tl.constexpr,
    SPLITS: tl.constexpr,
    COLS: tl.constexpr,
    REPEAT: tl.constexpr,
):
    s = tl.program_id(0)
    b = tl.program_id(1)
    cols = s * COLS + tl.arange(0, COLS)
    four = tl.arange(0, 4)

    x = tl.load(x_ptr + b * H + cols).to(tl.float32)
    post = tl.load(post_ptr + b * 4 + four)
    comb = tl.load(comb_ptr + b * 16 + four[:, None] * 4 + four[None, :])  # [src, dst]
    res = tl.load(res_ptr + b * 4 * H + four[:, None] * H + cols[None, :]).to(
        tl.float32
    )  # [4, COLS]
    # new[j] = sum_i comb[i, j] * res[i] + post[j] * x
    new = (
        tl.sum(comb[:, :, None] * res[:, None, :], axis=0) + post[:, None] * x[None, :]
    )
    new = new.to(tl.bfloat16)
    tl.store(res_out_ptr + b * 4 * H + four[:, None] * H + cols[None, :], new)
    new = new.to(tl.float32)

    # Partial prenorm GEMV: mixes[m] += sum_j sum_c fn[m, j * H + c] * new[j, c].
    rows = tl.arange(0, 32)
    rmask = rows < 24
    acc = tl.zeros((32,), dtype=tl.float32)
    for j in tl.static_range(4):
        nj = tl.sum(tl.where(four[:, None] == j, new, 0.0), axis=0)
        f = tl.load(
            fn_ptr + rows[:, None] * (4 * H) + j * H + cols[None, :],
            mask=rmask[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(f * nj[None, :], axis=1)
    sq = tl.sum(tl.sum(new * new, axis=1), axis=0)
    acc = tl.where(rows == 24, sq, acc)
    tl.store(part_ptr + (b * SPLITS + s) * 32 + rows, acc)
    tl.debug_barrier()
    _wait(counter_ptr + b * 2, SPLITS)
    tl.debug_barrier()

    splits = tl.arange(0, SPLITS)
    parts = tl.load(
        part_ptr + (b * SPLITS + splits[:, None]) * 32 + rows[None, :], volatile=True
    )
    total = tl.sum(parts, axis=0)
    sumsq = tl.sum(tl.where(rows == 24, total, 0.0), axis=0)
    rstd = tl.rsqrt(sumsq / (4 * H) + rms_eps)
    mixes = total * rstd
    scale0 = tl.load(scale_ptr)
    scale1 = tl.load(scale_ptr + 1)
    scale2 = tl.load(scale_ptr + 2)
    m_pre = tl.sum(tl.where(rows[None, :] == four[:, None], mixes[None, :], 0.0), 1)
    m_post = tl.sum(
        tl.where(rows[None, :] == four[:, None] + 4, mixes[None, :], 0.0), 1
    )
    pre = tl.sigmoid(m_pre * scale0 + tl.load(base_ptr + four)) + pre_eps
    post_new = tl.sigmoid(m_post * scale1 + tl.load(base_ptr + 4 + four)) * post_mult
    idx = four[:, None] * 4 + four[None, :]
    m_comb = tl.sum(
        tl.where(
            rows[None, None, :] == (8 + idx)[:, :, None], mixes[None, None, :], 0.0
        ),
        2,
    )
    logits = m_comb * scale2 + tl.load(base_ptr + 8 + idx)
    e = tl.exp(logits - tl.max(logits, axis=1)[:, None])
    c = e / tl.sum(e, axis=1)[:, None] + sinkhorn_eps
    c = c / (tl.sum(c, axis=0)[None, :] + sinkhorn_eps)
    for _ in tl.static_range(REPEAT - 1):
        c = c / (tl.sum(c, axis=1)[:, None] + sinkhorn_eps)
        c = c / (tl.sum(c, axis=0)[None, :] + sinkhorn_eps)

    y = tl.sum(pre[:, None] * new, axis=0).to(tl.bfloat16).to(tl.float32)
    tl.store(part2_ptr + b * SPLITS + s, tl.sum(y * y, axis=0))
    tl.debug_barrier()
    _wait(counter_ptr + b * 2 + 1, SPLITS)
    tl.debug_barrier()
    ysq = tl.sum(tl.load(part2_ptr + b * SPLITS + splits, volatile=True), axis=0)
    w = tl.load(norm_w_ptr + cols).to(tl.float32)
    out = y * tl.rsqrt(ysq / H + norm_eps) * w
    tl.store(y_ptr + b * H + cols, out.to(tl.bfloat16))
    if s == 0:
        tl.store(post_out_ptr + b * 4 + four, post_new)
        tl.store(comb_out_ptr + b * 16 + idx, c)


class MhcBoundary:
    """Drop-in for ``hc_fused_post_pre`` with a norm weight, for <= 16 tokens."""

    def __init__(self, layer, hidden: int, device, splits=SPLITS, warps=4):
        self.layer = layer
        self.hidden = hidden
        self.splits, self.warps = splits, warps
        self.part = torch.zeros(MAX_TOKENS * MAX_SPLITS * 32, device=device)
        self.part2 = torch.zeros(MAX_TOKENS * MAX_SPLITS, device=device)
        self.counter = torch.zeros(MAX_TOKENS * 2, dtype=torch.int32, device=device)

    def __call__(self, x, residual, post, comb, fn, scale, base, norm_w, norm_eps):
        layer, b, h = self.layer, x.shape[0], self.hidden
        res_out = torch.empty_like(residual)
        post_out = torch.empty((b, HC, 1), dtype=torch.float32, device=x.device)
        comb_out = torch.empty((b, HC, HC), dtype=torch.float32, device=x.device)
        y = torch.empty_like(x)
        splits = self.splits
        _boundary_kernel[(splits, b)](
            x, residual, post, comb, fn, scale, base, norm_w,
            res_out, post_out, comb_out, y,
            self.part, self.part2, self.counter,
            layer.rms_norm_eps, layer.hc_eps, layer.hc_eps,
            layer.mhc_post_mult_value, norm_eps,
            H=h, SPLITS=splits, COLS=h // splits,
            REPEAT=layer.mhc_sinkhorn_iterations,
            num_warps=self.warps,
        )  # fmt: skip
        return res_out, post_out, comb_out, y
