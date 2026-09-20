# SPDX-License-Identifier: Apache-2.0
"""Fused MoE router epilogue for MiMo-V2 style routing (sigmoid + noaux_tc).

Decode-path chain per layer was:
    gate GEMM (nvjet) -> splitK reduce -> sigmoid -> +bias -> topk -> gather
    -> renormalize   (multiple kernels)
Now:
    torch.mm(out_dtype=fp32)   # cuBLAS wins at M<=16; fp32 accum matches naive
    _router_topk_kernel        # one Triton kernel for everything after the GEMM

Numerics: sigmoid in fp32, bias add in fp32, selection on biased scores,
routing weights from unbiased sigmoid scores, optional renorm — identical to
``grouped_topk`` with n_group=1/topk_group=1.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _router_topk_kernel(
    logits_ptr,  # [M, N] fp32
    bias_ptr,  # [N] fp32
    out_w_ptr,  # [M, TOPK] fp32
    out_i_ptr,  # [M, TOPK] int32
    N: tl.constexpr,
    N_PAD: tl.constexpr,
    TOPK: tl.constexpr,
    RENORM: tl.constexpr,
):
    m = tl.program_id(0)
    offs_n = tl.arange(0, N_PAD)
    n_mask = offs_n < N
    logits = tl.load(logits_ptr + m * N + offs_n, mask=n_mask, other=0.0)
    scores = tl.sigmoid(logits)
    biased = scores + tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
    biased = tl.where(n_mask, biased, float("-inf"))

    # iterative top-k (N is a few hundred; loop is cheap)
    vals = tl.full((TOPK,), float("-inf"), dtype=tl.float32)
    ids = tl.zeros((TOPK,), dtype=tl.int32)
    work = biased
    for i in tl.static_range(TOPK):
        v = tl.max(work, axis=0)
        idx = tl.argmax(work, axis=0)
        vals = tl.where(tl.arange(0, TOPK) == i, v, vals)
        ids = tl.where(tl.arange(0, TOPK) == i, idx, ids)
        work = tl.where(offs_n == idx, float("-inf"), work)

    # routing weights use the *unbiased* sigmoid scores
    w = tl.load(logits_ptr + m * N + ids)
    w = tl.sigmoid(w)
    if RENORM:
        w = w / tl.sum(w, axis=0)

    offs_t = tl.arange(0, TOPK)
    tl.store(out_w_ptr + m * TOPK + offs_t, w)
    tl.store(out_i_ptr + m * TOPK + offs_t, ids)


def fused_router_topk(
    hidden_states: torch.Tensor,  # [M, K] bf16
    weight: torch.Tensor,  # [N, K] bf16
    e_score_correction_bias: torch.Tensor,  # [N] fp32
    topk: int = 8,
    renormalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused MoE router: skinny GEMM + sigmoid + bias + topk (+ renorm).

    Returns (topk_weights fp32 [M, topk], topk_ids int32 [M, topk]).
    """
    M, K = hidden_states.shape
    N = weight.shape[0]
    assert N % 128 == 0 and K % 64 == 0
    assert hidden_states.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    assert e_score_correction_bias.dtype == torch.float32

    logits = torch.mm(hidden_states, weight.t(), out_dtype=torch.float32)
    topk_w = torch.empty((M, topk), dtype=torch.float32,
                         device=hidden_states.device)
    topk_i = torch.empty((M, topk), dtype=torch.int32,
                         device=hidden_states.device)
    _router_topk_kernel[(M,)](
        logits, e_score_correction_bias, topk_w, topk_i, N, 512, topk,
        RENORM=renormalize, num_warps=4,
    )
    return topk_w, topk_i
