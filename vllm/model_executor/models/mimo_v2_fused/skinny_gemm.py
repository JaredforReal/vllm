# SPDX-License-Identifier: Apache-2.0
"""Skinny bf16 GEMM (M <= 16) tuned for the decode path.

At M <= 16 cuBLAS/nvjet tiles are heavily underutilized; a simple Triton
kernel with fp32 accumulation and K-splitting across programs wins by keeping
the reduction in registers and reading weights exactly once per program.

Y = X @ W.T   with X [M, K] bf16, W [N, K] bf16, out [M, N] (bf16 or fp32).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _skinny_gemm_kernel(
    x_ptr,  # [M, K] bf16
    w_ptr,  # [N, K] bf16
    out_ptr,  # [M_PAD, N] out_dtype
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    M_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUT_FP32: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, M_PAD)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * K + offs_k[None, :]

    acc = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs)
        acc = tl.dot(x, tl.trans(w), acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    if OUT_FP32:
        tl.store(out_ptrs, acc, mask=m_mask[:, None])
    else:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None])


def skinny_gemm_bf16(
    x: torch.Tensor,  # [M, K] bf16
    weight: torch.Tensor,  # [N, K] bf16
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute x @ weight.T for M <= 16. Returns [M, N] in out_dtype."""
    M, K = x.shape
    N = weight.shape[0]
    assert M <= 16 and K % 64 == 0 and N % 128 == 0
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16

    M_PAD = 16
    if out is None:
        out = torch.empty((M_PAD, N), dtype=out_dtype, device=x.device)
    else:
        assert out.shape[0] >= M_PAD and out.shape[1] == N
    _skinny_gemm_kernel[(N // 128,)](
        x, weight, out, M, N, K, M_PAD,
        BLOCK_N=128, BLOCK_K=64, OUT_FP32=(out_dtype == torch.float32),
        num_warps=4,
    )
    return out[:M]
