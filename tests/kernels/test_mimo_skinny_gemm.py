# SPDX-License-Identifier: Apache-2.0
"""Precision test: skinny_gemm_bf16 vs PyTorch matmul (fp32 accumulate)."""

import pytest
import torch

from vllm.model_executor.models.mimo_v2_fused import skinny_gemm_bf16


@pytest.mark.parametrize("m", [1, 4, 9, 16])
@pytest.mark.parametrize("n,k", [(6144, 2048), (384, 6144), (27648, 6144)])
@pytest.mark.parametrize("out_fp32", [False, True])
def test_skinny_gemm_precision(m, n, k, out_fp32):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    device = "cuda"
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    w = torch.randn(n, k, dtype=torch.bfloat16, device=device)

    ref = (x.float() @ w.float().T)
    out = skinny_gemm_bf16(
        x, w, out_dtype=torch.float32 if out_fp32 else torch.bfloat16)

    assert out.shape == (m, n)
    if out_fp32:
        assert out.dtype == torch.float32
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-2)
    else:
        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(out, ref.to(torch.bfloat16),
                                   rtol=1.6e-2, atol=1e-2)


def test_skinny_gemm_matches_cublas_bf16():
    """Sanity vs actual cuBLAS bf16 output (the op being replaced)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(3)
    x = torch.randn(9, 6144, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(384, 6144, dtype=torch.bfloat16, device="cuda")
    ref = x @ w.T
    out = skinny_gemm_bf16(x, w)
    torch.testing.assert_close(out, ref, rtol=1.6e-2, atol=1e-2)
