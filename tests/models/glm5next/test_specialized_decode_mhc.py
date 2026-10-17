# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's one-kernel mHC boundary against the stock PyTorch
reference of mHC post, pre and the layer-input RMSNorm."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch
from vllm.platforms import current_platform

HIDDEN, HC, SINKHORN = 4096, 4, 20
EPS = 1e-6

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the boundary kernel is a CUDA kernel"
)


def _inputs(tokens, device):
    g = torch.Generator(device=device).manual_seed(tokens)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=device)

    return dict(
        x=randn(tokens, HIDDEN).bfloat16(),
        residual=randn(tokens, HC, HIDDEN).bfloat16(),
        post=torch.rand(tokens, HC, 1, generator=g, device=device) * 2,
        comb=torch.softmax(randn(tokens, HC, HC), -1),
        # hc_fn is BF16 in the checkpoint; the kernel reads a BF16 copy.
        fn=(randn(2 * HC + HC * HC, HC * HIDDEN) * 0.02).bfloat16(),
        scale=torch.tensor([0.5, 0.5, 1.0], device=device),
        base=randn(2 * HC + HC * HC) * 0.1,
        norm=(1 + 0.1 * randn(HIDDEN)).bfloat16(),
    )


def _reference(i):
    residual = mhc_post_torch(i["x"], i["residual"], i["post"], i["comb"])
    post, comb, collapsed = mhc_pre_torch(
        residual, i["fn"].float(), i["scale"], i["base"],
        EPS, EPS, EPS, 2.0, SINKHORN,
    )  # fmt: skip
    v = collapsed.float()
    rstd = torch.rsqrt(v.square().mean(-1, keepdim=True) + EPS)
    return residual, post, comb, (v * rstd * i["norm"].float()).bfloat16()


@pytest.mark.parametrize("tokens", [1, 4, 16])
def test_boundary_matches_reference(tokens):
    from vllm.models.glm5next.nvidia.specialized_decode.mhc import MhcBoundary

    device = torch.device("cuda")
    layer = SimpleNamespace(
        rms_norm_eps=EPS,
        hc_eps=EPS,
        mhc_post_mult_value=2.0,
        mhc_sinkhorn_iterations=SINKHORN,
    )
    boundary = MhcBoundary(layer, HIDDEN, device)
    i = _inputs(tokens, device)
    args = (i["x"], i["residual"], i["post"], i["comb"], i["fn"], i["scale"])
    out = boundary(*args, i["base"], i["norm"], EPS)
    ref = _reference(i)

    # The post-mixed streams and the layer input are BF16: one rounding apart at
    # most. The mixes are FP32 and differ only in summation order.
    torch.testing.assert_close(out[0], ref[0], atol=2**-6, rtol=2**-7)
    torch.testing.assert_close(out[1], ref[1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[2], ref[2], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[3], ref[3], atol=2**-5, rtol=2**-7)

    # The cross-CTA counters keep growing; a second call must give the same result.
    again = boundary(*args, i["base"], i["norm"], EPS)
    for a, b in zip(out, again):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
