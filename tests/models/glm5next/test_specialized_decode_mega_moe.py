# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's mega MoE kernel against PyTorch, on weights
prepared like the TRTLLM-Gen FP8 MoE path prepares them."""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

EXPERTS, HIDDEN, INTER, TOP_K, LIMIT, SCALE = 288, 4096, 512, 8, 10.0, 2.5
FP8 = torch.float8_e4m3fn

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="Requires SM100"
)


def _quant(x, reciprocal=False, ue8m0=False):
    """FP8 per 128 along the last dim with scale amax / 448; ``reciprocal``
    multiplies by 1 / scale and ``ue8m0`` rounds the scale up to a power of
    two, like per_token_group_quant_fp8."""
    blocks = x.float().unflatten(-1, (-1, 128))
    amax = blocks.abs().amax(-1).clamp_min(1e-10)
    # IEEE division: dividing by a Python scalar multiplies by its reciprocal.
    scale = amax / torch.full_like(amax, 448.0)
    if ue8m0:
        scale = torch.exp2(torch.ceil(torch.log2(scale.clamp_min(1e-10))))
    scaled = (
        blocks * (1 / scale[..., None]) if reciprocal else blocks / scale[..., None]
    )
    return scaled.to(FP8).flatten(-2), scale


def _dequant(q, scale):
    return (q.float().unflatten(-1, (-1, 128)) * scale[..., None]).flatten(-2)


def _weight(w, s):
    """[N, K] FP8 with [N / 128, K / 128] block scales."""
    return w.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)


def _route(logits, bias):
    scores = logits.float().sigmoid()
    ids = (scores + bias).topk(TOP_K, dim=-1).indices
    weights = scores.gather(1, ids)
    return ids, weights / weights.sum(-1, keepdim=True) * SCALE


@pytest.mark.parametrize(
    ("grouped", "tokens"),
    [(False, 1), (False, 2), (True, 1), (True, 2), (True, 4), (True, 8), (True, 16)],
)
@pytest.mark.parametrize("ue8m0", [False, True])
def test_mega_moe_matches_reference(grouped, tokens, ue8m0):
    """Routing, activation quant and both expert GEMMs against PyTorch."""
    import vllm.model_executor.layers.fused_moe  # noqa: F401
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        _shuffle_deepseek_fp8_moe_weights,
        swap_w13_to_w31,
    )
    from vllm.models.glm5next.nvidia.specialized_decode.grouped_moe import (
        GroupedExperts,
        shared_weights,
    )
    from vllm.models.glm5next.nvidia.specialized_decode.mega_moe import MegaExperts

    g = torch.Generator(device="cuda").manual_seed(tokens)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda")

    w13 = (randn(EXPERTS, 2 * INTER, HIDDEN) * 0.5).to(FP8)  # [gate; up]
    w13_s = torch.rand(EXPERTS, 2 * INTER // 128, HIDDEN // 128, generator=g,
                       device="cuda") * 0.02  # fmt: skip
    w2 = (randn(EXPERTS, HIDDEN, INTER) * 0.5).to(FP8)
    w2_s = torch.rand(EXPERTS, HIDDEN // 128, INTER // 128, generator=g,
                      device="cuda") * 0.02  # fmt: skip
    w31, w31_s = swap_w13_to_w31(w13), swap_w13_to_w31(w13_s)
    w31_trtllm, w2_trtllm = _shuffle_deepseek_fp8_moe_weights(w31, w2)
    args = (w31_trtllm, w31_s, w2_trtllm, w2_s, TOP_K, LIMIT, SCALE, ue8m0)
    if grouped:
        # The shared expert folds in as one more expert of every token.
        se13 = (randn(2 * INTER, HIDDEN) * 0.5).to(FP8)
        se13_s = torch.rand(2 * INTER // 128, HIDDEN // 128, generator=g,
                            device="cuda") * 0.02  # fmt: skip
        se2 = (randn(HIDDEN, INTER) * 0.5).to(FP8)
        se2_s = torch.rand(HIDDEN // 128, INTER // 128, generator=g,
                           device="cuda") * 0.02  # fmt: skip
        shared = shared_weights(se13, se13_s, se2, se2_s)
        mega = GroupedExperts(*args, shared=shared)
    else:
        mega = MegaExperts(*args)

    x = randn(tokens, HIDDEN).bfloat16()
    logits = randn(tokens, EXPERTS)
    bias = randn(EXPERTS) * 0.1
    out, ids, weights, *shared_out = mega(x, logits, bias)
    if grouped:  # the workspace counters must be reset for the next call
        out, ids, weights, shared_out = mega(x, logits, bias)

    ref_ids, ref_weights = _route(logits, bias)
    order, ref_order = ids.long().sort(-1), ref_ids.sort(-1)
    assert torch.equal(order.values, ref_order.values)
    # The kernel's sigmoid is a few ulp off PyTorch's.
    torch.testing.assert_close(
        weights.gather(1, order.indices),
        ref_weights.gather(1, ref_order.indices),
        atol=0,
        rtol=1e-5,
    )

    x = _dequant(*_quant(x, reciprocal=True, ue8m0=ue8m0))

    def expert(w31, w31_s, w2, w2_s, token):
        up, gate = (_weight(w31, w31_s) @ x[token]).split(INTER)
        act = F.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT)
        return _weight(w2, w2_s) @ _dequant(*_quant(act))

    ref = torch.stack(
        [
            expert(w31[e], w31_s[e], w2[e], w2_s[e], t)
            for t in range(tokens)
            for e in ids[t].tolist()
        ]
    )
    # BF16 outputs; rare FP8 rounding flips of the hidden move a few elements.
    torch.testing.assert_close(out.float(), ref, atol=5e-3 * ref.abs().max(), rtol=0)
    if grouped:
        se31, se31_s = swap_w13_to_w31(se13), swap_w13_to_w31(se13_s)
        ref = torch.stack(
            [expert(se31, se31_s, se2, se2_s, t) for t in range(tokens)]
        )
        torch.testing.assert_close(
            shared_out.float(), ref, atol=5e-3 * ref.abs().max(), rtol=0
        )
