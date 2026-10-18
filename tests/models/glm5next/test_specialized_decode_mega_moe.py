# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's mega MoE kernel and top-k routing against PyTorch,
on weights prepared like the TRTLLM-Gen FP8 MoE path prepares them."""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

EXPERTS, HIDDEN, INTER, TOP_K, LIMIT = 288, 4096, 512, 8, 10.0
FP8 = torch.float8_e4m3fn

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="Requires SM100"
)


def _quant(x):
    """FP8 per 128 along the last dim with scale amax / 448."""
    blocks = x.float().unflatten(-1, (-1, 128))
    scale = blocks.abs().amax(-1).clamp_min(1e-10) / 448.0
    return (blocks / scale[..., None]).to(FP8).flatten(-2), scale


def _dequant(q, scale):
    return (q.float().unflatten(-1, (-1, 128)) * scale[..., None]).flatten(-2)


def _weight(w, s):
    """[N, K] FP8 with [N / 128, K / 128] block scales."""
    return w.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)


@pytest.mark.parametrize("tokens", [1, 2])
def test_mega_moe_matches_reference(tokens):
    import vllm.model_executor.layers.fused_moe  # noqa: F401
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        _shuffle_deepseek_fp8_moe_weights,
        swap_w13_to_w31,
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
    mega = MegaExperts(w31_trtllm, w31_s, w2_trtllm, w2_s, TOP_K, LIMIT)

    x_q, x_s = _quant(randn(tokens, HIDDEN).bfloat16())
    ids = torch.stack(
        [torch.randperm(EXPERTS, generator=g, device="cuda")[:TOP_K] for _ in
         range(tokens)]
    ).int()  # fmt: skip
    out = mega(x_q, x_s, ids)

    x = _dequant(x_q, x_s)
    ref = []
    for t in range(tokens):
        for e in ids[t].tolist():
            up, gate = (_weight(w31[e], w31_s[e]) @ x[t]).split(INTER)
            act = F.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT)
            hidden = _dequant(*_quant(act))
            ref.append(_weight(w2[e], w2_s[e]) @ hidden)
    ref = torch.stack(ref)
    # BF16 outputs; rare FP8 rounding flips of the hidden move a few elements.
    torch.testing.assert_close(out.float(), ref, atol=5e-3 * ref.abs().max(), rtol=0)


@pytest.mark.parametrize("tokens", [1, 4, 16])
def test_routing_matches_torch(tokens):
    from vllm.models.glm5next.nvidia.specialized_decode.routing import route

    g = torch.Generator(device="cuda").manual_seed(tokens)
    logits = torch.randn(tokens, EXPERTS, generator=g, device="cuda")
    bias = torch.randn(EXPERTS, generator=g, device="cuda") * 0.1
    ids, weights = route(logits, bias, TOP_K, 2.5)

    scores = logits.sigmoid()
    ref_ids = (scores + bias).topk(TOP_K, dim=-1).indices
    ref_w = scores.gather(1, ref_ids)
    ref_w = ref_w / ref_w.sum(-1, keepdim=True) * 2.5
    assert torch.equal(ids.long().sort(-1).values, ref_ids.sort(-1).values)
    order, ref_order = ids.long().argsort(-1), ref_ids.argsort(-1)
    torch.testing.assert_close(
        weights.gather(1, order), ref_w.gather(1, ref_order), atol=1e-6, rtol=1e-6
    )
