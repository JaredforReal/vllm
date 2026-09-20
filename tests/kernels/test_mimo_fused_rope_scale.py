# SPDX-License-Identifier: Apache-2.0
"""Precision test: fused_rope_scale vs vLLM RotaryEmbedding + naive scale."""

import pytest
import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CompilationConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.mimo_v2_fused import fused_rope_scale

HEAD_DIM = 192
V_HEAD_DIM = 128
ROT_DIM = 64  # int(192 * 0.334)
V_SCALE = 0.612


def _make_rope(device, max_pos=4096):
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["all"]))
    with set_current_vllm_config(vllm_config):
        rope = get_rope(
            head_size=HEAD_DIM,
            max_position=max_pos,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.334,
            },
            dtype=torch.bfloat16,
        ).to(device)
    return rope


@pytest.mark.parametrize("m", [1, 4, 9])
@pytest.mark.parametrize("num_kv", [1, 8])
def test_fused_rope_scale_precision(m, num_kv):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    device = "cuda"
    rope = _make_rope(device)
    num_q = 16

    q_sz, k_sz, v_sz = num_q * HEAD_DIM, num_kv * HEAD_DIM, num_kv * V_HEAD_DIM
    qkv = torch.randn(m, q_sz + k_sz + v_sz, dtype=torch.bfloat16,
                      device=device)
    positions = torch.randint(0, 4096, (m,), device=device, dtype=torch.long)

    # reference: vllm rope + naive scale
    # NOTE: vllm's rotary op is in-place — clone the views first so the
    # fused kernel still sees pristine input.
    q_ref, k_ref = rope(positions, qkv[:, :q_sz].clone(),
                        qkv[:, q_sz:q_sz + k_sz].clone())
    v_ref = qkv[:, q_sz + k_sz:] * V_SCALE

    q, k, v = fused_rope_scale(qkv, positions, rope.cos_sin_cache, V_SCALE,
                               num_q, num_kv, HEAD_DIM, V_HEAD_DIM)

    torch.testing.assert_close(q, q_ref, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(k, k_ref, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(v, v_ref, rtol=1e-3, atol=1e-4)


def test_fused_rope_scale_passthrough_channels():
    """Channels beyond rotary_dim must pass through bit-exactly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(1)
    device = "cuda"
    m, num_q, num_kv = 4, 16, 1
    q_sz, k_sz, v_sz = num_q * HEAD_DIM, num_kv * HEAD_DIM, num_kv * V_HEAD_DIM
    qkv = torch.randn(m, q_sz + k_sz + v_sz, dtype=torch.bfloat16,
                      device=device)
    positions = torch.zeros(m, device=device, dtype=torch.long)
    rope = _make_rope(device, max_pos=64)
    q, k, v = fused_rope_scale(qkv, positions, rope.cos_sin_cache, V_SCALE,
                               num_q, num_kv, HEAD_DIM, V_HEAD_DIM)
    # pass-through region of q must equal input exactly
    torch.testing.assert_close(q.view(m, num_q, HEAD_DIM)[:, :, ROT_DIM:],
                               qkv[:, :q_sz].view(m, num_q, HEAD_DIM)[:, :,
                                                                     ROT_DIM:],
                               rtol=0, atol=0)
