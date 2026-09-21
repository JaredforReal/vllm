# SPDX-License-Identifier: Apache-2.0
"""Fused partial-RoPE + value-scale for MiMo-V2 attention inputs.

One kernel covers what is otherwise an inductor split/rotate/scale chain:
  qkv [M, q_sz + k_sz + v_sz] bf16 -> q_rot [M, q_sz], k_rot [M, k_sz],
  v_scaled [M, v_sz]

- partial rotary: only the first ``rotary_dim`` (= head_dim *
  partial_rotary_factor) channels of each q/k head are rotated
- GPT-NeoX pairing: channel i pairs with i + rotary_dim//2
- math in fp32 (matching vLLM's CUDA rotary_embedding kernel), stored bf16
- v is scaled by the scalar ``attention_value_scale``
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_scale_kernel(
    qkv_ptr,  # [M, QKV] bf16, packed q|k|v per token
    cos_sin_ptr,  # [max_pos, ROT_DIM] bf16 (first half cos, second half sin)
    q_ptr,  # [M, Q_SZ] bf16 out
    k_ptr,  # [M, K_SZ] bf16 out
    v_ptr,  # [M, V_SZ] bf16 out
    pos_ptr,  # [M] int32/int64
    v_scale,
    M,
    QKV: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    ROT_DIM: tl.constexpr,
    K_SZ: tl.constexpr,
    V_SZ: tl.constexpr,
):
    m = tl.program_id(0)
    head = tl.program_id(1)
    if m >= M:
        return

    HALF: tl.constexpr = ROT_DIM // 2
    pos = tl.load(pos_ptr + m).to(tl.int64)
    cos = tl.load(cos_sin_ptr + pos * ROT_DIM + tl.arange(0, HALF)).to(tl.float32)
    sin = tl.load(
        cos_sin_ptr + pos * ROT_DIM + HALF + tl.arange(0, HALF)).to(tl.float32)

    if head < NUM_Q_HEADS + NUM_KV_HEADS:
        # q or k head: rotate first ROT_DIM channels, pass through the rest
        base_in = head * HEAD_DIM
        base_out = head * HEAD_DIM
        x0 = tl.load(qkv_ptr + m * QKV + base_in + tl.arange(0, HALF)).to(
            tl.float32)
        x1 = tl.load(qkv_ptr + m * QKV + base_in + HALF + tl.arange(0, HALF)).to(
            tl.float32)
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        if head < NUM_Q_HEADS:
            dst = q_ptr + m * (NUM_Q_HEADS * HEAD_DIM) + base_out
        else:
            dst = k_ptr + m * (NUM_KV_HEADS * HEAD_DIM) + base_out - NUM_Q_HEADS * HEAD_DIM
        tl.store(dst + tl.arange(0, HALF), y0.to(tl.bfloat16))
        tl.store(dst + HALF + tl.arange(0, HALF), y1.to(tl.bfloat16))
        # pass-through channels
        rest = tl.load(qkv_ptr + m * QKV + base_in + ROT_DIM +
                       tl.arange(0, HEAD_DIM - ROT_DIM))
        tl.store(dst + ROT_DIM + tl.arange(0, HEAD_DIM - ROT_DIM), rest)
    else:
        # v: scale only; one program per kv head
        v_head = head - NUM_Q_HEADS - NUM_KV_HEADS
        base_in = (NUM_Q_HEADS + NUM_KV_HEADS) * HEAD_DIM + v_head * V_HEAD_DIM
        v = tl.load(qkv_ptr + m * QKV + base_in +
                    tl.arange(0, V_HEAD_DIM)).to(tl.float32)
        tl.store(v_ptr + m * V_SZ + v_head * V_HEAD_DIM +
                 tl.arange(0, V_HEAD_DIM),
                 (v * v_scale).to(tl.bfloat16))


def fused_rope_scale(
    qkv: torch.Tensor,  # [M, q_sz + k_sz + v_sz] bf16
    positions: torch.Tensor,  # [M] int
    cos_sin_cache: torch.Tensor,  # [max_pos, rot_dim] bf16
    v_scale: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    v_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split packed qkv and apply partial RoPE to q/k and scale to v."""
    M = qkv.shape[0]
    q_sz = num_q_heads * head_dim
    k_sz = num_kv_heads * head_dim
    v_sz = num_kv_heads * v_head_dim
    assert qkv.shape[1] == q_sz + k_sz + v_sz
    rot_dim = cos_sin_cache.shape[1]
    assert rot_dim % 2 == 0 and rot_dim <= head_dim

    q = torch.empty((M, q_sz), dtype=torch.bfloat16, device=qkv.device)
    k = torch.empty((M, k_sz), dtype=torch.bfloat16, device=qkv.device)
    v = torch.empty((M, v_sz), dtype=torch.bfloat16, device=qkv.device)

    grid = (max(M, 1), num_q_heads + 2 * num_kv_heads)
    _rope_scale_kernel[grid](
        qkv, cos_sin_cache, q, k, v, positions, v_scale, M,
        QKV=qkv.shape[1], NUM_Q_HEADS=num_q_heads, NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim, V_HEAD_DIM=v_head_dim, ROT_DIM=rot_dim,
        K_SZ=k_sz, V_SZ=v_sz, num_warps=2,
    )
    return q, k, v
