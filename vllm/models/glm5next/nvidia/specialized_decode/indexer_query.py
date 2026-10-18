# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The indexer's per-token epilogue in one kernel: key LayerNorm, query
Hadamard rotation and FP8 quant, head-weight scaling, and the zero padding of
the query heads and weights that DeepGEMM's MQA logits need (stock: five
kernels)."""

import torch

from vllm.models.glm5next.nvidia.ops.kpool_compress import _fwht_stage
from vllm.triton_utils import tl, triton


@triton.jit
def _query_kernel(
    q_ptr,
    k_ptr,
    weights_ptr,
    norm_w_ptr,
    norm_b_ptr,
    q_out_ptr,
    weights_out_ptr,
    k_out_ptr,
    scale,
    eps,
    HEADS: tl.constexpr,
    PADDED: tl.constexpr,
    DIM: tl.constexpr,
):
    t = tl.program_id(0)
    heads = tl.arange(0, HEADS)
    offs = tl.arange(0, DIM)

    x = tl.load(q_ptr + t * HEADS * DIM + heads[:, None] * DIM + offs[None, :])
    x = tl.reshape(x.to(tl.float32), (HEADS * DIM,))
    N: tl.constexpr = HEADS * DIM
    x = _fwht_stage(x, N, HEADS * 64, 1)
    x = _fwht_stage(x, N, HEADS * 32, 2)
    x = _fwht_stage(x, N, HEADS * 16, 4)
    x = _fwht_stage(x, N, HEADS * 8, 8)
    x = _fwht_stage(x, N, HEADS * 4, 16)
    x = _fwht_stage(x, N, HEADS * 2, 32)
    x = _fwht_stage(x, N, HEADS, 64)
    x = x * 0.08838834764831845  # 1/sqrt(128), exact in fp32
    x = x.to(tl.bfloat16).to(tl.float32)
    x = tl.reshape(x, (HEADS, DIM))
    absmax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    q_scale = tl.exp2(tl.ceil(tl.log2(absmax * (1.0 / 448.0))))
    y = tl.minimum(tl.maximum(x / q_scale[:, None], -448.0), 448.0)
    q_out = q_out_ptr + t * PADDED * DIM
    tl.store(q_out + heads[:, None] * DIM + offs[None, :], y.to(tl.float8e4nv))
    w = tl.load(weights_ptr + t * HEADS + heads)
    tl.store(weights_out_ptr + t * PADDED + heads, w * q_scale * scale)
    if PADDED > HEADS:
        pad = tl.arange(0, PADDED - HEADS)
        tl.store(
            q_out + (HEADS + pad[:, None]) * DIM + offs[None, :],
            tl.zeros((PADDED - HEADS, DIM), dtype=tl.float8e4nv),
        )
        tl.store(
            weights_out_ptr + t * PADDED + HEADS + pad,
            tl.zeros((PADDED - HEADS,), dtype=tl.float32),
        )

    k = tl.load(k_ptr + t * DIM + offs).to(tl.float32)
    mean = tl.sum(k, axis=0) / DIM
    centered = k - mean
    var = tl.sum(centered * centered, axis=0) / DIM
    k = centered * tl.rsqrt(var + eps) * tl.load(norm_w_ptr + offs) + tl.load(
        norm_b_ptr + offs
    )
    tl.store(k_out_ptr + t * DIM + offs, k.to(tl.bfloat16))


def indexer_query(q, k, weights, norm, scale: float, padded: int):
    """Returns (q_fp8 [M, padded, 128], weights [M, padded] FP32, key [M, 128]).

    Args:
        q: [M, heads * 128] BF16 indexer query.
        k: [M, 128] BF16 indexer key before its LayerNorm.
        weights: [M, heads] FP32 head weights.
        norm: the key LayerNorm module (FP32 weight and bias).
        scale: the folded softmax and head-count scale.
        padded: heads after zero padding.

    """
    m, dim = k.shape
    heads = weights.shape[1]
    q_out = torch.empty((m, padded, dim), dtype=torch.float8_e4m3fn, device=q.device)
    w_out = torch.empty((m, padded), dtype=torch.float32, device=q.device)
    k_out = torch.empty_like(k)
    _query_kernel[(m,)](
        q, k, weights, norm.weight, norm.bias, q_out, w_out, k_out,
        scale, norm.eps, HEADS=heads, PADDED=padded, DIM=dim, num_warps=4,
    )  # fmt: skip
    return q_out, w_out, k_out
