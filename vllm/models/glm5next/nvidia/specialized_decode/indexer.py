# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA indexer stage: the indexer's input projections and per-token query
epilogue in two kernels instead of eleven.

Stock decode runs the indexer key (``wk``), its FP32 head weights (an FP32 copy
of ``weights_proj`` against an FP32 copy of the input) and the kpool gate as a
skinny GEMM, an upcast, a cuBLAS dot with its reduce and a split-K GEMM with its
reduce. Here one GEMV reads the concatenated [wk; weights_proj; gate] rows once
and writes each part in its stock dtype: the products of BF16 values are exact
in FP32, so only the summation order differs. ``indexer_query`` then replaces
the key LayerNorm, Hadamard quant, weight scaling and head padding kernels.
"""

import torch

from vllm.triton_utils import tl, triton

from .indexer_query import indexer_query
from .linear import planned

# DeepGEMM's MQA logits take 32 or 64 heads; fewer are zero-padded.
MQA_MIN_HEADS = 32

MAX_BLOCK_M = 8


def _tactic(tokens: int) -> tuple[int, int]:
    """(rows per program, warps) from a GB300 sweep; K is one block."""
    if tokens <= 2:
        return 2, 8
    return (2, 4) if tokens <= 16 else (4, 4)


@triton.jit
def _projection_kernel(
    x_ptr,
    w_ptr,
    key_ptr,
    weights_ptr,
    gate_ptr,
    M,
    K: tl.constexpr,
    KEY: tl.constexpr,
    HEADS: tl.constexpr,
    GATE: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    tokens = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    live = tokens < M
    acc = tl.zeros((BLOCK_M, ROWS), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(w_ptr + rows[:, None] * K + cols[None, :]).to(tl.float32)
        x = tl.load(
            x_ptr + tokens[:, None] * K + cols[None, :], mask=live[:, None], other=0.0
        ).to(tl.float32)
        acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2)
    t = tokens[:, None]
    r = rows[None, :]
    tl.store(key_ptr + t * KEY + r, acc.to(tl.bfloat16), mask=live[:, None] & (r < KEY))
    head = r - KEY
    tl.store(
        weights_ptr + t * HEADS + head,
        acc,
        mask=live[:, None] & (head >= 0) & (head < HEADS),
    )
    g = r - KEY - HEADS
    tl.store(
        gate_ptr + t * GATE + g, acc.to(tl.bfloat16), mask=live[:, None] & (g >= 0)
    )


class IndexerProjection:
    """(key BF16, head weights FP32, kpool gate BF16) of the layer input."""

    def __init__(self, indexer):
        self.key = indexer.head_dim
        self.heads = indexer.n_head
        gate = indexer.index_kpool_compress_gate.data
        self.gate = gate.shape[0]
        self.weight = torch.cat([indexer.wk_weights_proj.weight.data, gate])
        if self.weight.shape[0] % 4 or self.weight.shape[1] & (
            self.weight.shape[1] - 1
        ):
            raise ValueError("indexer projection rows or K do not tile")

    def __call__(self, x: torch.Tensor):
        m, k = x.shape
        key = torch.empty((m, self.key), dtype=torch.bfloat16, device=x.device)
        weights = torch.empty((m, self.heads), dtype=torch.float32, device=x.device)
        gate = torch.empty((m, self.gate), dtype=torch.bfloat16, device=x.device)
        rows, warps = _tactic(m)
        block_m = min(MAX_BLOCK_M, triton.next_power_of_2(m))
        grid = (self.weight.shape[0] // rows, triton.cdiv(m, block_m))
        _projection_kernel[grid](
            x, self.weight, key, weights, gate, m,
            K=k, KEY=self.key, HEADS=self.heads, GATE=self.gate,
            ROWS=rows, BLOCK_K=k, BLOCK_M=block_m, num_warps=warps,
        )  # fmt: skip
        return key, weights, gate


class IndexerStage:
    """``Indexer.forward`` with the fused projection inside planned steps.

    Installed as the indexer's ``forward``: the DSA attention calls it from its
    stock module, and every call outside a planned step runs the stock forward.
    """

    def __init__(self, indexer):
        if indexer.rope_dim != 0:
            raise ValueError("the fused indexer path assumes a NoPE indexer")
        self.indexer = indexer
        self.stock = indexer.forward
        self.projection = IndexerProjection(indexer)
        indexer.forward = self.forward

    def forward(self, hidden_states, qr, positions, rotary_emb):
        if not planned():
            return self.stock(hidden_states, qr, positions, rotary_emb)
        ind = self.indexer
        q, _ = ind.wq_b(qr)
        k, weights, gate_score = self.projection(hidden_states)
        q_fp8, weights, k = indexer_query(
            q,
            k,
            weights,
            ind.k_norm,
            ind.softmax_scale * ind.n_head**-0.5,
            max(ind.n_head, MQA_MIN_HEADS),
        )
        return ind.indexer_op(
            hidden_states,
            q_fp8,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=ind.index_kpool_compress_ape,
            index_kpool=ind.index_kpool,
            positions=positions,
        )
