# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA indexer stage: the indexer's input projections and per-token query
epilogue in two kernels instead of eleven.

Stock decode runs the indexer key (``wk``), its FP32 head weights (an FP32 copy
of ``weights_proj`` against an FP32 copy of the input) and the kpool gate as a
skinny GEMM, an upcast, a cuBLAS dot with its reduce and a split-K GEMM with its
reduce. Here one GEMM reads the concatenated [wk; weights_proj; gate] rows once and
writes each part in its stock dtype: the products of BF16 values are exact in
FP32, so only the summation order differs. ``indexer_query`` then replaces
the key LayerNorm, Hadamard quant, weight scaling and head padding kernels.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int64, Uint32

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
    stream,
    to_cute,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    alloc,
    asm,
    load_vector,
    load_words,
    mma_into,
    shared_address,
)

from .indexer_query import indexer_query
from .linear import planned
from .mega_moe import _store_bf16

# DeepGEMM's MQA logits take 32 or 64 heads; fewer are zero-padded.
MQA_MIN_HEADS = 32

ROWS = 8  # weight rows per CTA: the n8 of an m16n8k16 mma
TILE = 16  # tokens per mma: its m16


def _tactic(tokens: int) -> tuple[int, int]:
    """(warps per CTA, CTAs per cluster splitting K) from a GB300 sweep."""
    return (16, 1) if tokens <= 16 else (8, 4)


@asm(Float32)
def _remote_f32(address, rank):
    code = "{ .reg .b32 a; mapa.shared::cluster.u32 a, $1, $2; "
    code += "ld.shared::cluster.f32 $0, [a]; }"
    return code, "=f,r,r", (Uint32(address), Uint32(rank))


class _ProjectionKernel:
    """``x @ weight.T`` on mma.sync: tokens are the m16, weight rows the n8.

    A cluster of ``split`` CTAs owns 8 weight rows, each CTA a K part and each
    warp a ``span`` of it. The dot product does not care which K a fragment
    slot holds, so lane (g, q) keeps the contiguous 16-byte chunks
    ``q * 8 + 32 p`` of the span for weight row g and tokens g and g + 8, and
    the loaded words are the mma fragments as they are. The weights are loaded
    into registers before the PDL wait; the warps' and the cluster's partial
    sums are added in shared memory.
    """

    def __init__(self, rows, k, key, heads, tokens, warps, split):
        assert rows % ROWS == 0 and k % (32 * warps * split) == 0
        self.rows, self.k, self.key, self.heads = rows, k, key, heads
        self.gate = rows - key - heads
        self.tokens, self.warps, self.split = tokens, warps, split
        self.span = k // (warps * split)
        self.tiles = (tokens + TILE - 1) // TILE
        self.smem = (warps * self.tiles * 32 * 4 + tokens * ROWS) * 4

    @cute.jit
    def __call__(
        self, w: cute.Tensor, x: cute.Tensor, key: cute.Tensor,
        weights: cute.Tensor, gate: cute.Tensor, st: cuda.CUstream,
    ):  # fmt: skip
        self.kernel(w, x, key, weights, gate).launch(
            grid=(self.rows // ROWS * self.split, 1, 1),
            block=(32 * self.warps, 1, 1),
            cluster=(self.split, 1, 1),
            smem=self.smem + 128,
            stream=st,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self, w: cute.Tensor, x: cute.Tensor, key: cute.Tensor,
        weights: cute.Tensor, gate: cute.Tensor,
    ):  # fmt: skip
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        block, _, _ = cute.arch.block_idx()
        cta = block // self.split
        rank = block % self.split
        g = lane // 4
        pairs = self.span // 32
        first = (rank * self.warps + warp) * self.span + lane % 4 * 8
        b = cute.make_rmem_tensor(cute.make_layout((pairs * 4,)), Uint32)
        for p in cutlass.range_constexpr(pairs):
            got = load_vector(
                w.iterator + Int64(cta * ROWS + g) * self.k + first + p * 32,
                cache="weight",
            )
            for i in cutlass.range_constexpr(4):
                b[p * 4 + i] = got[i]
        smem = cutlass.utils.SmemAllocator()
        partial = alloc(smem, Float32, (self.warps, self.tiles, 32, 4))
        sums = alloc(smem, Float32, (self.tokens * ROWS,))

        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        both = self.tokens > 8
        for tile in cutlass.range(self.tiles, unroll=1):
            # Padding tokens repeat the last one; their sums are not stored.
            a = cute.make_rmem_tensor(cute.make_layout((2, pairs * 4)), Uint32)
            for h in cutlass.range_constexpr(2 if both else 1):
                t = tile * TILE + h * 8 + g
                if t >= self.tokens:
                    t = self.tokens - 1
                for p in cutlass.range_constexpr(pairs):
                    got = load_words(
                        x.iterator + t * self.k + first + p * 32,
                        count=4, word_type="u32", mutable=True,
                    )  # fmt: skip
                    for i in cutlass.range_constexpr(4):
                        a[h, p * 4 + i] = got[i]
            # Two accumulators halve the dependent mma chain.
            acc = cute.make_rmem_tensor(cute.make_layout((2, 4)), Float32)
            acc.fill(0.0)
            high = 1 if both else 0
            for j in cutlass.range_constexpr(0, pairs * 4, 2):
                mma_into(
                    acc[j // 2 % 2, None],
                    (a[0, j], a[high, j], a[0, j + 1], a[high, j + 1]),
                    (b[j], b[j + 1]), bf16=True,
                )  # fmt: skip
            for i in cutlass.range_constexpr(4):
                partial[warp, tile, lane, i] = acc[0, i] + acc[1, i]
        cute.arch.sync_threads()

        # Element e = (token, row) of C sits in lane (token % 8) * 4 + row / 2.
        for e in cutlass.range(tid, self.tokens * ROWS, 32 * self.warps, unroll=1):
            t = e // ROWS
            r = e % ROWS
            src = t % 8 * 4 + r // 2
            idx = t % TILE // 8 * 2 + r % 2
            total = Float32(0.0)
            for v in cutlass.range_constexpr(self.warps):
                total = total + partial[v, t // TILE, src, idx]
            if cutlass.const_expr(self.split == 1):
                self.store(key, weights, gate, t, cta * ROWS + r, total)
            else:
                sums[e] = total
        if cutlass.const_expr(self.split > 1):
            # Rank c adds elements c, c + split, ... over the cluster's K parts.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            base = shared_address(sums.iterator)
            for e in cutlass.range(
                tid * self.split + rank, self.tokens * ROWS,
                32 * self.warps * self.split, unroll=1,
            ):  # fmt: skip
                total = Float32(0.0)
                for c in cutlass.range_constexpr(self.split):
                    total = total + _remote_f32(base + e * 4, c)
                self.store(key, weights, gate, e // ROWS, cta * ROWS + e % ROWS, total)
            # Keep this CTA's sums alive until the cluster has read them.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

    @cute.jit
    def store(self, key, weights, gate, t, row, total):
        if row < self.key:
            _store_bf16(key.iterator + t * self.key + row, total)
        elif row < self.key + self.heads:
            weights[t, row - self.key] = total
        else:
            _store_bf16(
                gate.iterator + t * self.gate + row - self.key - self.heads, total
            )


_compiled: dict = {}


class IndexerProjection:
    """(key BF16, head weights FP32, kpool gate BF16) of the layer input."""

    def __init__(self, indexer):
        self.key = indexer.head_dim
        self.heads = indexer.n_head
        gate = indexer.index_kpool_compress_gate.data
        self.gate = gate.shape[0]
        self.weight = torch.cat([indexer.wk_weights_proj.weight.data, gate])
        rows, k = self.weight.shape
        if self.key % ROWS or self.heads % ROWS or rows % ROWS or k % 1024:
            raise ValueError("indexer projection rows or K do not tile")

    def __call__(self, x: torch.Tensor):
        m, k = x.shape
        assert x.is_contiguous()
        key = torch.empty((m, self.key), dtype=torch.bfloat16, device=x.device)
        weights = torch.empty((m, self.heads), dtype=torch.float32, device=x.device)
        gate = torch.empty((m, self.gate), dtype=torch.bfloat16, device=x.device)
        args = (self.weight, x, key, weights, gate)
        st = stream(x.device)
        rows = self.weight.shape[0]
        cache = (rows, k, self.heads, m, x.device.index)
        if cache not in _compiled:
            kernel = _ProjectionKernel(rows, k, self.key, self.heads, m, *_tactic(m))
            _compiled[cache] = cute.compile(kernel, *(to_cute(t, 16) for t in args), st)
        _compiled[cache](*(to_cute(t, 16) for t in args), st)
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
