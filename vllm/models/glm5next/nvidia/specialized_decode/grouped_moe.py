# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed experts of up to 16 decode tokens as one persistent kernel, grouped by
expert, on the TRTLLM-Gen FP8 weights (see ``mega_moe``).

One CTA per SM, no clusters: 16-CTA clusters are GPC-bound (7 fit on GB300), so
a kernel with a cluster per route runs two waves already at 8 routes. Every CTA
redoes the routing and numbers the distinct experts ("slots") in expert order,
so all CTAs agree on the task list without communicating:

- GEMM1 task (slot, row tile, K chunk): the slot's tokens are the UMMA N side
  (swap-AB). K block ``kb`` puts token ``j`` in column ``(kb % G) * W + j``, so
  one 16-column TMEM accumulator holds ``G`` unscaled K-block partials that the
  epilogue scales (FP32 weight block scale times the token's activation scale)
  and sums. The FP32 partial goes to a global workspace; the last of the 2 * C
  tasks of an (up tile, gate tile) pair sums them in a fixed order, applies the
  clamped SwiGLU, quantizes the hidden block to FP8 per 128 and publishes it.
- GEMM2 task (slot, output row tile): waits for the slot's four hidden blocks,
  then writes each route's BF16 output row (finalize-ready, like ``mega_moe``).

Each CTA takes tasks ``cta, cta + grid, ...`` with every GEMM1 task before any
GEMM2 task, so a waiting GEMM2 task only waits on GEMM1 tasks of resident CTAs
that have no GEMM2 work ahead of them. The last CTA to finish zeroes the
counters for the next call.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32, Uint64

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
    stream,
    to_cute,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    acquire_tensormap_asm,
    alloc,
    asm,
    bits_float,
    div_rn,
    fence_tmem,
    float_bits,
    load_bf16x8,
    load_tma3,
    load_tmem,
    pack_fp8,
    redux,
    release_tmem,
    sigmoid,
    smem_desc,
    view,
    wait,
    wait_tmem,
)

from .mega_moe import (
    B_BYTES,
    FP8_MAX,
    IDESC,
    STAGE_BYTES,
    TILE,
    _commit,
    _logical_rows,
    _ordered,
    _store_bf16,
    _umma_block,
    _weight_map,
)

COLUMNS = 16
MAX_TOKENS = 16
TMEM_COLUMNS = 256  # two task buffers of up to 8 accumulators
SMEM_LIMIT = 227 * 1024


@asm(Uint32)
def _atomic_add(pointer, value):
    return "atom.add.acq_rel.gpu.global.u32 $0,[$1],$2;", "=r,l,r", (pointer, value)


@asm(Uint32)
def _load_acquire(pointer):
    return "ld.acquire.gpu.global.b32 $0,[$1];", "=r,l", (pointer,)


@asm()
def _fence():
    return "fence.acq_rel.gpu;", "~{memory}", ()


@asm(Float32)
def _load_l2(pointer):
    return "ld.global.cg.f32 $0,[$1];", "=f,l", (pointer,)


@asm((Uint32,) * 4)
def _load_l2_words(pointer):
    return "ld.global.cg.v4.b32 {$0,$1,$2,$3},[$4];", "=r,=r,=r,=r,l", (pointer,)


@asm()
def _store_u32(pointer, value):
    return "st.global.b32 [$0],$1;", "l,r", (pointer, Uint32(value))


@asm()
def _store_f32(pointer, value):
    return "st.global.f32 [$0],$1;", "l,f", (pointer, value)


@asm()
def _store_u8(pointer, value):
    return "st.global.u8 [$0],$1;", "l,r", (pointer, Uint32(value))


@asm(Uint32, pure=True)
def _popc(value):
    return "popc.b32 $0,$1;", "=r,r", (Uint32(value),)


def _plan(tokens: int, nk1: int) -> tuple[int, int, int]:
    """(W token columns per K block, G K blocks per accumulator, C K chunks)."""
    width = 1
    while width < tokens:
        width *= 2
    group = COLUMNS // width
    # At most 8 accumulators per task; more chunks only at higher token counts.
    chunks = {1: 2, 2: 4, 4: 4}.get(tokens, 0)
    if not chunks:
        chunks = max(4, nk1 // (8 * group))
    assert nk1 % chunks == 0 and (nk1 // chunks + group - 1) // group <= 8
    return width, group, chunks


class _GroupedKernel:
    threads = 192  # producer, MMA, 4 epilogue warps

    def __init__(
        self, *, hidden, inter, experts, top_k, limit, scale, ue8m0, tokens, grid,
        shared,
    ):  # fmt: skip
        assert tokens <= MAX_TOKENS and experts % 32 == 0
        self.hidden, self.inter, self.experts = hidden, inter, experts
        self.top_k, self.limit, self.scale, self.ue8m0 = top_k, limit, scale, ue8m0
        self.tokens, self.grid, self.shared = tokens, grid, shared
        self.nk1, self.nk2 = hidden // TILE, inter // TILE
        self.tiles1 = 2 * inter // TILE
        self.pairs = self.nk2
        assert self.tiles1 == 2 * self.pairs
        self.tiles2 = hidden // TILE
        self.width, self.group, self.chunks = _plan(tokens, self.nk1)
        self.blocks1 = self.nk1 // self.chunks
        self.accs1 = (self.blocks1 + self.group - 1) // self.group
        self.accs2 = (self.nk2 + self.group - 1) // self.group
        # The shared expert, if folded in, is the last slot with every token.
        self.slots = min(tokens * top_k, experts) + int(shared)
        routes = tokens * top_k
        w = self.width
        b_buffer = max(self.blocks1, self.nk2) * B_BYTES
        tables = (
            2 * b_buffer  # B tiles of two tasks
            + TILE * w * 4  # gate values of a pair, by logical row
            + tokens * self.nk1 * 4  # activation scales
            + self.slots * w * 4  # route of (slot, j)
            + experts * 4 * 2  # token mask and slot of each expert
            + self.slots * 4 * 2  # expert and token count of each slot
            + routes * 4  # picks
            + 4 * w * 4  # amax per warp and token
            + w * 4 * 4  # hidden scales of a GEMM2 task
            + 64  # flags, slot count
        )
        fixed = tables + 512 + 1024
        self.stages = min(10, (SMEM_LIMIT - fixed) // STAGE_BYTES)
        assert self.stages >= 4
        offset = 0

        def take(size, align=16):
            nonlocal offset
            offset = (offset + align - 1) // align * align
            start = offset
            offset += size
            return start

        self.off_stage = take(self.stages * STAGE_BYTES, 1024)
        self.b_buffer = b_buffer
        self.off_b = take(2 * b_buffer, 1024)
        self.off_gate = take(TILE * w * 4)
        self.off_xscale = take(tokens * self.nk1 * 4)
        self.off_route = take(self.slots * w * 4)
        self.off_mask = take(experts * 4)
        self.off_slot_of = take(experts * 4)
        self.off_slot_expert = take(self.slots * 4)
        self.off_slot_count = take(self.slots * 4)
        self.off_picks = take(routes * 4)
        self.off_amax = take(4 * w * 4)
        self.off_hscale = take(w * 4 * 4)
        self.off_flags = take(64)
        self.bars = 2 * self.stages + 8
        self.off_bar = take(self.bars * 8, 8)
        self.off_tmem = take(16)
        self.smem = offset

    @cute.jit
    def __call__(
        self, maps: cute.Tensor, w13_scale: cute.Tensor, w2_scale: cute.Tensor,
        x: cute.Tensor, logits: cute.Tensor, bias: cute.Tensor, rows1: cute.Tensor,
        rows2: cute.Tensor, out: cute.Tensor, ids: cute.Tensor, weights: cute.Tensor,
        partial: cute.Tensor, hidden: cute.Tensor, hscale: cute.Tensor,
        counters: cute.Tensor, shared_scale1: cute.Tensor, shared_scale2: cute.Tensor,
        shared_out: cute.Tensor, st: cuda.CUstream,
    ):  # fmt: skip
        self.kernel(
            maps,
            w13_scale,
            w2_scale,
            x,
            logits,
            bias,
            rows1,
            rows2,
            out,
            ids,
            weights,
            partial,
            hidden,
            hscale,
            counters,
            shared_scale1,
            shared_scale2,
            shared_out,
        ).launch(  # fmt: skip
            grid=(self.grid, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.smem + 1024,
            stream=st,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self, maps: cute.Tensor, w13_scale: cute.Tensor, w2_scale: cute.Tensor,
        x: cute.Tensor, logits: cute.Tensor, bias: cute.Tensor, rows1: cute.Tensor,
        rows2: cute.Tensor, out: cute.Tensor, ids: cute.Tensor, weights: cute.Tensor,
        partial: cute.Tensor, hidden: cute.Tensor, hscale: cute.Tensor,
        counters: cute.Tensor, shared_scale1: cute.Tensor, shared_scale2: cute.Tensor,
        shared_out: cute.Tensor,
    ):  # fmt: skip
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        cta, _, _ = cute.arch.block_idx()
        stages = self.stages

        sm = alloc(cutlass.utils.SmemAllocator(), Uint8, self.smem, 1024)
        bars = view(sm.iterator, self.off_bar, Uint64, self.bars)
        full = bars.iterator
        empty = full + stages
        b_full = empty + stages  # 2, one arrival per epilogue warp
        acc_full = b_full + 2  # 2, MMA commit
        acc_empty = acc_full + 2  # 2, one arrival per epilogue warp
        tmem_base = view(sm.iterator, self.off_tmem, Uint32, 1)
        mask = view(sm.iterator, self.off_mask, Uint32, self.experts)
        slot_of = view(sm.iterator, self.off_slot_of, Int32, self.experts)
        slot_expert = view(sm.iterator, self.off_slot_expert, Int32, self.slots)
        slot_count = view(sm.iterator, self.off_slot_count, Int32, self.slots)
        picks = view(sm.iterator, self.off_picks, Int32, self.tokens * self.top_k)
        flags = view(sm.iterator, self.off_flags, Int32, 16)

        offsets = cute.make_rmem_tensor(
            cute.make_layout((self.experts // 32,)), Float32
        )
        for j in cutlass.range_constexpr(self.experts // 32):
            offsets[j] = bias[j * 32 + lane]
        for e in cutlass.range(tid, self.experts, self.threads, unroll=1):
            mask[e] = Uint32(0)
        if tid == 0:
            for s in cutlass.range_constexpr(stages):
                cute.arch.mbarrier_init(full + s, 1)
                cute.arch.mbarrier_init(empty + s, 1)
            for b in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(b_full + b, 4)
                cute.arch.mbarrier_init(acc_full + b, 1)
                cute.arch.mbarrier_init(acc_empty + b, 4)
            cute.arch.mbarrier_init_fence()
            acquire_tensormap_asm(maps.iterator)
            acquire_tensormap_asm(maps.iterator + 128)
            if cutlass.const_expr(self.shared):
                acquire_tensormap_asm(maps.iterator + 256)
                acquire_tensormap_asm(maps.iterator + 384)
        if warp == 1:
            cute.arch.alloc_tmem(TMEM_COLUMNS, tmem_base.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        fence_tmem(before=True)
        cute.arch.sync_threads()
        fence_tmem()
        tmem = tmem_base[0]

        # No early launch_dependents: the finalize publish reads the routing
        # weights this kernel writes before its own PDL wait.
        cute.arch.griddepcontrol_wait()
        self.route(logits, offsets, ids, weights, mask, picks, warp, lane, cta)
        cute.arch.sync_threads()
        self.number_slots(mask, slot_of, slot_expert, slot_count, flags, warp, lane)
        cute.arch.sync_threads()
        route_of = view(sm.iterator, self.off_route, Int32, self.slots * self.width)
        for r in cutlass.range(tid, self.tokens * self.top_k, self.threads, unroll=1):
            expert = picks[r]
            token = r // self.top_k
            below = mask[expert] & ((Uint32(1) << Uint32(token)) - Uint32(1))
            route_of[slot_of[expert] * self.width + Int32(_popc(below))] = r
        shared_slot = Int32(-1)
        if cutlass.const_expr(self.shared):
            shared_slot = flags[0] - 1
            if tid < self.tokens:
                route_of[shared_slot * self.width + tid] = tid * self.top_k
        cute.arch.sync_threads()
        slots = flags[0]
        tasks1 = slots * self.tiles1 * self.chunks
        tasks = tasks1 + slots * self.tiles2

        if warp == 0:
            if lane == 0:
                self.produce(sm, maps, full, empty, slot_expert, cta, tasks1, tasks,
                             shared_slot)  # fmt: skip
        elif warp == 1:
            self.issue(sm, tmem, full, empty, b_full, acc_full, acc_empty, cta,
                       tasks1, tasks)  # fmt: skip
        else:
            self.epilogue(
                sm, tmem, warp, lane, cta, tasks1, tasks, b_full, acc_full, acc_empty,
                slot_expert, slot_count, route_of, flags, w13_scale, w2_scale, x,
                rows1, rows2, out, partial, hidden, hscale, counters, shared_slot,
                shared_scale1, shared_scale2, shared_out,
            )  # fmt: skip
        release_tmem(tmem, TMEM_COLUMNS, warp, 1)
        if tid == 0:
            _fence()
            done = counters.iterator + self.slots * (self.pairs + 1)
            if _atomic_add(done, Uint32(1)) == Uint32(self.grid - 1):
                for i in cutlass.range(self.slots * (self.pairs + 1) + 1, unroll=1):
                    _store_u32(counters.iterator + i, 0)

    @cute.jit
    def route(self, logits, offsets, ids, weights, mask, picks, warp, lane, cta):
        """Top-k of every token (sigmoid scores plus bias, ties to the lower
        expert, weights renormalized and scaled), one warp per token; lane l
        holds experts l + 32 j."""
        per = self.experts // 32
        for token in cutlass.range(warp, self.tokens, self.threads // 32, unroll=1):
            scores = cute.make_rmem_tensor(cute.make_layout((per,)), Float32)
            keys = cute.make_rmem_tensor(cute.make_layout((per,)), Uint32)
            for j in cutlass.range_constexpr(per):
                scores[j] = Float32(logits[token, j * 32 + lane])
            for j in cutlass.range_constexpr(per):
                scores[j] = sigmoid(scores[j], fast_math=True)
                keys[j] = _ordered(scores[j] + offsets[j])
            total = Float32(0.0)
            mine = Int32(0)
            my_score = Float32(0.0)
            for r in cutlass.range_constexpr(self.top_k):
                best = Uint32(0)
                for j in cutlass.range_constexpr(per):
                    if keys[j] > best:
                        best = keys[j]
                top = redux(best, "max")
                candidate = Uint32(0xFFFFFFFF)
                for j in cutlass.range_constexpr(per - 1, -1, -1):
                    if keys[j] == top:
                        candidate = Uint32(j * 32) + Uint32(lane)
                winner = Int32(redux(candidate, "min"))
                score = Float32(0.0)
                for j in cutlass.range_constexpr(per):
                    if winner == j * 32 + lane:
                        score = scores[j]
                        keys[j] = Uint32(0)
                score = cute.arch.shuffle_sync(score, winner % 32)
                total = total + score
                if lane == r:
                    mine = winner
                    my_score = score
            if lane < self.top_k:
                route = token * self.top_k + lane
                picks[route] = mine
                cute.arch.atomic_or(
                    mask.iterator + mine, Uint32(1) << Uint32(token)
                )  # fmt: skip
                if cta == 0:
                    ids[route] = mine
                    weights[route] = div_rn(my_score, total) * Float32(self.scale)

    @cute.jit
    def number_slots(self, mask, slot_of, slot_expert, slot_count, flags, warp, lane):
        """Slots of the routed experts in expert order (warp 0): lane l numbers
        experts l * per ... l * per + per - 1."""
        if warp == 0:
            per = self.experts // 32
            count = Int32(0)
            for j in cutlass.range_constexpr(per):
                if mask[lane * per + j] != Uint32(0):
                    count = count + 1
            prefix = count
            for shift in cutlass.range_constexpr(5):
                other = cute.arch.shuffle_sync_up(prefix, offset=1 << shift)
                if lane >= (1 << shift):
                    prefix = prefix + other
            slot = prefix - count
            for j in cutlass.range_constexpr(per):
                e = lane * per + j
                tokens = mask[e]
                if tokens != Uint32(0):
                    slot_of[e] = slot
                    slot_expert[slot] = e
                    slot_count[slot] = Int32(_popc(tokens))
                    slot = slot + 1
            if lane == 31:
                if cutlass.const_expr(self.shared):
                    slot_expert[prefix] = 0
                    slot_count[prefix] = self.tokens
                    prefix = prefix + 1
                flags[0] = prefix

    @cute.jit
    def task(self, t, tasks1):
        """(is GEMM1, slot, row tile, K chunk) of task t."""
        first = t < tasks1
        slot = Int32(0)
        tile = Int32(0)
        chunk = Int32(0)
        if first:
            slot = t // (self.tiles1 * self.chunks)
            rest = t % (self.tiles1 * self.chunks)
            tile = rest // self.chunks
            chunk = rest % self.chunks
        else:
            slot = (t - tasks1) // self.tiles2
            tile = (t - tasks1) % self.tiles2
        return first, slot, tile, chunk

    @cute.jit
    def produce(self, sm, maps, full, empty, slot_expert, cta, tasks1, tasks,
                shared_slot):  # fmt: skip
        n = Int32(0)
        for t in cutlass.range(cta, tasks, self.grid, unroll=1):
            first, slot, tile, chunk = self.task(t, tasks1)
            expert = slot_expert[slot]
            table = maps.iterator
            if slot == shared_slot:
                table = maps.iterator + 256
            blocks = self.blocks1 if first else self.nk2
            for kb in cutlass.range(blocks, unroll=1):
                stage = n % self.stages
                if n >= self.stages:
                    wait(empty + stage, ((n // self.stages) - 1) % 2, scope="cta")
                cute.arch.mbarrier_arrive_and_expect_tx(full + stage, STAGE_BYTES)
                dst = sm.iterator + self.off_stage + stage * STAGE_BYTES
                if first:
                    load_tma3(
                        dst, table, full + stage, Int32(tile * TILE),
                        Int32(expert * self.nk1 + chunk * self.blocks1 + kb),
                    )  # fmt: skip
                else:
                    load_tma3(
                        dst, table + 128, full + stage, Int32(tile * TILE),
                        Int32(expert * self.nk2 + kb),
                    )  # fmt: skip
                n = n + 1

    @cute.jit
    def issue(self, sm, tmem, full, empty, b_full, acc_full, acc_empty, cta, tasks1,
              tasks):  # fmt: skip
        adesc = smem_desc(sm.iterator + self.off_stage, True)
        bdesc = smem_desc(sm.iterator + self.off_b, True)
        n = Int32(0)
        i = Int32(0)
        for t in cutlass.range(cta, tasks, self.grid, unroll=1):
            buffer = i % 2
            wait(b_full + buffer, (i // 2) % 2, scope="cta")
            if i >= 2:
                wait_tmem(acc_empty + buffer, ((i // 2) - 1) % 2, scope="cta")
            blocks = self.blocks1 if t < tasks1 else self.nk2
            d = tmem + buffer * (TMEM_COLUMNS // 2)
            b = bdesc + buffer * (self.b_buffer >> 4)
            for kb in cutlass.range(blocks, unroll=1):
                stage = n % self.stages
                wait_tmem(full + stage, (n // self.stages) % 2, scope="cta")
                _umma_block(
                    d + (kb // self.group) * COLUMNS,
                    adesc + stage * (STAGE_BYTES >> 4), b + kb * (B_BYTES >> 4),
                    IDESC, kb % self.group, empty + stage,
                )  # fmt: skip
                cute.arch.sync_warp()
                n = n + 1
            _commit(acc_full + buffer)
            i = i + 1

    @cute.jit
    def epilogue(
        self, sm, tmem, warp, lane, cta, tasks1, tasks, b_full, acc_full, acc_empty,
        slot_expert, slot_count, route_of, flags, w13_scale, w2_scale, x, rows1,
        rows2, out, partial, hidden, hscale, counters, shared_slot, shared_scale1,
        shared_scale2, shared_out,
    ):  # fmt: skip
        e = (warp - 2) * 32 + lane
        quadrant = warp % 4
        row = quadrant * 32 + lane
        lanes = tmem + (Uint32(quadrant * 32) << 16)
        xscale = view(sm.iterator, self.off_xscale, Float32, self.tokens * self.nk1)
        i = Int32(0)
        if cta < tasks:
            self.build(sm, cta, tasks1, 0, e, lane, slot_count, route_of, xscale, x,
                       hidden, hscale, counters)  # fmt: skip
        for t in cutlass.range(cta, tasks, self.grid, unroll=1):
            buffer = i % 2
            following = t + self.grid
            # Build the next GEMM1 task's B tiles first; a GEMM2 task's wait for
            # hidden blocks must not precede this task's own publication.
            early = following < tasks1
            if early:
                self.build(sm, following, tasks1, 1 - buffer, e, lane, slot_count,
                           route_of, xscale, x, hidden, hscale, counters)  # fmt: skip
            wait_tmem(acc_full + buffer, (i // 2) % 2, scope="cta")
            first, slot, tile, chunk = self.task(t, tasks1)
            d = lanes + buffer * (TMEM_COLUMNS // 2)
            if first:
                self.finish1(
                    sm, d, slot, tile, chunk, row, lane, warp, quadrant, e, acc_empty,
                    buffer, slot_expert, slot_count, route_of, xscale, w13_scale,
                    rows1, partial, hidden, hscale, counters, slot == shared_slot,
                    shared_scale1,
                )  # fmt: skip
            else:
                self.finish2(
                    sm, d, slot, tile, row, lane, acc_empty, buffer, slot_expert,
                    slot_count, route_of, w2_scale, rows2, out, slot == shared_slot,
                    shared_scale2, shared_out,
                )  # fmt: skip
            if not early and following < tasks:
                self.build(sm, following, tasks1, 1 - buffer, e, lane, slot_count,
                           route_of, xscale, x, hidden, hscale, counters)  # fmt: skip
            i = i + 1

    @cute.jit
    def build(self, sm, t, tasks1, buffer, e, lane, slot_count, route_of, xscale, x,
              hidden, hscale, counters):  # fmt: skip
        """B tiles of task t: K block kb, token j at row (kb % G) * W + j, the rest
        zero; rows are 128B-swizzled (16B chunk c at c ^ (row % 8)). GEMM1
        quantizes the BF16 tokens per 128 (like per_token_group_quant_fp8, with
        its UE8M0 rounding when the stock path uses it); GEMM2 copies the
        published hidden blocks."""
        first, slot, tile, chunk = self.task(t, tasks1)
        count = slot_count[slot]
        base = cute.recast_ptr(
            sm.iterator + self.off_b + buffer * self.b_buffer, dtype=Uint32
        )
        w, g = self.width, self.group
        if first:
            for it in cutlass.range_constexpr(self.blocks1):
                q = it * 128 + e
                kb = q // 128
                row = (q // 8) % COLUMNS
                c = q % 8
                j = row - (kb % g) * w
                live = j >= 0 and j < w and j < count
                values = cute.make_rmem_tensor(cute.make_layout((16,)), Float32)
                for v in cutlass.range_constexpr(16):
                    values[v] = Float32(0.0)
                token = Int32(0)
                if live:
                    token = route_of[slot * w + j] // self.top_k
                    source = (
                        x.iterator + Int64(token) * self.hidden
                        + (chunk * self.blocks1 + kb) * TILE + c * 16
                    )  # fmt: skip
                    for h in cutlass.range_constexpr(2):
                        got = load_bf16x8(source + h * 8, mutable=True)
                        for v in cutlass.range_constexpr(8):
                            values[h * 8 + v] = got[v]
                amax = Float32(1e-10)
                for v in cutlass.range_constexpr(16):
                    amax = cute.arch.fmax(amax, cute.arch.fmax(values[v], -values[v]))
                for offset in cutlass.range_constexpr(3):
                    amax = cute.arch.fmax(
                        amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << offset)
                    )
                x_scale = div_rn(amax, Float32(FP8_MAX))
                if cutlass.const_expr(self.ue8m0):
                    bits = Uint32(float_bits(cute.arch.fmax(x_scale, Float32(1e-10))))
                    exponent = (bits >> 23) + Uint32((bits & Uint32(0x7FFFFF)) != 0)
                    x_scale = bits_float(exponent << 23)
                inverse = div_rn(Float32(1.0), x_scale)
                target = base + (kb * B_BYTES + row * TILE + (c ^ (row % 8)) * 16) // 4
                for v in cutlass.range_constexpr(4):
                    word = pack_fp8(
                        tuple(values[4 * v + u] * inverse for u in range(4))
                    )
                    cute.arch.store((target + v).llvm_ptr, word)
                if live and c == 0:
                    xscale[token * self.nk1 + chunk * self.blocks1 + kb] = x_scale
        else:
            if e == 0:
                ready = counters.iterator + self.slots * self.pairs + slot
                while _load_acquire(ready) < Uint32(self.pairs):
                    pass
            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            _fence()
            hs = view(sm.iterator, self.off_hscale, Float32, w * 4)
            if e < w * self.nk2:
                hs[e] = _load_l2(hscale.iterator + slot * w * self.nk2 + e)
            for it in cutlass.range_constexpr(self.nk2 * COLUMNS * 8 // 128):
                q = it * 128 + e
                kb = q // 128
                row = (q // 8) % COLUMNS
                c = q % 8
                j = row - (kb % g) * w
                words = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                if j >= 0 and j < w and j < count:
                    words = _load_l2_words(
                        hidden.iterator
                        + (slot * w + j) * self.inter
                        + kb * TILE
                        + c * 16
                    )
                target = base + (kb * B_BYTES + row * TILE + (c ^ (row % 8)) * 16) // 4
                for v in cutlass.range_constexpr(4):
                    cute.arch.store((target + v).llvm_ptr, words[v])
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        if lane == 0:
            cute.arch.mbarrier_arrive(
                cute.recast_ptr(sm.iterator + self.off_bar, dtype=Uint64)
                + 2 * self.stages
                + buffer
            )

    @cute.jit
    def sums(self, d, blocks, accs, scales):
        """W FP32 sums of this row: K block kb's token-j partial (accumulator
        kb // G, column (kb % G) * W + j) times scales[kb, j]."""
        w, g = self.width, self.group
        total = cute.make_rmem_tensor(cute.make_layout((w,)), Float32)
        for j in cutlass.range_constexpr(w):
            total[j] = Float32(0.0)
        for a in cutlass.range_constexpr(accs):
            parts = load_tmem(d + a * COLUMNS, count=COLUMNS)
            for k in cutlass.range_constexpr(g):
                kb = a * g + k
                if cutlass.const_expr(kb < blocks):
                    for j in cutlass.range_constexpr(w):
                        total[j] = total[j] + parts[k * w + j] * scales[kb * w + j]
        return total

    @cute.jit
    def finish1(
        self, sm, d, slot, tile, chunk, row, lane, warp, quadrant, e, acc_empty,
        buffer, slot_expert, slot_count, route_of, xscale, w13_scale, rows1, partial,
        hidden, hscale, counters, is_shared, shared_scale1,
    ):  # fmt: skip
        w = self.width
        expert = slot_expert[slot]
        count = slot_count[slot]
        scales = cute.make_rmem_tensor(cute.make_layout((self.blocks1 * w,)), Float32)
        first = (Int64(expert) * self.tiles1 + tile) * self.nk1 + chunk * self.blocks1
        for kb in cutlass.range_constexpr(self.blocks1):
            ws = Float32(0.0)
            if is_shared:
                ws = cute.arch.load(
                    (shared_scale1.iterator + first + kb).llvm_ptr, Float32
                )
            else:
                ws = cute.arch.load((w13_scale.iterator + first + kb).llvm_ptr, Float32)
            for j in cutlass.range_constexpr(w):
                token = route_of[slot * w + cutlass.min(j, count - 1)] // self.top_k
                scales[kb * w + j] = (
                    ws * xscale[token * self.nk1 + chunk * self.blocks1 + kb]
                )
        total = self.sums(d, self.blocks1, self.accs1, scales)
        fence_tmem(before=True)
        cute.arch.sync_warp()
        if lane == 0:
            cute.arch.mbarrier_arrive(acc_empty + buffer)
        piece = partial.iterator + (
            (Int64(slot) * self.tiles1 + tile) * self.chunks + chunk
        ) * (w * TILE)
        for j in cutlass.range_constexpr(w):
            _store_f32(piece + j * TILE + row, total[j])
        _fence()
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        flags = view(sm.iterator, self.off_flags, Int32, 16)
        pair = tile % self.pairs
        if e == 0:
            arrivals = counters.iterator + slot * self.pairs + pair
            flags[1] = Int32(_atomic_add(arrivals, Uint32(1)))
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        if flags[1] == 2 * self.chunks - 1:
            _fence()
            self.hidden_block(sm, slot, pair, row, lane, quadrant, e, count, rows1,
                              partial, hidden, hscale, counters)  # fmt: skip

    @cute.jit
    def hidden_block(self, sm, slot, pair, row, lane, quadrant, e, count, rows1,
                     partial, hidden, hscale, counters):  # fmt: skip
        """Sum the pair's partials in chunk order, clamped SwiGLU, FP8 per 128 per
        token (scale amax / 448), publish hidden block ``pair``."""
        w = self.width
        gate_buf = view(sm.iterator, self.off_gate, Float32, TILE * w)
        amax_buf = view(sm.iterator, self.off_amax, Float32, 4 * w)
        gate_tile = pair + self.pairs
        logical = rows1[gate_tile * TILE + row] - gate_tile * TILE
        for j in cutlass.range_constexpr(w):
            value = Float32(0.0)
            for c in cutlass.range_constexpr(self.chunks):
                piece = partial.iterator + (
                    (Int64(slot) * self.tiles1 + gate_tile) * self.chunks + c
                ) * (w * TILE)
                value = value + _load_l2(piece + j * TILE + row)
            gate_buf[logical * w + j] = value
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        logical = rows1[pair * TILE + row] - pair * TILE
        limit = Float32(self.limit)
        act = cute.make_rmem_tensor(cute.make_layout((w,)), Float32)
        for j in cutlass.range_constexpr(w):
            up = Float32(0.0)
            for c in cutlass.range_constexpr(self.chunks):
                piece = partial.iterator + (
                    (Int64(slot) * self.tiles1 + pair) * self.chunks + c
                ) * (w * TILE)
                up = up + _load_l2(piece + j * TILE + row)
            gate = cute.arch.fmin(gate_buf[logical * w + j], limit)
            up = cute.arch.fmax(cute.arch.fmin(up, limit), -limit)
            act[j] = gate / (Float32(1.0) + cute.math.exp(-gate)) * up
            amax = bits_float(redux(float_bits(cute.arch.fmax(act[j], -act[j])), "max"))
            if lane == 0:
                amax_buf[quadrant * w + j] = amax
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        for j in cutlass.range_constexpr(w):
            if j < count:
                amax = cute.arch.fmax(
                    cute.arch.fmax(amax_buf[j], amax_buf[w + j]),
                    cute.arch.fmax(amax_buf[2 * w + j], amax_buf[3 * w + j]),
                )
                scale = cute.arch.fmax(amax, Float32(1e-10)) / Float32(FP8_MAX)
                byte = pack_fp8(
                    (act[j] / scale, Float32(0.0), Float32(0.0), Float32(0.0))
                )
                _store_u8(
                    hidden.iterator
                    + (slot * w + j) * self.inter
                    + pair * TILE
                    + logical,
                    byte & Uint32(0xFF),
                )
                if e == 0:
                    _store_f32(
                        hscale.iterator + (slot * w + j) * self.nk2 + pair, scale
                    )
        _fence()
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        if e == 0:
            _atomic_add(counters.iterator + self.slots * self.pairs + slot, Uint32(1))

    @cute.jit
    def finish2(self, sm, d, slot, tile, row, lane, acc_empty, buffer, slot_expert,
                slot_count, route_of, w2_scale, rows2, out, is_shared, shared_scale2,
                shared_out):  # fmt: skip
        w = self.width
        expert = slot_expert[slot]
        count = slot_count[slot]
        hs = view(sm.iterator, self.off_hscale, Float32, w * 4)
        scales = cute.make_rmem_tensor(cute.make_layout((self.nk2 * w,)), Float32)
        first = (Int64(expert) * self.tiles2 + tile) * self.nk2
        for kb in cutlass.range_constexpr(self.nk2):
            ws = Float32(0.0)
            if is_shared:
                ws = cute.arch.load(
                    (shared_scale2.iterator + first + kb).llvm_ptr, Float32
                )
            else:
                ws = cute.arch.load((w2_scale.iterator + first + kb).llvm_ptr, Float32)
            for j in cutlass.range_constexpr(w):
                scales[kb * w + j] = ws * hs[j * self.nk2 + kb]
        total = self.sums(d, self.nk2, self.accs2, scales)
        fence_tmem(before=True)
        cute.arch.sync_warp()
        if lane == 0:
            cute.arch.mbarrier_arrive(acc_empty + buffer)
        logical = rows2[tile * TILE + row]
        for j in cutlass.range_constexpr(w):
            if j < count:
                route = route_of[slot * w + j]
                if is_shared:
                    _store_bf16(
                        shared_out.iterator + Int64(j) * self.hidden + logical, total[j]
                    )
                else:
                    _store_bf16(
                        out.iterator + Int64(route) * self.hidden + logical, total[j]
                    )


_workspaces: dict = {}


class GroupedExperts:
    """Routed experts of up to 16 tokens: routing, GEMM1, clamped SwiGLU, FP8
    hidden and GEMM2 of every distinct expert in one persistent kernel."""

    def __init__(
        self, w13, w13_scale, w2, w2_scale, top_k, limit, scale, ue8m0, shared=None
    ):
        """``shared``: the shared expert's TRTLLM-Gen-layout (w13, FP32 scales, w2,
        FP32 scales) with one expert each (see ``shared_weights``), or None."""
        experts, nk1, n1, _ = w13.shape
        self.hidden, self.inter, self.experts = nk1 * TILE, n1 // 2, experts
        self.top_k, self.limit, self.scale, self.ue8m0 = top_k, limit, scale, ue8m0
        self.w13, self.w2 = w13, w2
        self.w13_scale = w13_scale.contiguous()
        self.w2_scale = w2_scale.contiguous()
        maps = [_weight_map(w13, TILE), _weight_map(w2, TILE)]
        self.shared = shared
        if shared is not None:
            maps += [_weight_map(shared[0], TILE), _weight_map(shared[2], TILE)]
            self.shared_scales = (shared[1].contiguous(), shared[3].contiguous())
        else:
            self.shared_scales = (self.w13_scale, self.w2_scale)
        self.maps = torch.cat(maps)
        self.rows1 = _logical_rows(n1, w13.device)
        self.rows2 = _logical_rows(w2.shape[2], w13.device)
        self.grid = torch.cuda.get_device_properties(w13.device).multi_processor_count
        self.compiled: dict = {}

    def prepare(self, tokens: list[int], logits_dtype: torch.dtype) -> None:
        """Compiles the kernel for these token counts before graph capture."""
        device = self.w13.device
        bias = torch.zeros(self.experts, device=device)
        for m in tokens:
            x = torch.zeros(m, self.hidden, dtype=torch.bfloat16, device=device)
            logits = torch.zeros(m, self.experts, dtype=logits_dtype, device=device)
            self(x, logits, bias)

    def workspace(self, tokens, device):
        kernel = self.kernels(tokens)
        key = (device.index, tokens, self.hidden, self.inter)
        if key not in _workspaces:
            slots, w = kernel.slots, kernel.width
            _workspaces[key] = (
                torch.empty(
                    slots * kernel.tiles1 * kernel.chunks * w * TILE,
                    dtype=torch.float32,
                    device=device,
                ),  # fmt: skip
                torch.empty(slots * w * self.inter, dtype=torch.uint8, device=device),
                torch.empty(slots * w * kernel.nk2, dtype=torch.float32, device=device),
                torch.zeros(
                    slots * (kernel.pairs + 1) + 1, dtype=torch.int32, device=device
                ),  # fmt: skip
            )
        return _workspaces[key]

    def kernels(self, tokens):
        return _GroupedKernel(
            hidden=self.hidden, inter=self.inter, experts=self.experts,
            top_k=self.top_k, limit=self.limit, scale=self.scale, ue8m0=self.ue8m0,
            tokens=tokens, grid=self.grid, shared=self.shared is not None,
        )  # fmt: skip

    def __call__(self, x, logits, bias):
        """``MegaExperts.__call__``'s outputs, then the shared expert's BF16
        [M, hidden] output (None unless folded in)."""
        m = x.shape[0]
        routes = m * self.top_k
        device = x.device
        out = torch.empty(routes, self.hidden, dtype=torch.bfloat16, device=device)
        ids = torch.empty(m, self.top_k, dtype=torch.int32, device=device)
        weights = torch.empty(m, self.top_k, dtype=torch.float32, device=device)
        shared = torch.empty(
            (m if self.shared is not None else 1, self.hidden),
            dtype=torch.bfloat16, device=device,
        )  # fmt: skip
        args = (
            self.maps, self.w13_scale, self.w2_scale, x, logits, bias, self.rows1,
            self.rows2, out, ids.view(-1), weights.view(-1),
            *self.workspace(m, device), *self.shared_scales, shared,
        )  # fmt: skip
        st = stream(device)
        key = (m, logits.dtype)
        if key not in self.compiled:
            self.compiled[key] = cute.compile(
                self.kernels(m), *(to_cute(t, 16) for t in args), st
            )
        self.compiled[key](*(to_cute(t, 16) for t in args), st)
        return out, ids, weights, shared if self.shared is not None else None


def shared_weights(w13, w13_scale, w2, w2_scale):
    """The shared expert ([gate; up] FP8 [2I, H] and [H, I] with FP32 block
    scales) as a one-expert TRTLLM-Gen BlockMajorK layout."""
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        _shuffle_deepseek_fp8_moe_weights,
        swap_w13_to_w31,
    )

    w31 = swap_w13_to_w31(w13[None])
    w31_shuffled, w2_shuffled = _shuffle_deepseek_fp8_moe_weights(w31, w2[None])
    return (
        w31_shuffled,
        swap_w13_to_w31(w13_scale[None]).contiguous(),
        w2_shuffled,
        w2_scale[None].contiguous(),
    )
