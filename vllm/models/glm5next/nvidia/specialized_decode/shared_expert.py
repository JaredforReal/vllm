# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The shared expert of a few decode tokens beside the mega MoE kernel.

The mega kernel's 16-CTA clusters take most of every GPC's SMs, and DeepGEMM's
large-shared-memory CTAs cannot share an SM with them, so the two slow each
other down. These two GEMVs use no shared memory and ask for the largest
carveout, so they run on SMs the clusters already occupy. They read the
DeepGEMM-prepared FP8 weights (UE8M0 block scales, in place) and follow the stock
rounding: FP8 activations per 128 with the stock scale rule, FP32 sums, BF16
gate/up and SwiGLU outputs.

Lane l of a warp owns 16-byte chunk l of every 512 bytes of a weight row, so the
8 lanes of an activation group of 128 are adjacent: the group's absolute max is
three shuffles away. Each warp loads its rows into registers before the PDL wait;
tokens run in a loop, so the register count (and the room left for the mega
kernel's CTAs) does not grow with them.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int64, Uint32

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
    _checked,
    resolve_kernel,
    stream,
    to_cute,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    asm,
    bits_float,
    div_rn,
    float_bits,
    load_bf16x8,
    load_vector,
    pack_fp8,
    warp_sum,
)

from .mega_moe import _store_bf16

GROUP = 128
WARPS = 4
FP8_MAX = 448.0


@asm((Float32,) * 4, pure=True)
def _fp8x4(word):
    """The four E4M3 values of ``word`` (low byte first) as FP32."""
    code = "{ .reg .b16 lo,hi,a,b,c,d; .reg .b32 h0,h1; mov.b32 {lo,hi},$4; "
    code += "cvt.rn.f16x2.e4m3x2 h0,lo; cvt.rn.f16x2.e4m3x2 h1,hi; "
    code += "mov.b32 {a,b},h0; mov.b32 {c,d},h1; cvt.f32.f16 $0,a; "
    code += "cvt.f32.f16 $1,b; cvt.f32.f16 $2,c; cvt.f32.f16 $3,d; }"
    return code, "=f,=f,=f,=f,r", (word,)


@asm(Float32, pure=True)
def _round_bf16(value):
    code = "{ .reg .b16 h,z; mov.b16 z,0; cvt.rn.bf16.f32 h,$1; mov.b32 $0,{z,h}; }"
    return code, "=f,f", (value,)


@cute.jit
def _quantize(values, ue8m0: cutlass.Constexpr[bool]):
    """FP8-rounded ``values`` (16 per lane, 8 lanes per group of 128) and the
    group scale: amax / 448, rounded up to a power of two when ``ue8m0``."""
    amax = Float32(1e-10)
    for v in cutlass.range_constexpr(16):
        amax = cute.arch.fmax(amax, cute.arch.fmax(values[v], -values[v]))
    for shift in cutlass.range_constexpr(3):
        amax = cute.arch.fmax(
            amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << shift)
        )
    scale = div_rn(amax, Float32(FP8_MAX))
    if cutlass.const_expr(ue8m0):
        bits = Uint32(float_bits(cute.arch.fmax(scale, Float32(1e-10))))
        scale = bits_float(
            ((bits >> 23) + Uint32((bits & Uint32(0x7FFFFF)) != 0)) << 23
        )
    inverse = div_rn(Float32(1.0), scale)
    for w in cutlass.range_constexpr(4):
        word = pack_fp8(tuple(values[4 * w + v] * inverse for v in range(4)))
        rounded = _fp8x4(word)
        for v in cutlass.range_constexpr(4):
            values[4 * w + v] = rounded[v]
    return scale


@cute.jit
def _dot(words, values):
    """Sum of 16 FP8 weights (four words) times 16 FP32 values."""
    total = Float32(0.0)
    for w in cutlass.range_constexpr(4):
        weights = _fp8x4(words[w])
        for v in cutlass.range_constexpr(4):
            total = total + weights[v] * values[4 * w + v]
    return total


class _GateUpKernel:
    """BF16 [tokens, n] = FP8-quantized x times the FP8 [n, k] gate/up weight."""

    def __init__(self, n: int, k: int, tokens: int, rows: int, ue8m0: bool):
        assert k % 512 == 0 and n % (WARPS * rows) == 0
        self.n, self.k, self.tokens, self.rows, self.ue8m0 = n, k, tokens, rows, ue8m0
        self.chunks = k // 512  # 16-byte chunks per lane per row
        self.ctas = n // (WARPS * rows)

    @cute.jit
    def __call__(
        self, w: cute.Tensor, s: cute.Tensor, x: cute.Tensor, out: cute.Tensor,
        st: cuda.CUstream,
    ):  # fmt: skip
        self.kernel(w, s, x, out).launch(
            grid=(self.ctas, 1, 1), block=(32 * WARPS, 1, 1), stream=st, use_pdl=True
        )

    @cute.kernel
    def kernel(self, w: cute.Tensor, s: cute.Tensor, x: cute.Tensor, out: cute.Tensor):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        cta, _, _ = cute.arch.block_idx()
        first = (cta * WARPS + warp) * self.rows
        chunks = self.chunks
        words = cute.make_rmem_tensor(cute.make_layout((self.rows, chunks, 4)), Uint32)
        scales = cute.make_rmem_tensor(cute.make_layout((self.rows, chunks)), Float32)
        for r in cutlass.range_constexpr(self.rows):
            row = Int64(first + r)
            for c in cutlass.range_constexpr(chunks):
                got = load_vector(
                    w.iterator + row * self.k + c * 512 + lane * 16, cache="weight"
                )
                for i in cutlass.range_constexpr(4):
                    words[r, c, i] = got[i]
                scales[r, c] = s[(first + r) // GROUP, c * 4 + lane // 8]

        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        for t in cutlass.range(self.tokens, unroll=1):
            acc = cute.make_rmem_tensor(cute.make_layout((self.rows,)), Float32)
            for r in cutlass.range_constexpr(self.rows):
                acc[r] = Float32(0.0)
            for c in cutlass.range_constexpr(chunks):
                values = cute.make_rmem_tensor(cute.make_layout((16,)), Float32)
                source = x.iterator + t * self.k + c * 512 + lane * 16
                for h in cutlass.range_constexpr(2):
                    activations = load_bf16x8(source + h * 8, mutable=True)
                    for v in cutlass.range_constexpr(8):
                        values[h * 8 + v] = activations[v]
                x_scale = _quantize(values, self.ue8m0)
                for r in cutlass.range_constexpr(self.rows):
                    part = _dot([words[r, c, i] for i in range(4)], values)
                    acc[r] = acc[r] + part * (scales[r, c] * x_scale)
            for r in cutlass.range_constexpr(self.rows):
                total = warp_sum(acc[r])
                if lane == 0:
                    _store_bf16(out.iterator + t * self.n + first + r, total)


class _DownKernel:
    """BF16 [tokens, n] = clamped SwiGLU of the BF16 gate/up rows, FP8-quantized,
    times the FP8 [n, 512] down weight."""

    def __init__(
        self, n: int, inter: int, tokens: int, rows: int, limit: float, ue8m0: bool
    ):
        assert inter == 512 and n % (WARPS * rows) == 0
        self.n, self.inter, self.tokens, self.rows = n, inter, tokens, rows
        self.limit, self.ue8m0 = limit, ue8m0
        self.ctas = n // (WARPS * rows)

    @cute.jit
    def __call__(
        self, w: cute.Tensor, s: cute.Tensor, gate_up: cute.Tensor, out: cute.Tensor,
        st: cuda.CUstream,
    ):  # fmt: skip
        self.kernel(w, s, gate_up, out).launch(
            grid=(self.ctas, 1, 1), block=(32 * WARPS, 1, 1), stream=st, use_pdl=True
        )

    @cute.kernel
    def kernel(
        self, w: cute.Tensor, s: cute.Tensor, gate_up: cute.Tensor, out: cute.Tensor
    ):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        cta, _, _ = cute.arch.block_idx()
        first = (cta * WARPS + warp) * self.rows
        words = cute.make_rmem_tensor(cute.make_layout((self.rows, 4)), Uint32)
        scales = cute.make_rmem_tensor(cute.make_layout((self.rows,)), Float32)
        for r in cutlass.range_constexpr(self.rows):
            got = load_vector(
                w.iterator + Int64(first + r) * self.inter + lane * 16, cache="weight"
            )
            for i in cutlass.range_constexpr(4):
                words[r, i] = got[i]
            scales[r] = s[(first + r) // GROUP, lane // 8]

        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        limit = Float32(self.limit)
        for t in cutlass.range(self.tokens, unroll=1):
            values = cute.make_rmem_tensor(cute.make_layout((16,)), Float32)
            base = gate_up.iterator + t * 2 * self.inter + lane * 16
            for h in cutlass.range_constexpr(2):
                gates = load_bf16x8(base + h * 8, mutable=True)
                ups = load_bf16x8(base + self.inter + h * 8, mutable=True)
                for v in cutlass.range_constexpr(8):
                    gate = cute.arch.fmin(gates[v], limit)
                    up = cute.arch.fmax(cute.arch.fmin(ups[v], limit), -limit)
                    silu = div_rn(
                        gate, Float32(1.0) + cute.math.exp(-gate, fastmath=False)
                    )
                    values[h * 8 + v] = _round_bf16(silu * up)
            h_scale = _quantize(values, self.ue8m0)
            for r in cutlass.range_constexpr(self.rows):
                part = _dot([words[r, i] for i in range(4)], values)
                total = warp_sum(part * (scales[r] * h_scale))
                if lane == 0:
                    _store_bf16(out.iterator + t * self.n + first + r, total)


def block_scales(packed: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """FP32 [n / 128, k / 128] block scales from DeepGEMM's packed UE8M0 layout
    (int32 [n, k / 512], byte g % 4 of column g // 4 holds K group g)."""
    groups = k // GROUP
    index = torch.arange(groups, device=packed.device)
    exponents = (packed[:, index // 4] >> (index % 4 * 8)) & 0xFF
    rows = exponents.view(n // GROUP, GROUP, groups)
    assert bool((rows == rows[:, :1]).all()), "scales vary inside a 128-row block"
    return (rows[:, 0].to(torch.int32) << 23).view(torch.float32).contiguous()


def _largest_carveout(compiled, device: int) -> None:
    """An SM configured for a small shared-memory carveout cannot take a mega MoE
    CTA until it drains: ask for the largest one."""
    _, function = resolve_kernel(compiled, device)
    attribute = cuda.CUfunction_attribute
    carveout = attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
    _checked((cuda.cuFuncSetAttribute(function, carveout, 100)[0],))


class SharedExpert:
    """FP8 block-scaled gate_up GEMV, then clamped SwiGLU and the down GEMV."""

    def __init__(self, w13, w13_scale, w2, w2_scale, limit: float, ue8m0: bool):
        self.w13, self.w2 = w13.contiguous(), w2.contiguous()
        self.n1, self.k = self.w13.shape
        self.n2, self.inter = self.w2.shape
        assert self.n1 == 2 * self.inter
        self.s13 = (
            w13_scale if w13_scale.dtype == torch.float32
            else block_scales(w13_scale, self.n1, self.k)
        )  # fmt: skip
        self.s2 = (
            w2_scale if w2_scale.dtype == torch.float32
            else block_scales(w2_scale, self.n2, self.inter)
        )  # fmt: skip
        self.limit, self.ue8m0 = limit, ue8m0
        self.compiled: dict = {}

    def prepare(self, tokens: list[int]) -> None:
        """Compiles for these token counts before graph capture."""
        for m in tokens:
            self(torch.zeros(m, self.k, dtype=torch.bfloat16, device=self.w13.device))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        m = x.shape[0]
        gate_up = torch.empty(m, self.n1, dtype=torch.bfloat16, device=x.device)
        out = torch.empty(m, self.n2, dtype=torch.bfloat16, device=x.device)
        first = (self.w13, self.s13, x, gate_up)
        second = (self.w2, self.s2, gate_up, out)
        st = stream(x.device)
        if m not in self.compiled:
            kernels = (
                _GateUpKernel(self.n1, self.k, m, 2, self.ue8m0),
                _DownKernel(self.n2, self.inter, m, 8, self.limit, self.ue8m0),
            )
            self.compiled[m] = tuple(
                cute.compile(kernel, *(to_cute(t, 16) for t in args), st)
                for kernel, args in zip(kernels, (first, second))
            )
            for compiled in self.compiled[m]:
                _largest_carveout(compiled, x.device.index)
        for compiled, args in zip(self.compiled[m], (first, second)):
            compiled(*(to_cute(t, 16) for t in args), st)
        return out
