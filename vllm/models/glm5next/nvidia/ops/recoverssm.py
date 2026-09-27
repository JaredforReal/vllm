# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RecoverSSM speculative verify/commit kernels for GLM-5.3-Flash KDA.

Same algorithm as Kimi-K3's RecoverSSM (one checkpoint per request, per-draft
records, state reconstructed after sampling), with a record layout that moves
all per-token transcendental work out of the serial recurrences:

- correction record ``[block, H, L, V]`` fp32: ``u_t = beta_t (v_t - S k_t)``
- key/decay record ``[block, H, L, 2K]`` fp32: ``[k_t / |k_t|, exp(gate_t)]``

A prep kernel computes the normalized key and decay once per (token, head)
instead of once per value tile; verify then runs only the state recurrence and
the commit is a reduction-free ``S = S * d_t + u_t k_t^T`` stream.
"""

import torch

from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
    KDARecoverSSMCommitContext,
    _compact_conv_state_kernel,
    _prepare_commit_plan_kernel,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


@triton.jit
def _glm_rssm_prep_kernel(
    k_ptr,
    raw_g_ptr,
    A_log_ptr,
    dt_bias_ptr,
    kd_cache_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    lower_bound,
    null_block_id,
    stride_k_token,
    stride_g_token,
    stride_kd_block,
    stride_kd_head,
    stride_kd_pos,
    K: tl.constexpr,
    BK: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    bos = tl.load(query_start_loc_ptr + pid_b).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + pid_b + 1).to(tl.int64)
    state_idx = tl.load(state_indices_ptr + pid_b).to(tl.int64)
    if state_idx <= null_block_id:
        return
    offs_k = tl.arange(0, BK)
    mask_k = offs_k < K
    A = tl.exp(tl.load(A_log_ptr + pid_h).to(tl.float32))
    dt_bias = tl.load(dt_bias_ptr + pid_h * K + offs_k, mask=mask_k, other=0.0).to(
        tl.float32
    )
    kd_ptr = kd_cache_ptr + state_idx * stride_kd_block + pid_h * stride_kd_head
    for pos in tl.static_range(SPEC_QUERY_LEN):
        valid = pos < eos - bos
        token = bos + pos
        k = tl.load(
            k_ptr + token * stride_k_token + pid_h * K + offs_k,
            mask=valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        raw_g = tl.load(
            raw_g_ptr + token * stride_g_token + pid_h * K + offs_k,
            mask=valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        normalized_k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
        # Bounded safe gate, bit-compatible with the native recurrent kernel.
        gate = lower_bound / (1.0 + tl.exp(-(A * (raw_g + dt_bias))))
        tl.store(kd_ptr + pos * stride_kd_pos + offs_k, normalized_k, mask=valid & mask_k)
        tl.store(
            kd_ptr + pos * stride_kd_pos + K + offs_k,
            tl.exp(gate),
            mask=valid & mask_k,
        )


@triton.jit
def _glm_rssm_verify_kernel(
    q_ptr,
    v_ptr,
    raw_beta_ptr,
    state_ptr,
    correction_cache_ptr,
    kd_cache_ptr,
    out_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    null_block_id,
    scale,
    stride_q_token,
    stride_v_token,
    stride_beta_token,
    stride_state_block,
    stride_state_head,
    stride_state_v,
    stride_correction_block,
    stride_correction_head,
    stride_correction_pos,
    stride_kd_block,
    stride_kd_head,
    stride_kd_pos,
    stride_out_token,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    bos = tl.load(query_start_loc_ptr + pid_b).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + pid_b + 1).to(tl.int64)
    query_len = eos - bos
    state_idx = tl.load(state_indices_ptr + pid_b).to(tl.int64)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    if state_idx <= null_block_id:
        for pos in tl.static_range(SPEC_QUERY_LEN):
            tl.store(
                out_ptr + (bos + pos) * stride_out_token + pid_h * V + offs_v,
                tl.zeros([BV], dtype=tl.float32),
                mask=(pos < query_len) & mask_v,
            )
        return

    state = tl.load(
        state_ptr
        + state_idx * stride_state_block
        + pid_h * stride_state_head
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :],
        mask=mask_v[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float32)
    kd_ptr = kd_cache_ptr + state_idx * stride_kd_block + pid_h * stride_kd_head
    correction_ptr = (
        correction_cache_ptr
        + state_idx * stride_correction_block
        + pid_h * stride_correction_head
    )
    for pos in tl.static_range(SPEC_QUERY_LEN):
        valid = pos < query_len
        token = bos + pos
        normalized_k = tl.load(
            kd_ptr + pos * stride_kd_pos + offs_k, mask=valid & mask_k, other=0.0
        )
        decay = tl.load(
            kd_ptr + pos * stride_kd_pos + K + offs_k, mask=valid & mask_k, other=1.0
        )
        q = tl.load(
            q_ptr + token * stride_q_token + pid_h * K + offs_k,
            mask=valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr + token * stride_v_token + pid_h * V + offs_v,
            mask=valid & mask_v,
            other=0.0,
        ).to(tl.float32)
        beta = tl.sigmoid(
            tl.load(raw_beta_ptr + token * stride_beta_token + pid_h, mask=valid, other=0.0).to(
                tl.float32
            )
        )
        q = q / tl.sqrt(tl.sum(q * q) + 1e-6) * scale
        state *= decay[None, :]
        correction = (v - tl.sum(state * normalized_k[None, :], axis=1)) * beta
        state += correction[:, None] * normalized_k[None, :]
        tl.store(
            out_ptr + token * stride_out_token + pid_h * V + offs_v,
            tl.sum(state * q[None, :], axis=1),
            mask=valid & mask_v,
        )
        tl.store(
            correction_ptr + pos * stride_correction_pos + offs_v,
            correction,
            mask=valid & mask_v,
        )


def glm_kda_recoverssm_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    checkpoint_state: torch.Tensor,
    correction_cache: torch.Tensor,
    kd_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Verify a KDA speculative window without modifying its checkpoint.

    Shapes: q/k/raw_g ``[1, T, H, K]``, v/out ``[1, T, H, V]``, raw_beta
    ``[1, T, H]``, checkpoint ``[blocks, H, V, K]``, correction
    ``[blocks, H, L, V]`` fp32, key/decay ``[blocks, H, L, 2K]`` fp32.
    Heads must be contiguous (stride 1 over the last dim).
    """
    _, total_tokens, num_heads, key_dim = q.shape
    value_dim = v.shape[-1]
    batch = state_indices.shape[0]
    if out is None:
        out = torch.empty_like(v)
    if total_tokens == 0 or batch == 0:
        return out
    assert correction_cache.dtype == torch.float32
    assert kd_cache.dtype == torch.float32
    assert kd_cache.shape[-2:] == (spec_query_len, 2 * key_dim)
    assert checkpoint_state.stride(-1) == 1 and kd_cache.stride(-1) == 1
    assert correction_cache.stride(-1) == 1
    block_k = triton.next_power_of_2(key_dim)
    _glm_rssm_prep_kernel[(batch, num_heads)](
        k,
        raw_g,
        A_log,
        dt_bias,
        kd_cache,
        query_start_loc,
        state_indices,
        lower_bound,
        NULL_BLOCK_ID,
        k.stride(1),
        raw_g.stride(1),
        kd_cache.stride(0),
        kd_cache.stride(1),
        kd_cache.stride(2),
        K=key_dim,
        BK=block_k,
        SPEC_QUERY_LEN=spec_query_len,
        num_warps=1,
    )
    block_v = 16
    _glm_rssm_verify_kernel[(triton.cdiv(value_dim, block_v), batch, num_heads)](
        q,
        v,
        raw_beta,
        checkpoint_state,
        correction_cache,
        kd_cache,
        out,
        query_start_loc,
        state_indices,
        NULL_BLOCK_ID,
        key_dim**-0.5,
        q.stride(1),
        v.stride(1),
        raw_beta.stride(1),
        checkpoint_state.stride(0),
        checkpoint_state.stride(1),
        checkpoint_state.stride(2),
        correction_cache.stride(0),
        correction_cache.stride(1),
        correction_cache.stride(2),
        kd_cache.stride(0),
        kd_cache.stride(1),
        kd_cache.stride(2),
        out.stride(1),
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        SPEC_QUERY_LEN=spec_query_len,
        num_warps=1,
    )
    return out


@triton.jit
def _glm_rssm_commit_state_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    correction_ref_ptr,
    correction_base_addrs_ptr,
    correction_block_strides_ptr,
    kd_ref_ptr,
    kd_base_addrs_ptr,
    kd_block_strides_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    stride_state_head,
    stride_state_v,
    stride_correction_head,
    stride_correction_pos,
    stride_kd_head,
    stride_kd_pos,
    stride_state_indices,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_lh = tl.program_id(2)
    pid_l = pid_lh // NUM_HEADS
    pid_h = pid_lh % NUM_HEADS
    source_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)
    if source_idx <= null_block_id:
        return
    commit_len = tl.load(commit_lens_ptr + pid_b)
    if commit_len == 0:
        return
    final_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    if final_idx <= null_block_id:
        return
    boundary_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
    boundary_len = tl.load(boundary_recovery_lens_ptr + pid_b)

    state_ptr = tl.load(state_base_addrs_ptr + pid_l).to(
        tl.pointer_type(state_ref_ptr.dtype.element_ty)
    )
    state_block_stride = tl.load(state_block_strides_ptr + pid_l)
    correction_ptr = tl.load(correction_base_addrs_ptr + pid_l).to(
        tl.pointer_type(correction_ref_ptr.dtype.element_ty)
    ) + (
        source_idx * tl.load(correction_block_strides_ptr + pid_l)
        + pid_h * stride_correction_head
    )
    kd_ptr = tl.load(kd_base_addrs_ptr + pid_l).to(
        tl.pointer_type(kd_ref_ptr.dtype.element_ty)
    ) + (source_idx * tl.load(kd_block_strides_ptr + pid_l) + pid_h * stride_kd_head)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]
    tile = pid_h * stride_state_head + offs_v[:, None] * stride_state_v + offs_k[None, :]
    state = tl.load(
        state_ptr + source_idx * state_block_stride + tile, mask=mask_state, other=0.0
    ).to(tl.float32)
    # Unrolled over the verify window so the record loads are independent of
    # the running state and issue ahead of the FMA chain.
    for pos in tl.static_range(SPEC_QUERY_LEN):
        valid = pos < commit_len
        if ALIGN_MODE:
            tl.store(
                state_ptr + boundary_idx * state_block_stride + tile,
                state,
                mask=mask_state & (boundary_idx > null_block_id) & (pos == boundary_len),
            )
        normalized_k = tl.load(
            kd_ptr + pos * stride_kd_pos + offs_k, mask=valid & mask_k, other=0.0
        )
        decay = tl.load(
            kd_ptr + pos * stride_kd_pos + K + offs_k, mask=valid & mask_k, other=1.0
        )
        correction = tl.load(
            correction_ptr + pos * stride_correction_pos + offs_v,
            mask=valid & mask_v,
            other=0.0,
        )
        state = state * decay[None, :] + correction[:, None] * normalized_k[None, :]
    if ALIGN_MODE:
        # The window may end exactly on the boundary.
        tl.store(
            state_ptr + boundary_idx * state_block_stride + tile,
            state,
            mask=mask_state & (boundary_idx > null_block_id) & (boundary_len == commit_len),
        )
    tl.store(state_ptr + final_idx * state_block_stride + tile, state, mask=mask_state)


class GlmKDARecoverSSMCommitContext(KDARecoverSSMCommitContext):
    """Kimi-K3's grouped commit with GLM's fp32 key/decay records."""

    def commit(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_indices: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,
        mamba_block_size: int | None = None,
    ) -> None:
        batch = state_indices.shape[0]
        if batch == 0:
            return
        if batch > self.commit_lens.shape[0]:
            raise ValueError("KDA RecoverSSM commit batch exceeds its plan capacity")
        if query_start_loc.shape[0] != batch + 1:
            raise ValueError("KDA RecoverSSM commit metadata is incompatible")
        align = block_table is not None
        if align and (num_computed_tokens is None or mamba_block_size is None):
            raise ValueError("KDA RecoverSSM align metadata is incomplete")
        block_table_stride = block_table.stride() if align else (0, 0)
        _prepare_commit_plan_kernel[(batch,)](
            num_accepted_tokens,
            request_indices,
            state_indices,
            query_start_loc,
            block_table,
            num_computed_tokens,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            mamba_block_size or 1,
            block_table.shape[1] if align else 1,
            num_accepted_tokens.stride(0),
            request_indices.stride(0) if request_indices is not None else 0,
            state_indices.stride(0),
            query_start_loc.stride(0),
            block_table_stride[0],
            block_table_stride[1],
            num_computed_tokens.stride(0) if num_computed_tokens is not None else 0,
            SPEC_QUERY_LEN=self.spec_query_len,
            num_warps=1,
        )
        num_layers = len(self.checkpoints)
        conv_ref = self.conv_states[0]
        conv_dim = conv_ref.shape[1]
        _compact_conv_state_kernel[(triton.cdiv(conv_dim, 256), batch, num_layers)](
            conv_ref,
            self.conv_state_base_addrs,
            self.conv_state_block_strides,
            self.conv_state_dim_strides,
            self.conv_state_token_strides,
            state_indices,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            conv_dim,
            self.conv_history_len,
            state_indices.stride(0),
            BLOCK_D=256,
            BLOCK_HISTORY=triton.next_power_of_2(self.conv_history_len),
            ALIGN_MODE=align,
            num_warps=4,
        )
        state_ref = self.checkpoints[0]
        _, num_heads, value_dim, key_dim = state_ref.shape
        block_v = 16
        _glm_rssm_commit_state_kernel[
            (triton.cdiv(value_dim, block_v), batch, num_layers * num_heads)
        ](
            state_ref,
            self.state_base_addrs,
            self.state_block_strides,
            self.correction_caches[0],
            self.correction_cache_base_addrs,
            self.correction_cache_block_strides,
            self.kg_caches[0],
            self.kg_cache_base_addrs,
            self.kg_cache_block_strides,
            state_indices,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            state_ref.stride(1),
            state_ref.stride(2),
            self.correction_caches[0].stride(1),
            self.correction_caches[0].stride(2),
            self.kg_caches[0].stride(1),
            self.kg_caches[0].stride(2),
            state_indices.stride(0),
            K=key_dim,
            V=value_dim,
            BK=triton.next_power_of_2(key_dim),
            BV=block_v,
            NUM_HEADS=num_heads,
            ALIGN_MODE=align,
            SPEC_QUERY_LEN=self.spec_query_len,
            num_warps=2,
        )


__all__ = ["GlmKDARecoverSSMCommitContext", "glm_kda_recoverssm_verify"]
