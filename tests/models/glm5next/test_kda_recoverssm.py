# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RecoverSSM on GLM-5.3-Flash KDA matches GLM's native speculative path.

The native path advances one recurrent state per draft position
(``fused_recurrent_kda`` with ``num_spec + 1`` state slots) and keeps the slot
of the accepted length. RecoverSSM verifies the same window off one checkpoint
and reconstructs the accepted state after sampling. Both must give the same
verify outputs and the same committed recurrent state (and, in align mode, the
same state at the crossed Mamba block boundary), with GLM's parameter layout
(``A_log`` as ``[1, 1, H, 1]``, flat ``dt_bias``) and bounded gate.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.glm5next.nvidia.ops import recoverssm as glm_recoverssm
from vllm.models.glm5next.nvidia.ops.recoverssm import (
    GlmKDARecoverSSMCommitContext,
    glm_kda_recoverssm_verify,
)
from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda
from vllm.models.kimi_k3.nvidia.ops import recoverssm as k3_recoverssm

DEVICE = "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("align_mode", [False, True])
@pytest.mark.parametrize("num_spec", [4, 7])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_glm_kda_recoverssm_matches_native_spec(
    monkeypatch: pytest.MonkeyPatch,
    num_spec: int,
    state_dtype: torch.dtype,
    align_mode: bool,
):
    monkeypatch.setattr(k3_recoverssm, "is_conv_state_dim_first", lambda: True)
    assert glm_recoverssm.KDARecoverSSMCommitContext is not None
    torch.manual_seed(0)
    num_heads, dim, lower_bound = 32, 128, -5.0
    query_len = num_spec + 1
    accepted = [1, 3, query_len]
    num_seqs = len(accepted)
    total = num_seqs * query_len

    def rnd(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16, device=DEVICE)

    q, k, v, raw_g = (rnd(1, total, num_heads, dim) for _ in range(4))
    raw_beta = rnd(1, total, num_heads)
    A_log = (0.2 * torch.randn(num_heads, device=DEVICE)).view(1, 1, num_heads, 1)
    dt_bias = 0.1 * torch.randn(num_heads * dim, device=DEVICE)
    cu = torch.arange(0, total + 1, query_len, dtype=torch.int32, device=DEVICE)
    init = (0.05 * torch.randn(num_seqs, num_heads, dim, dim, device=DEVICE)).to(
        state_dtype
    )

    # Native: slot 0 of each row holds the initial state; token t writes slot t.
    native_states = torch.zeros(
        1 + total, num_heads, dim, dim, dtype=state_dtype, device=DEVICE
    )
    slots = (1 + torch.arange(total, device=DEVICE, dtype=torch.int32)).view(
        num_seqs, query_len
    )
    native_states[slots[:, 0]] = init
    native_out, _ = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=raw_g,
        beta=raw_beta,
        initial_state=native_states,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        ssm_state_indices=slots,
        num_accepted_tokens=torch.ones(num_seqs, dtype=torch.int32, device=DEVICE),
        sigmoid_beta=True,
        a_log=A_log,
        g_bias=dt_bias,
        compute_gate=True,
        lower_bound=lower_bound,
    )

    # RecoverSSM: checkpoint blocks 1..n; align mode adds a next block per
    # request so that crossing a Mamba block boundary moves the final state.
    num_blocks = 1 + 3 * num_seqs
    checkpoint = torch.zeros(
        num_blocks, num_heads, dim, dim, dtype=state_dtype, device=DEVICE
    )
    blocks = torch.arange(1, num_seqs + 1, dtype=torch.int32, device=DEVICE)
    checkpoint[blocks] = init
    correction = torch.empty(
        num_blocks, num_heads, query_len, dim, dtype=torch.float32, device=DEVICE
    )
    kd = torch.empty(
        num_blocks, num_heads, query_len, 2 * dim, dtype=torch.float32, device=DEVICE
    )
    out = glm_kda_recoverssm_verify(
        q=q,
        k=k,
        v=v,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log.view(-1),
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        checkpoint_state=checkpoint,
        correction_cache=correction,
        kd_cache=kd,
        query_start_loc=cu,
        state_indices=blocks,
        spec_query_len=query_len,
    )
    torch.testing.assert_close(checkpoint[blocks], init)  # verify is read-only
    torch.testing.assert_close(out.float(), native_out.float(), atol=2e-2, rtol=2e-2)

    conv_dim, history_len = 12, 3
    conv_state = torch.zeros(
        num_blocks,
        conv_dim,
        history_len + query_len - 1,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    layer = SimpleNamespace(
        kv_cache=(conv_state, checkpoint, correction, kd),
        A_log=A_log,
        dt_bias=dt_bias,
        local_num_heads=num_heads,
        head_dim=dim,
        gate_lower_bound=lower_bound,
    )
    context = GlmKDARecoverSSMCommitContext.create(
        [layer], spec_query_len=query_len, max_num_reqs=num_seqs
    )
    num_accepted = torch.tensor(accepted, dtype=torch.int32, device=DEVICE)
    expected_final = {}
    expected_boundary = {}
    if align_mode:
        # Mamba block of 2 * query_len tokens; each request has computed
        # 2 * query_len - 2 tokens, so accepting >= 2 tokens crosses into
        # column 1 and materializes the state after 2 tokens at column 0.
        block_size = 2 * query_len
        num_computed = torch.full(
            (num_seqs,), block_size - 2, dtype=torch.int32, device=DEVICE
        )
        block_table = torch.stack(
            [
                blocks + num_seqs,  # column 0: boundary block
                blocks + 2 * num_seqs,  # column 1: next block
            ],
            dim=1,
        ).to(torch.int32)
        for i, n in enumerate(accepted):
            if n >= 2:
                expected_final[int(block_table[i, 1])] = native_states[slots[i, n - 1]]
                expected_boundary[int(block_table[i, 0])] = native_states[slots[i, 1]]
            else:
                expected_final[int(block_table[i, 0])] = native_states[slots[i, n - 1]]
        context.commit(
            num_accepted,
            blocks,
            cu,
            block_table=block_table,
            num_computed_tokens=num_computed,
            mamba_block_size=block_size,
        )
    else:
        for i, n in enumerate(accepted):
            expected_final[int(blocks[i])] = native_states[slots[i, n - 1]]
        context.commit(num_accepted, blocks, cu)

    tol = 2e-3 if state_dtype == torch.float32 else 2e-2
    for block, expected in {**expected_boundary, **expected_final}.items():
        torch.testing.assert_close(
            checkpoint[block].float(), expected.float(), atol=tol, rtol=tol
        )
