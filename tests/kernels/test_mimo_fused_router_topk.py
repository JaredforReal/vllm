# SPDX-License-Identifier: Apache-2.0
"""Precision test: fused_router_topk vs PyTorch-naive grouped_topk semantics."""

import pytest
import torch

from vllm.model_executor.models.mimo_v2_fused import fused_router_topk

NUM_EXPERTS = 384
HIDDEN = 6144
TOPK = 8


def naive_router(hidden_states, weight, bias, topk=TOPK, renormalize=True):
    """PyTorch naive reference matching grouped_topk with n_group=1."""
    logits = hidden_states.float() @ weight.float().T
    scores = torch.sigmoid(logits)
    biased = scores + bias.unsqueeze(0)
    topk_ids = torch.topk(biased, k=topk, dim=-1).indices
    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


@pytest.mark.parametrize("m", [1, 4, 9, 16])
@pytest.mark.parametrize("seed", [0, 1])
def test_fused_router_topk_precision(m, seed):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(seed)
    device = "cuda"
    x = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device=device)
    w = torch.randn(NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16, device=device)
    bias = torch.randn(NUM_EXPERTS, dtype=torch.float32, device=device) * 0.1

    tw_ref, ti_ref = naive_router(x, w, bias)
    tw, ti = fused_router_topk(x, w, bias, topk=TOPK, renormalize=True)

    assert tw.dtype == torch.float32 and ti.dtype == torch.int32
    assert tw.shape == (m, TOPK) and ti.shape == (m, TOPK)

    # expert selection must match exactly as sets
    for r in range(m):
        assert set(ti[r].tolist()) == set(ti_ref[r].tolist()), \
            f"row {r}: ids {ti[r].tolist()} vs ref {ti_ref[r].tolist()}"

    # weights: compare per-expert values gathered by id (order-free)
    w_map = {int(i): float(v) for i, v in zip(ti[r].tolist(), tw[r].tolist())}
    ref_map = {int(i): float(v) for i, v in
               zip(ti_ref[r].tolist(), tw_ref[r].tolist())}
    for r in range(m):
        for eid in ref_map:
            assert abs(w_map[eid] - ref_map[eid]) < 2e-3, \
                f"row {r} expert {eid}: {w_map[eid]} vs {ref_map[eid]}"


@pytest.mark.parametrize("m", [1, 9])
def test_fused_router_topk_extreme_logits(m):
    """Large-magnitude logits: sigmoid saturation must not flip selection."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(2)
    device = "cuda"
    x = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device=device) * 4
    w = torch.randn(NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16, device=device) * 4
    bias = torch.randn(NUM_EXPERTS, dtype=torch.float32, device=device)

    tw_ref, ti_ref = naive_router(x, w, bias)
    tw, ti = fused_router_topk(x, w, bias, topk=TOPK, renormalize=True)
    for r in range(m):
        assert set(ti[r].tolist()) == set(ti_ref[r].tolist())
