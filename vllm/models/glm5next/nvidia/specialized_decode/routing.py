# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash top-k routing of a few decode tokens in one kernel: sigmoid
scores, top-k by score plus the correction bias (a single expert group),
renormalized and scaled weights of the picked scores."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _route_kernel(
    logits_ptr,
    bias_ptr,
    ids_ptr,
    weights_ptr,
    scale,
    EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
    TOP_K: tl.constexpr,
):
    t = tl.program_id(0)
    experts = tl.arange(0, BLOCK)
    live = experts < EXPERTS
    logit = tl.load(logits_ptr + t * EXPERTS + experts, mask=live, other=0.0)
    score = tl.sigmoid(logit.to(tl.float32))
    choice = score + tl.load(bias_ptr + experts, mask=live, other=0.0)
    choice = tl.where(live, choice, float("-inf"))
    picks = tl.arange(0, TOP_K)
    ids = tl.zeros((TOP_K,), dtype=tl.int32)
    picked = tl.zeros((TOP_K,), dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        best = tl.argmax(choice, axis=0)
        ids = tl.where(picks == k, best, ids)
        picked = tl.where(
            picks == k, tl.sum(tl.where(experts == best, score, 0.0)), picked
        )
        choice = tl.where(experts == best, float("-inf"), choice)
    picked = picked / tl.sum(picked, axis=0) * scale
    tl.store(ids_ptr + t * TOP_K + picks, ids)
    tl.store(weights_ptr + t * TOP_K + picks, picked)


def route(logits: torch.Tensor, bias: torch.Tensor, top_k: int, scale: float):
    """(int32 [M, top_k] experts, FP32 [M, top_k] weights)."""
    m, experts = logits.shape
    ids = torch.empty(m, top_k, dtype=torch.int32, device=logits.device)
    weights = torch.empty(m, top_k, dtype=torch.float32, device=logits.device)
    _route_kernel[(m,)](
        logits, bias, ids, weights, scale,
        EXPERTS=experts, BLOCK=triton.next_power_of_2(experts), TOP_K=top_k,
    )  # fmt: skip
    return ids, weights
