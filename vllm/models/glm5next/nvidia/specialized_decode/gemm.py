# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skinny BF16 projections for B1/B2 decode steps: the CuTe skinny GEMM that
issues its static weight loads before the PDL wait.

Above B2 cuBLAS stays: the Kimi K3 split-K tactics measured slower than cuBLAS
on the GLM-5.3-Flash KDA in_proj at B4/B8 (14.4 vs 10.7 us).
"""

import torch

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)

# (BLOCK_N, warps, split K) per token count, from Kimi K3's SkinnyProjection.
TACTICS = {1: (128, 4, 1), 2: (64, 4, 2)}


class Projection:
    """``x @ weight.T`` for 1 or 2 tokens."""

    def __init__(self, weight):
        self.weight, k = weight, weight.shape[1]
        self.configs = {m: SkinnyGemmConfig(m, *t, 8, k) for m, t in TACTICS.items()}
        for m, cfg in self.configs.items():  # compile before capture
            shape_dynamic_skinny_gemm(weight.new_zeros(m, k), weight, cfg, None)

    def supports(self, tokens: int) -> bool:
        return tokens in self.configs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return shape_dynamic_skinny_gemm(x, self.weight, self.configs[x.shape[0]], None)
