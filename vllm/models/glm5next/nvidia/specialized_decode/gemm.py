# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skinny BF16 projections for 1-4 token decode steps: the CuTe skinny GEMM that
issues its static weight loads before the PDL wait.

At 8+ tokens cuBLAS stays: the skinny GEMM falls behind it there, and the Kimi K3
split-K tactics measured slower than cuBLAS on the GLM-5.3-Flash KDA in_proj at
B4/B8 (14.4 vs 10.7 us).
"""

import torch

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)

# (block size, outputs per block, K unroll) per token count, by weight (N, K),
# from a GB300 sweep with the weights rotated out of L2.
TACTICS = {
    (6416, 4096): {1: (64, 4, 2), 2: (64, 8, 2), 4: (64, 8, 2)},  # KDA in_proj
    (4096, 2048): {1: (64, 8, 2), 2: (128, 4, 2), 4: (64, 8, 2)},  # KDA o_proj
    (4096, 4096): {1: (64, 4, 2), 2: (128, 2, 1), 4: (64, 2, 2)},  # DSA o_proj
}
DEFAULT = {1: (64, 4, 2), 2: (64, 8, 2), 4: (64, 8, 2)}


class Projection:
    """``x @ weight.T`` for 1, 2 or 4 tokens."""

    def __init__(self, weight):
        self.weight = weight
        n, k = weight.shape
        self.configs = {}
        for m, (block, outputs, unroll) in TACTICS.get((n, k), DEFAULT).items():
            if k % (block * 8) or n % outputs:
                continue
            cfg = SkinnyGemmConfig(m, block, outputs, unroll, 8, k)
            shape_dynamic_skinny_gemm(weight.new_zeros(m, k), weight, cfg, None)
            self.configs[m] = cfg  # compiled before capture

    def supports(self, tokens: int) -> bool:
        return tokens in self.configs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return shape_dynamic_skinny_gemm(x, self.weight, self.configs[x.shape[0]], None)
