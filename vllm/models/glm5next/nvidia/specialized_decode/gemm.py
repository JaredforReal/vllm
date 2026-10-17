# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skinny BF16 projections for decode batches: the CuTe skinny GEMM at B1/B2 and
the split-K GEMM above, both from the Kimi K3 specialized decode."""

import torch

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (
    PreparedProjection,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.providers import (
    SkinnyProjection,
    projection_policies,
)

TP = 4


class Projection:
    """``x @ weight.T`` for the prepared token counts, into a fresh output."""

    def __init__(self, weight, batches):
        self.weight = weight
        split_k = PreparedProjection(
            weight, projection_policies(TP, weight.shape[0], batches)
        ).prepare()
        self.skinny = SkinnyProjection(weight, split_k)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = x.new_empty((x.shape[0], self.weight.shape[0]))
        return self.skinny.run(x, out)
