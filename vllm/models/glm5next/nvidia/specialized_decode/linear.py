# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skinny projections for stock BF16 linear modules inside planned decode steps.

Stages that still call stock modules (the DSA attention) reach their BF16
projections through ``quant_method.apply``; ``route`` wraps that call so a planned
step's prepared token counts take the skinny / split-K projection, and every other
call (prefills, stock steps, unprepared counts) the stock GEMM.
"""

from contextlib import contextmanager

import torch

from .gemm import Projection

_active = [False]


@contextmanager
def planned_step():
    _active[0] = True
    try:
        yield
    finally:
        _active[0] = False


class _RoutedApply:
    def __init__(self, inner, projection, batches):
        self.inner, self.projection, self.batches = inner, projection, batches

    def apply(self, layer, x, bias=None):
        if (
            _active[0]
            and bias is None
            and x.dim() == 2
            and x.shape[0] in self.batches
            and x.stride(-1) == 1
        ):
            return self.projection(x)
        return self.inner.apply(layer, x, bias)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def route(module, batches) -> bool:
    """Routes a BF16 linear module's decode GEMMs; False if it cannot."""
    weight = getattr(module, "weight", None)
    if weight is None or weight.dtype != torch.bfloat16 or weight.dim() != 2:
        return False
    if getattr(module, "_use_min_latency_gemm", False):
        return False
    if isinstance(module.quant_method, _RoutedApply):
        return True
    module.quant_method = _RoutedApply(
        module.quant_method, Projection(weight, batches), frozenset(batches)
    )
    return True
