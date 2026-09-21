# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-KV pinning for sliding-window KV-cache groups.

A hybrid model such as MiMo-V2 pairs a handful of full-attention layers with
many narrow sliding-window layers (60 of 70 layers at window 128). The FA4
split-KV heuristic mis-sizes those windows -- see small_window_num_splits() --
and schedules ~128 splits per layer on a 176K context, so each of those layers
pays for a combine pass over 128 fp32 partials plus a prepare-scheduler launch
that reduce nothing.
"""

import pytest

from vllm.v1.attention.backends.flash_attn import (
    _FA_KV_TILE,
    small_window_num_splits,
)


class _Spec:
    """Minimal stand-in for an AttentionSpec's window attribute."""

    def __init__(self, sliding_window=None):
        self.sliding_window = sliding_window


@pytest.mark.parametrize(
    "window,expected",
    [
        (None, None),  # full attention: the heuristic is correct
        (128, 1),  # MiMo-V2's SWA layers: one KV tile, nothing to split
        (64, 1),
        (_FA_KV_TILE, 1),
        (_FA_KV_TILE + 1, None),  # spans >1 tile: leave the heuristic alone
        (1024, None),  # DFlash drafter's window
        (4096, None),
    ],
)
def test_small_window_num_splits(window, expected):
    assert small_window_num_splits(_Spec(window)) is expected


def test_spec_without_window_attribute():
    """Specs that predate sliding_window must not raise."""

    class _Bare:
        pass

    assert small_window_num_splits(_Bare()) is None
