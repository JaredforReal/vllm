# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the PCP head-tail split math (vllm/v1/worker/pcp_manager.py).

These validate the pure CPU/numpy port against vLLM-Ascend's *documented*
example (``pcp_utils.py:544``) and an all-gather + restore round-trip, with no
GPU or process groups required.
"""

import numpy as np

from vllm.v1.worker.pcp_manager import cumsum_and_arange, update_tokens_for_pcp

ARANGE = np.arange(2**16, dtype=np.int32)


def test_cumsum_and_arange():
    cu, ar = cumsum_and_arange(np.array([2, 5, 3], dtype=np.int32), ARANGE)
    assert cu.tolist() == [2, 7, 10]
    assert ar.tolist() == [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]


def test_pcp_split_matches_ascend_documented_example():
    # ascend pcp_utils.py:544: tokens=[1, 5, 8], pcp_world_size=2; req0 is decode.
    tokens = np.array([1, 5, 8], dtype=np.int32)

    r0 = update_tokens_for_pcp(tokens, 2, 0, 1, 1, ARANGE)
    r1 = update_tokens_for_pcp(tokens, 2, 1, 1, 1, ARANGE)

    assert r0.pcp_tokens.tolist() == [1, 4, 4]
    assert r1.pcp_tokens.tolist() == [1, 4, 4]
    assert r0.positions.tolist() == [0, 0, 1, 6, 7, 0, 1, 6, 7]
    assert r1.positions.tolist() == [0, 2, 3, 4, 5, 2, 3, 4, 5]
    assert r0.num_pcp_pads.tolist() == [1, 3, 0]
    assert r0.pcp_allgather_restore_idx.tolist() == [
        0,
        9,
        1,
        2,
        10,
        11,
        12,
        13,
        3,
        4,
        5,
        6,
        14,
        15,
        16,
        17,
        7,
        8,
    ]
    assert r0.pcp_unpad_mask.astype(int).tolist() == [
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]


def test_allgather_restore_is_permutation_roundtrip():
    # After all_gather concatenates per-rank chunks, index_select(buf, restore_idx)
    # must recover natural order. Equivalently, restore_idx is a permutation of
    # range(total_padded).
    for tokens, ps, ndec in [
        (np.array([1, 5, 8], dtype=np.int32), 2, 1),
        (np.array([1, 7, 10, 3], dtype=np.int32), 3, 1),
        (np.array([1, 8, 12], dtype=np.int32), 4, 1),
        (np.array([16], dtype=np.int32), 2, 0),  # pure prefill, no decode
    ]:
        ndec_tokens = int(tokens[:ndec].sum())
        res = update_tokens_for_pcp(tokens, ps, 0, ndec, ndec_tokens, ARANGE)
        restore = res.pcp_allgather_restore_idx.tolist()
        total = restore.__len__()
        assert sorted(restore) == list(range(total)), (tokens, ps)


def test_decode_positions_are_natural_arange():
    # Decode tokens are replicated; their per-rank positions must be the natural
    # arange (independent of pcp rank), not the zigzag chunk math.
    tokens = np.array([1, 1, 8], dtype=np.int32)  # 2 decode reqs (1 token each)
    for r in range(2):
        res = update_tokens_for_pcp(tokens, 2, r, 2, 2, ARANGE)
        # first 2 positions are the decode tokens (req0 pos 0, req1 pos 0)
        assert res.positions[:2].tolist() == [0, 0]
