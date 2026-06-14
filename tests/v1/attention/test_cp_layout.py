# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm/v1/attention/cp.py layout primitives.

CPU-only: no process groups or GPUs required. These validate the zigzag
split/owner math, the balanced partition, and the all-gather+restore+unpad
round-trip that PCP prefill relies on.
"""

import numpy as np
import pytest

from vllm.v1.attention.cp import (
    CPShardingPolicy,
    pcp_build_layout,
    pcp_gather_logits,
    pcp_logits_owner_rank,
    pcp_zigzag_owner_rank,
    pcp_zigzag_padded_len,
    pcp_zigzag_rank_positions,
)


@pytest.mark.parametrize("pcp_size", [2, 3, 4])
@pytest.mark.parametrize("query_len", [1, 5, 8, 12, 16, 23])
def test_zigzag_rank_positions_partition(query_len, pcp_size):
    """Union over ranks == all padded positions; disjoint; balanced."""
    padded_len = pcp_zigzag_padded_len(query_len, pcp_size)
    chunk = padded_len // (2 * pcp_size)
    all_positions = []
    per_rank_counts = []
    for r in range(pcp_size):
        positions, plen = pcp_zigzag_rank_positions(query_len, pcp_size, r)
        assert plen == padded_len
        assert len(positions) == 2 * chunk  # head + tail
        per_rank_counts.append(len(positions))
        all_positions.extend(positions.tolist())
    # Disjoint and complete cover of [0, padded_len).
    assert sorted(all_positions) == list(range(padded_len))
    # Balanced: every rank gets the same count.
    assert len(set(per_rank_counts)) == 1


@pytest.mark.parametrize("pcp_size", [2, 3, 4])
@pytest.mark.parametrize("query_len", [5, 8, 12, 23])
def test_zigzag_owner_rank_matches_positions(query_len, pcp_size):
    """owner(position) matches which rank's positions contain it."""
    padded_len = pcp_zigzag_padded_len(query_len, pcp_size)
    rank_of_position = {}
    for r in range(pcp_size):
        positions, _ = pcp_zigzag_rank_positions(query_len, pcp_size, r)
        for p in positions.tolist():
            # Each padded position owned by exactly one rank.
            assert p not in rank_of_position
            rank_of_position[p] = r
    for p in range(padded_len):
        assert pcp_zigzag_owner_rank(p, padded_len, pcp_size) == rank_of_position[p]


@pytest.mark.parametrize("pcp_size", [2, 3, 4])
@pytest.mark.parametrize("query_lens", [[5, 8, 3], [12], [7, 7, 7, 1], [16, 23, 4]])
def test_build_layout_balanced_and_roundtrip(query_lens, pcp_size):
    """local_query_lens balanced across ranks; all-gather+restore+unpad
    reconstructs real K/V in natural order."""
    query_lens_np = np.array(query_lens, dtype=np.int32)
    num_reqs = len(query_lens_np)

    layouts = [pcp_build_layout(query_lens_np, pcp_size, r) for r in range(pcp_size)]

    # All ranks have identical batch-level layout fields except pcp_rank.
    for r in range(1, pcp_size):
        assert np.array_equal(
            layouts[r].local_query_lens_cpu, layouts[0].local_query_lens_cpu
        )
        assert layouts[r].num_local_tokens_padded == layouts[0].num_local_tokens_padded

    total_padded = layouts[0].total_tokens_padded
    assert total_padded == sum(
        pcp_zigzag_padded_len(int(q), pcp_size) for q in query_lens
    )
    assert layouts[0].num_local_tokens_padded * pcp_size == total_padded

    # restore_idx length == total_padded (full gathered buffer).
    assert len(layouts[0].restore_idx_cpu) == total_padded
    # unpad_mask length == total_padded (natural padded buffer).
    assert len(layouts[0].unpad_mask_cpu) == total_padded

    # --- Simulate the all-gather + restore + unpad round-trip ---
    # "full padded" identity tensor: full_padded[buf_idx] = buf_idx.
    full_padded = np.arange(total_padded, dtype=np.int32)

    # Recompute each rank's local buffer indices (head+tail per request,
    # sorted, flattened across requests) -- mirrors how the runner slices.
    req_offsets = np.zeros(num_reqs, dtype=np.int32)
    np.cumsum(
        [pcp_zigzag_padded_len(int(q), pcp_size) for q in query_lens][:-1],
        out=req_offsets[1:],
    )
    gathered = []
    for r in range(pcp_size):
        for i in range(num_reqs):
            positions, _ = pcp_zigzag_rank_positions(int(query_lens[i]), pcp_size, r)
            gathered.append(req_offsets[i] + positions)
    gathered = np.concatenate(gathered)  # [rank0_locals | rank1_locals | ...]
    assert len(gathered) == total_padded

    restored = gathered[layouts[0].restore_idx_cpu]
    # Restored == natural padded order.
    np.testing.assert_array_equal(restored, full_padded)

    # unpad drops pad positions -> real tokens in natural order.
    real_natural = restored[layouts[0].unpad_mask_cpu]
    expected_real = np.concatenate(
        [
            np.arange(int(q), dtype=np.int32) + req_offsets[i]
            for i, q in enumerate(query_lens)
        ]
    )
    np.testing.assert_array_equal(real_natural, expected_real)


def test_build_layout_inactive_when_pcp_size_one():
    layout = pcp_build_layout(np.array([5, 8], dtype=np.int32), pcp_size=1, pcp_rank=0)
    assert not layout.active
    # No gather needed -> empty restore_idx.
    assert len(layout.restore_idx_cpu) == 0


def test_logits_owner_rank_covers_all_requests():
    query_lens = np.array([5, 8, 12, 23], dtype=np.int32)
    for pcp_size in (2, 3, 4):
        owners = pcp_logits_owner_rank(query_lens, pcp_size)
        assert len(owners) == len(query_lens)
        # Each owner is a valid rank.
        assert (owners >= 0).all() and (owners < pcp_size).all()
        # Owner matches pcp_zigzag_owner_rank for the last position.
        for i, ql in enumerate(query_lens):
            padded_len = pcp_zigzag_padded_len(int(ql), pcp_size)
            assert owners[i] == pcp_zigzag_owner_rank(int(ql) - 1, padded_len, pcp_size)


def test_policy_only_zigzag_supported():
    with pytest.raises(AssertionError):
        pcp_build_layout(
            np.array([8], dtype=np.int32),
            pcp_size=2,
            pcp_rank=0,
            policy=CPShardingPolicy.CONTINUOUS,
        )


class _FakePCPGroup:
    """Stand-in for a GroupCoordinator: all_gather concatenates a fixed list
    of per-rank tensors (simulating a PCP group)."""

    def __init__(self, rank_tensors, pcp_size, pcp_rank):
        self._rank_tensors = rank_tensors
        self.world_size = pcp_size
        self.rank_in_group = pcp_rank

    def all_gather(self, tensor, dim=0):
        return np.concatenate(self._rank_tensors, axis=dim)


def test_pcp_gather_logits_selects_owner():
    """pcp_gather_logits picks each request's owner-rank row from the gathered
    stack."""
    import torch

    from vllm.v1.attention.cp import CPContext

    pcp_size = 2
    num_reqs = 3
    vocab = 4
    query_lens = np.array([5, 8, 12], dtype=np.int32)
    owners = pcp_logits_owner_rank(query_lens, pcp_size)  # [num_reqs]
    # Each rank produces a [num_reqs, vocab] where row i = rank*100 + i (so we
    # can tell which rank's row was selected).
    rank_tensors = [
        torch.full((num_reqs, vocab), float(r * 100), dtype=torch.float32)
        for r in range(pcp_size)
    ]
    for r in range(pcp_size):
        ctx = CPContext(
            pcp_size=pcp_size,
            pcp_rank=r,
            dcp_size=1,
            dcp_rank=0,
            pcp_group=_FakePCPGroup(rank_tensors, pcp_size, r),  # type: ignore[arg-type]
            dcp_group=None,
            interleave=1,
            dcp_comm_backend="ag_rs",
        )
        out = pcp_gather_logits(
            ctx,
            rank_tensors[r],
            torch.from_numpy(owners.astype(np.int64)),
            num_reqs,
        )
        # Each request's row should equal its owner rank's value.
        for i in range(num_reqs):
            assert out[i, 0].item() == float(owners[i] * 100)
