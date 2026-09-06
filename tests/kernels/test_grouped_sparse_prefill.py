# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped sparse-prefill metadata (union index lists + head-group masks) for DSA."""

import pytest
import torch

from vllm.models.deepseek_v32.nvidia.ops.grouped_sparse_prefill import (
    build_grouped_sparse_prefill,
    grouped_sparse_prefill_reference,
)
from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("CUDA only", allow_module_level=True)


@pytest.mark.parametrize(
    "query_lens,max_seq_len",
    [([5, 17, 64], 4096), ([1000, 2000], 32768), ([9, 1, 300], 8192)],
)
@pytest.mark.parametrize("group", [8, 4])
def test_grouped_metadata_matches_reference(
    query_lens: list[int], max_seq_len: int, group: int
):
    """The Triton builder must reproduce the torch reference exactly: sorted
    union per group, its length, and the per-token membership bitmask in the
    FlashMLA head-group-mask layout, including -1 padding and partial groups."""
    torch.manual_seed(0)
    num_tokens = sum(query_lens)
    topk = 2048
    # unique keys per token (a real indexer never repeats a key); some rows
    # are short (e.g. early positions) and padded with -1.
    scores = torch.rand(num_tokens, max_seq_len, device="cuda")
    idx = scores.argsort(dim=1)[:, :topk].int().contiguous()
    idx[0, topk // 2 :] = -1
    idx[-1, 10:] = -1

    got = build_grouped_sparse_prefill(idx, query_lens, max_seq_len, group=group)
    ref = grouped_sparse_prefill_reference(idx, query_lens, group=group)

    assert torch.equal(got.tok_map, ref.tok_map)
    assert torch.equal(got.req_of_group, ref.req_of_group)
    assert torch.equal(got.ulen, ref.ulen)
    assert torch.equal(got.union, ref.union)
    assert torch.equal(got.mask, ref.mask)
    assert got.union_pad % 128 == 0
    assert got.mask.shape == (got.num_groups, got.union_pad // 128, 128)
