# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused per-token decode metadata for the sparse indexer / sparse MLA builders.

The fused kernels replace repeat_interleave/arange/searchsorted chains; each
test checks them against the torch reference they replaced, including the
CUDA-graph padding rows that follow the real tokens.
"""

import pytest
import torch

from vllm.v1.attention.backends.mla.indexer import (
    compute_kpool_tail_slot_mapping,
    expand_varlen_decode,
)
from vllm.v1.attention.backends.utils import fill_token_to_req_indices

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)
DEVICE = torch.device("cuda")


def _query_start_loc(query_lens: list[int]) -> torch.Tensor:
    return torch.tensor([0] + query_lens, dtype=torch.int32, device=DEVICE).cumsum(
        0, dtype=torch.int32
    )


@pytest.mark.parametrize(
    "decode_lens,num_decode_tokens",
    [
        ([2, 2, 2], 6),  # uniform, no padding
        ([2, 2, 2], 8),  # uniform, cudagraph token padding
        ([3, 1, 4, 0], 8),  # variable lengths with a zero-length padding request
        ([1], 1),
        ([1, 1, 1], 16),
    ],
)
def test_expand_varlen_decode_matches_reference(decode_lens, num_decode_tokens):
    num_decodes = len(decode_lens)
    width = 7
    query_start_loc = _query_start_loc(decode_lens)
    seq_lens = torch.tensor([10, 7, 12, 0][:num_decodes], dtype=torch.int32)
    seq_lens = torch.cat(
        [seq_lens, torch.full((num_decodes - seq_lens.numel(),), 5, dtype=torch.int32)]
    ).to(DEVICE)
    block_table = torch.randint(
        1, 1000, (num_decodes, width), dtype=torch.int32, device=DEVICE
    )
    actual = sum(decode_lens)

    out_seq_lens = torch.full(
        (num_decode_tokens,), -7, dtype=torch.int32, device=DEVICE
    )
    out_block_table = torch.full(
        (num_decode_tokens, width), -7, dtype=torch.int32, device=DEVICE
    )
    out_decode_lens = torch.zeros(num_decode_tokens, dtype=torch.int32, device=DEVICE)
    out_per_req = torch.zeros(num_decodes, dtype=torch.int32, device=DEVICE)
    out_indices = torch.full((num_decode_tokens,), -7, dtype=torch.int32, device=DEVICE)
    expand_varlen_decode(
        query_start_loc,
        seq_lens,
        block_table,
        out_seq_lens,
        out_block_table,
        out_decode_lens,
        out_per_req,
        out_indices,
        num_decodes,
        actual,
        num_decode_tokens,
    )

    # Reference: the repeat_interleave formulation this kernel replaced.
    lens = torch.tensor(decode_lens, dtype=torch.int32, device=DEVICE)
    offsets = torch.repeat_interleave(
        seq_lens - lens - query_start_loc[:-1], lens, output_size=actual
    )
    ref_seq_lens = torch.zeros(num_decode_tokens, dtype=torch.int32, device=DEVICE)
    ref_seq_lens[:actual] = (
        offsets + torch.arange(actual, dtype=torch.int32, device=DEVICE) + 1
    )
    ref_indices = torch.empty(num_decode_tokens, dtype=torch.int32, device=DEVICE)
    ref_indices[:actual] = torch.repeat_interleave(
        torch.arange(num_decodes, dtype=torch.int32, device=DEVICE),
        lens,
        output_size=actual,
    )
    ref_indices[actual:] = num_decodes + torch.arange(
        num_decode_tokens - actual, dtype=torch.int32, device=DEVICE
    )
    ref_block_rows = torch.repeat_interleave(
        block_table, lens, dim=0, output_size=actual
    )

    torch.testing.assert_close(out_seq_lens, ref_seq_lens)
    torch.testing.assert_close(out_indices, ref_indices)
    torch.testing.assert_close(out_block_table[:actual], ref_block_rows)
    assert (out_block_table[actual:, 0] == 0).all()
    assert (out_decode_lens == 1).all()
    torch.testing.assert_close(out_per_req, lens)


@pytest.mark.parametrize(
    "query_lens,num_tokens",
    [([1, 1, 1], 3), ([2, 5, 0, 3], 10), ([2, 5, 0, 3], 16), ([4], 4)],
)
def test_fill_token_to_req_indices_matches_repeat_interleave(query_lens, num_tokens):
    query_start_loc = _query_start_loc(query_lens)
    num_mapped = sum(query_lens)
    out = torch.full((num_tokens,), -7, dtype=torch.int32, device=DEVICE)
    fill_token_to_req_indices(query_start_loc, out, num_mapped, num_tokens)

    lens = torch.tensor(query_lens, dtype=torch.int32, device=DEVICE)
    ref = torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE)
    ref[:num_mapped] = torch.repeat_interleave(
        torch.arange(len(query_lens), dtype=torch.int32, device=DEVICE),
        lens,
        output_size=num_mapped,
    )
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize(
    "query_lens,num_actual,num_tokens",
    [([3, 2], 5, 5), ([3, 2], 5, 8), ([2, 2, 2], 7, 8), ([1], 1, 1)],
)
def test_kpool_tail_slot_mapping_kernel_matches_torch(
    query_lens, num_actual, num_tokens
):
    """The Triton path (CUDA tensors) must match the torch path (CPU tensors),
    including tokens between the last request boundary and num_actual_tokens
    (mapped to the last request) and untouched padding beyond num_actual."""
    kpool = 4
    num_reqs = len(query_lens)
    query_start_loc = _query_start_loc(query_lens)
    positions = torch.randint(0, 100, (num_tokens,), dtype=torch.int64, device=DEVICE)
    slot_mapping = torch.randint(
        0, 5000, (num_tokens,), dtype=torch.int64, device=DEVICE
    )
    block_table = torch.randint(1, 50, (num_reqs, 3), dtype=torch.int32, device=DEVICE)

    got = compute_kpool_tail_slot_mapping(
        slot_mapping,
        block_table,
        query_start_loc,
        positions,
        num_actual,
        num_reqs,
        kpool,
    )
    ref = compute_kpool_tail_slot_mapping(
        slot_mapping.cpu(),
        block_table.cpu(),
        query_start_loc.cpu(),
        positions.cpu(),
        num_actual,
        num_reqs,
        kpool,
    )
    torch.testing.assert_close(got.cpu(), ref)
