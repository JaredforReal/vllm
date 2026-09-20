# SPDX-License-Identifier: Apache-2.0
"""Precision test: fused expand_idx_mapping + combine_sampled_and_draft_tokens
vs running the two original vLLM kernels."""

import pytest
import torch

from vllm.model_executor.models.mimo_v2_fused.spec_prep import (
    expand_idx_mapping_and_combine_tokens,
)
from vllm.v1.worker.gpu.input_batch import (
    _combine_sampled_and_draft_tokens_kernel,
    _expand_idx_mapping_kernel,
)
import triton


def run_naive(num_reqs, total_num_logits, num_spec, max_num_reqs, device,
              seed=0, with_draft=True, num_new_sampled=1):
    torch.manual_seed(seed)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)

    # per-request logits counts: 1 sampled + num_spec draft (or 1 prefill)
    num_logits = torch.full((num_reqs,), num_spec + num_new_sampled,
                            dtype=torch.int32, device=device)
    num_logits[0] = 1  # simulate a prefill request at slot 0
    cu_num_logits = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    torch.cumsum(num_logits, 0, out=cu_num_logits[1:])
    total = int(cu_num_logits[-1].item())

    max_num_tokens = total + 8
    input_ids = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
    last_sampled = torch.randint(1000, 2000, (max_num_reqs,), dtype=torch.int64,
                                 device=device)
    draft = torch.randint(0, 50000, (max_num_reqs, num_spec),
                          dtype=torch.int64, device=device)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32,
                                  device=device)
    torch.cumsum(num_logits, 0, out=query_start_loc[1:])
    seq_lens = (query_start_loc[1:] + 100).to(torch.int32)
    prefill_len = torch.full((max_num_reqs,), 100, dtype=torch.int32,
                             device=device)
    prefill_len[0] = 10_000  # request 0 is prefilling: seq <= prefill
    seq_lens[0] = 5000

    # naive: run the two original kernels
    expanded_idx = idx_mapping.new_empty(total)
    expanded_pos = torch.empty(total, dtype=torch.int32, device=device)
    _expand_idx_mapping_kernel[(num_reqs,)](
        idx_mapping, expanded_idx, expanded_pos, cu_num_logits,
        BLOCK_SIZE=triton.next_power_of_2(max(num_spec + num_new_sampled, 2)),
    )
    logits_idx = torch.empty(total, dtype=torch.int64, device=device)
    _combine_sampled_and_draft_tokens_kernel[(num_reqs,)](
        input_ids, idx_mapping, last_sampled, query_start_loc, seq_lens,
        prefill_len, draft, draft.stride(0), cu_num_logits, logits_idx,
        NUM_NEW_SAMPLED_TOKENS=num_new_sampled,
        BLOCK_SIZE=triton.next_power_of_2(num_spec + num_new_sampled),
    )
    return (expanded_idx, expanded_pos, logits_idx, input_ids,
            idx_mapping, last_sampled, draft, query_start_loc, seq_lens,
            prefill_len, cu_num_logits, total)


@pytest.mark.parametrize("num_spec", [8])
@pytest.mark.parametrize("num_reqs", [1, 4])
def test_fused_spec_prep_matches_naive(num_spec, num_reqs):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = "cuda"
    max_num_reqs = num_reqs + 2
    (exp_idx, exp_pos, log_idx, input_ids, idx_mapping, last_sampled, draft,
     qsl, seq_lens, prefill_len, cu_num_logits, total) = run_naive(
         num_reqs, 0, num_spec, max_num_reqs, device)

    # fused: fresh input_ids copy
    input_ids2 = torch.zeros_like(input_ids)
    e2, p2, l2 = expand_idx_mapping_and_combine_tokens(
        idx_mapping, total, cu_num_logits,
        max_expand_len=num_spec + 1,
        input_ids=input_ids2,
        last_sampled_tokens=last_sampled,
        query_start_loc=qsl,
        seq_lens=seq_lens,
        prefill_len=prefill_len,
        draft_tokens=draft,
        num_new_sampled_tokens=1,
    )
    torch.testing.assert_close(e2, exp_idx)
    torch.testing.assert_close(p2, exp_pos)
    torch.testing.assert_close(l2, log_idx)
    torch.testing.assert_close(input_ids2, input_ids)
