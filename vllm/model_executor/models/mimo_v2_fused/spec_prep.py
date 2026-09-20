# SPDX-License-Identifier: Apache-2.0
"""Fused spec-decode step-start index prep (K4).

Fuses two per-request Triton kernels used at the start of every decode step
in vLLM's input preparation:

  - ``_expand_idx_mapping_kernel`` (expanded idx mapping + local positions)
  - ``_combine_sampled_and_draft_tokens_kernel`` (write last sampled + draft
    tokens into input_ids; build logits_indices)

One kernel launch instead of two; pure index math — bit-exact vs running the
two original kernels.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _expand_and_combine_kernel(
    # expand_idx_mapping inputs
    idx_mapping_ptr,  # [num_reqs]
    expanded_idx_mapping_ptr,  # [total_num_logits]
    expanded_local_pos_ptr,  # [total_num_logits]
    # combine_sampled_and_draft_tokens inputs
    input_ids_ptr,  # [max_num_tokens] (mutated)
    last_sampled_tokens_ptr,  # [max_num_reqs]
    query_start_loc_ptr,  # [num_reqs + 1]
    seq_lens_ptr,  # [num_reqs]
    prefill_len_ptr,  # [max_num_reqs]
    draft_tokens_ptr,  # [max_num_reqs, num_spec]
    draft_tokens_stride,
    logits_indices_ptr,  # [total_num_logits]
    # shared
    cu_num_logits_ptr,  # [num_reqs + 1]
    BLOCK_SIZE: tl.constexpr,
    NUM_NEW_SAMPLED_TOKENS: tl.constexpr = 1,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    cu_start = tl.load(cu_num_logits_ptr + batch_idx)
    cu_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = cu_end - cu_start
    num_draft_tokens = num_logits - NUM_NEW_SAMPLED_TOKENS

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < num_logits

    # ---- expand_idx_mapping body ----
    tl.store(expanded_idx_mapping_ptr + cu_start + block, req_state_idx,
             mask=mask)
    tl.store(expanded_local_pos_ptr + cu_start + block, block, mask=mask)

    # ---- combine_sampled_and_draft_tokens body ----
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    logits_start = query_end - num_logits
    tl.store(
        logits_indices_ptr + cu_start + block,
        logits_start + block,
        mask=mask,
    )

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    if seq_len <= prefill_len:
        # Prefill tokens: no sampled or draft tokens to write.
        return

    first_logit_seq_pos = seq_len - num_logits
    if NUM_NEW_SAMPLED_TOKENS > 0 and first_logit_seq_pos >= prefill_len:
        last_token_id = tl.load(last_sampled_tokens_ptr + req_state_idx)
        tl.store(input_ids_ptr + logits_start, last_token_id)

    if num_draft_tokens > 0:
        dmask = block < num_draft_tokens
        draft_tokens = tl.load(
            draft_tokens_ptr + req_state_idx * draft_tokens_stride + block,
            mask=dmask,
        )
        tl.store(
            input_ids_ptr + query_end - num_draft_tokens + block,
            draft_tokens,
            mask=dmask,
        )


def expand_idx_mapping_and_combine_tokens(
    idx_mapping: torch.Tensor,
    total_num_logits: int,
    cu_num_logits: torch.Tensor,
    max_expand_len: int,
    input_ids: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_len: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_new_sampled_tokens: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused expand_idx_mapping + combine_sampled_and_draft_tokens."""
    num_reqs = idx_mapping.shape[0]
    num_speculative_steps = draft_tokens.shape[-1]

    expanded_idx_mapping = idx_mapping.new_empty(total_num_logits)
    expanded_local_pos = torch.empty(
        total_num_logits, dtype=torch.int32, device=idx_mapping.device
    )
    logits_indices = torch.empty(
        total_num_logits, dtype=torch.int64, device=input_ids.device
    )
    _expand_and_combine_kernel[(num_reqs,)](
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        input_ids,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        draft_tokens.stride(0),
        logits_indices,
        cu_num_logits,
        NUM_NEW_SAMPLED_TOKENS=num_new_sampled_tokens,
        BLOCK_SIZE=triton.next_power_of_2(
            max(max_expand_len, num_speculative_steps + num_new_sampled_tokens)
        ),
    )
    return expanded_idx_mapping, expanded_local_pos, logits_indices
