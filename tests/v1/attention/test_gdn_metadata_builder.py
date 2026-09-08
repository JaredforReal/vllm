# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GDNAttentionMetadataBuilder.build() — specifically the
reclassification of non-spec decodes as prefills when spec decodes exist.
Covers the fix for https://github.com/vllm-project/vllm/issues/34845.
"""

from dataclasses import dataclass

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    mamba_get_block_table_tensor,
)
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


@dataclass
class GDNBuildTestCase:
    """Specification for a GDN metadata builder classification test."""

    seq_lens: list[int]
    query_lens: list[int]
    num_decode_draft_tokens: list[int] | None  # None = no spec config
    num_speculative_tokens: int
    expected_num_decodes: int
    expected_num_prefills: int
    expected_num_prefill_tokens: int
    expected_num_spec_decodes: int


GDN_BUILD_TEST_CASES = {
    # The original #34845 crash: non-spec query_len=1 + spec decode
    "mixed_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[65, 20],
        query_lens=[1, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
    # All requests are spec decodes — no reclassification needed
    "pure_spec_decode": GDNBuildTestCase(
        seq_lens=[50, 30],
        query_lens=[3, 3],
        num_decode_draft_tokens=[2, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=2,
    ),
    # No speculative config at all — standard decode path
    "pure_regular_decode": GDNBuildTestCase(
        seq_lens=[40, 30, 20],
        query_lens=[1, 1, 1],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=3,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=0,
    ),
    # Multi-token prefill alongside spec decode — no decode to reclassify
    "spec_decode_with_real_prefill": GDNBuildTestCase(
        seq_lens=[100, 20],
        query_lens=[50, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=1,
    ),
    # All three types in one batch — decode gets reclassified
    "prefill_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[100, 65, 20],
        query_lens=[50, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=2,
        expected_num_prefill_tokens=51,
        expected_num_spec_decodes=1,
    ),
    # Multiple non-spec query_len=1 requests all reclassified
    "multiple_decodes_reclassified": GDNBuildTestCase(
        seq_lens=[40, 50, 60, 20],
        query_lens=[1, 1, 1, 3],
        num_decode_draft_tokens=[-1, -1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=3,
        expected_num_prefill_tokens=3,
        expected_num_spec_decodes=1,
    ),
    # Zero-length padded sequence excluded from counts
    "zero_length_padding_with_spec": GDNBuildTestCase(
        seq_lens=[16, 65, 20],
        query_lens=[0, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
}


def _create_gdn_builder(
    num_speculative_tokens: int = 0,
    full_cuda_graph: bool = False,
) -> GDNAttentionMetadataBuilder:
    """Create a GDNAttentionMetadataBuilder with minimal config."""
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
    )
    if full_cuda_graph:
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    if num_speculative_tokens > 0:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _build(
    builder: GDNAttentionMetadataBuilder,
    batch_spec: BatchSpec,
    num_decode_draft_tokens: list[int] | None = None,
) -> GDNAttentionMetadata:
    """Build GDN attention metadata, optionally with spec-decode kwargs."""
    common = create_common_attn_metadata(batch_spec, BLOCK_SIZE, DEVICE)
    kwargs: dict = {}
    if num_decode_draft_tokens is not None:
        kwargs["num_decode_draft_tokens_cpu"] = torch.tensor(
            num_decode_draft_tokens, dtype=torch.int32
        )
        kwargs["num_accepted_tokens"] = torch.ones(
            batch_spec.batch_size, dtype=torch.int32, device=DEVICE
        )
    return builder.build(common_prefix_len=0, common_attn_metadata=common, **kwargs)


@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_gdn_build_classification(test_case: GDNBuildTestCase):
    """Test that GDN metadata builder classifies requests correctly."""
    builder = _create_gdn_builder(test_case.num_speculative_tokens)
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    meta = _build(builder, batch, test_case.num_decode_draft_tokens)

    assert meta.num_decodes == test_case.expected_num_decodes
    assert meta.num_prefills == test_case.expected_num_prefills
    assert meta.num_prefill_tokens == test_case.expected_num_prefill_tokens
    assert meta.num_spec_decodes == test_case.expected_num_spec_decodes


def test_has_initial_state_after_reclassification():
    """After reclassification, num_prefills > 0 so the prefill kernel path
    should compute has_initial_state. For the reclassified request with
    context_lens > 0, the corresponding entry must be True."""
    builder = _create_gdn_builder(num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[65, 20], query_lens=[1, 3])
    meta = _build(builder, batch, num_decode_draft_tokens=[-1, 2])

    assert meta.num_prefills > 0, "reclassification should produce prefills"
    assert meta.has_initial_state is not None
    # req0 has context_lens = 65 - 1 = 64 > 0, so has_initial_state[0] = True
    assert meta.has_initial_state[0].item() is True


def test_full_cudagraph_spec_metadata_uses_request_count():
    """FULL cudagraph token padding must not pad request-indexed metadata."""
    num_speculative_tokens = 3
    builder = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        full_cuda_graph=True,
    )
    batch = BatchSpec(seq_lens=[80, 96], query_lens=[4, 4])
    meta = _build(builder, batch, num_decode_draft_tokens=[3, 3])

    assert meta.num_spec_decodes == batch.batch_size
    assert meta.num_spec_decode_tokens == batch.compute_num_tokens()
    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.shape == (
        batch.batch_size,
        num_speculative_tokens + 1,
    )
    assert meta.spec_sequence_masks is not None
    assert meta.spec_sequence_masks.shape == (batch.batch_size,)
    assert meta.spec_query_start_loc is not None
    assert meta.spec_query_start_loc.shape == (batch.batch_size + 1,)
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.shape == (batch.batch_size,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("num_speculative_tokens", [1, 2])
def test_full_cudagraph_pure_spec_staging_contents(num_speculative_tokens: int):
    """A pure spec-decode batch (real requests first, cudagraph padding last)
    is staged into the persistent buffers by one kernel; the staged contents
    must equal the reference per-tensor slice/pad construction."""
    device = torch.device("cuda")
    builder = _create_gdn_builder(num_speculative_tokens, full_cuda_graph=True)
    builder = GDNAttentionMetadataBuilder(
        builder.kv_cache_spec, builder.layer_names, builder.vllm_config, device
    )
    k1 = num_speculative_tokens + 1
    # 3 real spec-decode requests (the last one with a shorter query) plus one
    # zero-length padding request.
    query_lens = [k1, k1, k1 - 1, 0]
    seq_lens = [80, 96, 40, 1]
    num_spec_decodes = 3
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens), BLOCK_SIZE, device
    )
    num_accepted = torch.tensor([1, k1, 2, 1], dtype=torch.int32, device=device)
    drafts = torch.tensor([num_speculative_tokens] * 3 + [-1], dtype=torch.int32)
    meta = builder.build(
        0, common, num_accepted_tokens=num_accepted, num_decode_draft_tokens_cpu=drafts
    )
    torch.cuda.synchronize()
    batch_size = common.num_reqs
    num_tokens = sum(query_lens)

    assert meta.num_spec_decodes == num_spec_decodes
    assert meta.num_prefills == 0 and meta.num_decodes == 0
    assert meta.spec_sequence_masks is not None
    assert meta.spec_sequence_masks.tolist() == [True] * 3 + [False]
    assert meta.spec_token_indx is not None
    assert meta.spec_token_indx.tolist() == list(range(num_tokens))
    assert meta.non_spec_token_indx is not None
    assert meta.non_spec_token_indx.numel() == 0
    assert meta.spec_query_start_loc is not None
    qsl = common.query_start_loc.tolist()
    assert meta.spec_query_start_loc.tolist() == qsl[: num_spec_decodes + 1] + [
        qsl[num_spec_decodes]
    ] * (batch_size - num_spec_decodes)
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.tolist() == [1, k1, 2] + [1]
    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.shape == (batch_size, k1)
    expected_states = mamba_get_block_table_tensor(
        common.block_table_tensor,
        common.seq_lens,
        builder.kv_cache_spec,
        builder.vllm_config.cache_config.mamba_cache_mode,
    )[:num_spec_decodes, :k1].expand(num_spec_decodes, k1)
    assert torch.equal(
        meta.spec_state_indices_tensor[:num_spec_decodes], expected_states
    )
    assert (meta.spec_state_indices_tensor[num_spec_decodes:] == NULL_BLOCK_ID).all()
