# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FLASHINFER_MLA_SPARSE's FlashMLA prefill path (bf16 NoPE-512, SM10x) must
match its trtllm-gen path on the same paged cache and top-k indices, and the
padded query heads must not leak into the real heads' output."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    _FLASHMLA_PREFILL_HEADS,
    FlashInferMLASparseImpl,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="FlashMLA sparse prefill with d_qk=512 needs SM10x",
)

HEAD_DIM = 512
BLOCK_SIZE = 64


@pytest.mark.parametrize("num_heads", [8, 32, 64])
@torch.inference_mode()
def test_flashmla_prefill_matches_trtllm(num_heads: int):
    torch.manual_seed(0)
    num_tokens, topk, num_blocks = 1536, 2048, 128
    # Other layers' pages sit between this cache's blocks (block stride of
    # 2 * BLOCK_SIZE rows), as in hybrid-model KV layouts.
    storage = torch.randn(
        num_blocks, 2 * BLOCK_SIZE, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    kv_cache = storage[:, :BLOCK_SIZE]
    rows = torch.arange(num_blocks, device="cuda")[:, None] * 2 * BLOCK_SIZE
    valid_rows = (rows + torch.arange(BLOCK_SIZE, device="cuda")).flatten()
    pick = torch.rand(num_tokens, valid_rows.numel(), device="cuda").argsort(-1)
    indices = valid_rows[pick[:, :topk]].to(torch.int32)
    # Ragged per-token lengths with -1 padding past each length.
    seq_lens = torch.randint(1, topk + 1, (num_tokens,), device="cuda")
    seq_lens[:4] = topk
    seq_lens = seq_lens.to(torch.int32)
    pos = torch.arange(topk, device="cuda")
    indices = torch.where(pos < seq_lens[:, None], indices, -1)

    # Head-padded query buffer as the MLA layer builds it; the padded heads
    # hold NaN to prove they cannot reach the real heads.
    buf = torch.full(
        (num_tokens, _FLASHMLA_PREFILL_HEADS, HEAD_DIM),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    q = buf[:, :num_heads]
    q.copy_(torch.randn_like(q))

    scale = HEAD_DIM**-0.5
    # A real impl object without running __init__ (which needs a vLLM config);
    # only the attributes forward_mqa reads are set.
    impl = FlashInferMLASparseImpl.__new__(FlashInferMLASparseImpl)
    impl.__dict__.update(
        sparse_prefill_q_pad_heads=_FLASHMLA_PREFILL_HEADS,
        sparse_prefill_min_tokens=1024,
        need_to_return_lse_for_decode=False,
        scale=scale,
        bmm1_scale=scale,
        bmm2_scale=1.0,
        is_nope_mla=True,
        index_group=None,
        dcp_world_size=1,
        kv_lora_rank=HEAD_DIM,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        _topk_indices_buffer=indices,
        _workspace_buffer=torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        ),
        # The logical -> physical top-k conversion is covered elsewhere.
        _convert_logical_to_physical_topk=lambda topk, *a, **k: (topk, seq_lens),
    )
    q_padded = impl._head_padded_query(q)
    assert q_padded is not None and q_padded.shape[1] == _FLASHMLA_PREFILL_HEADS
    # A dense query of real heads only is not a padded buffer.
    if num_heads < _FLASHMLA_PREFILL_HEADS:
        assert impl._head_padded_query(q.clone()) is None
    # Below the token threshold the trtllm-gen path is kept.
    assert impl._head_padded_query(q[:1000]) is None

    q_pe = q.new_empty(num_tokens, num_heads, 0)
    prefill = SimpleNamespace(block_size=BLOCK_SIZE, num_decode_tokens=0)
    out, lse = impl.forward_mqa((q, q_pe), kv_cache, prefill, layer=None)
    assert lse is None and out.shape == (num_tokens, num_heads, HEAD_DIM)
    assert out.data_ptr() != q.data_ptr() and not out.isnan().any()

    # Same query as a mixed batch (one decode token) takes the trtllm-gen path.
    mixed = SimpleNamespace(block_size=BLOCK_SIZE, num_decode_tokens=1)
    ref, _ = impl.forward_mqa((q, q_pe), kv_cache, mixed, layer=None)
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-2)
