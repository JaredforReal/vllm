# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode executor over the stock GLM-5.3-Flash backbone: one
routing decision per step, then every layer's stages in one loop."""

from typing import Any, NamedTuple

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available

# Token counts (requests x tokens per request) with a planned decode path.
MAX_TOKENS = 64


class Step(NamedTuple):
    q: int  # tokens per request
    positions: torch.Tensor


def is_decode(md, q: int, b: int) -> bool:
    """Whether one layer's metadata is a pure decode (q = 1) or a uniform verify of
    b tokens. FULL graphs pad whole requests, so every row verifies q tokens."""
    if md.num_prefills:
        return False
    if q == 1:
        return not getattr(md, "num_spec_decodes", 0)
    if not hasattr(md, "num_spec_decodes"):
        return md.num_decode_tokens == md.num_decodes * q == b
    rows, tokens = md.spec_state_indices_tensor, md.num_spec_decode_tokens
    full = rows is not None and rows.shape[0] * q == b
    return full and not md.num_decodes and tokens == md.num_spec_decodes * q


class SpecializedDecode:
    """Built once the weights are loaded. Reads the backbone, never modifies it."""

    def __init__(self, backbone, config):
        self.backbone = backbone
        self.config = config
        self.q = config.q
        self.layers = list(backbone.layers)
        self.batches = frozenset(
            b for b in config.capture_sizes if b <= MAX_TOKENS and b % self.q == 0
        )

    def prepare_kv(self) -> None:
        """Rebinds the layer stages to the KV cache tensors the runner allocated."""

    def plan(self, positions, intermediate_tensors) -> Step | None:
        """This step's decode plan, or None to run the stock forward instead.

        Steps that run the stock forward:
        - pipeline-parallel stage inputs (``intermediate_tensors``);
        - token counts without a planned path (above 64 or not a multiple of q);
        - steps outside a CUDA graph capture (profiling, warmup and the warmup run
          before each capture), so every stock kernel runs before its capture;
        - PIECEWISE steps, keyed by token count alone and so captured as stock;
        - steps with prompt rows (prefills, chunked prefills, fresh prompts);
        - speculative steps whose requests do not all verify q tokens;
        - models that collect aux hidden states (EAGLE-3 / DFlash drafts).

        Every predicate is the same on all TP ranks, so all ranks take one path.
        """
        b = positions.shape[0]
        if intermediate_tensors is not None or b not in self.batches:
            return None
        if self.backbone.aux_hidden_state_layers:
            return None
        if not torch.cuda.is_current_stream_capturing():
            return None
        if not is_forward_context_available():
            return None
        context = get_forward_context()
        if context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return None
        by_layer = context.attn_metadata
        if not isinstance(by_layer, dict) or not by_layer:
            return None
        rows: list[Any] = list(by_layer.values())
        q = self.q if any(getattr(md, "num_spec_decodes", 0) for md in rows) else 1
        if b % q or not all(is_decode(md, q, b) for md in rows):
            return None
        return Step(q, positions)

    def run(self, step: Step, input_ids, inputs_embeds) -> torch.Tensor:
        """Normalized hidden states of a planned step."""
        backbone = self.backbone
        x = inputs_embeds
        if x is None:
            x = backbone.embed_input_ids(input_ids)
        residual = post = comb = None
        for layer in self.layers:
            x, residual, post, comb = layer(step.positions, x, residual, post, comb)
        return backbone.norm(x)
