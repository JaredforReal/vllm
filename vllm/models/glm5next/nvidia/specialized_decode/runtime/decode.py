# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode executor over the stock GLM-5.3-Flash backbone: one
routing decision per step, then every layer's stages in one loop."""

from typing import Any, NamedTuple

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.mhc import hc_contract, hc_expand

from ..linear import planned_step

logger = init_logger(__name__)

# Token counts (requests x tokens per request) with a planned decode path.
MAX_TOKENS = 64
# BF16 projections of the DSA layer and its indexer that dsa_gemm routes.
DSA_PROJECTIONS = ("fused_qkv_a_proj", "q_b_proj", "o_proj")
INDEXER_PROJECTIONS = ("wq_b", "wk_weights_proj")


class Step(NamedTuple):
    q: int  # tokens per request
    positions: torch.Tensor
    metadata: dict[str, Any]  # by layer name


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
        self.batches = sorted(
            b for b in config.capture_sizes if b <= MAX_TOKENS and b % self.q == 0
        )
        self.kda: dict[int, Any] = {}
        if "kda" in config.stages:
            from ..kda import KdaStage

            for layer in self.layers:
                if layer.layer_kind == "kda":
                    stage = KdaStage(
                        layer.self_attn,
                        self.batches,
                        self.q,
                        gemm="gemm" in config.stages,
                    )
                    self.kda[layer.layer_idx] = stage
        self.routed: list[str] = []
        if "dsa_gemm" in config.stages:
            from ..linear import route

            for layer in self.layers:
                if layer.layer_kind != "mla":
                    continue
                attn = layer.self_attn
                modules = [getattr(attn, n, None) for n in DSA_PROJECTIONS]
                indexer = getattr(attn, "indexer", None)
                modules += [getattr(indexer, n, None) for n in INDEXER_PROJECTIONS]
                for m in modules:
                    if m is not None and route(m, self.batches):
                        self.routed.append(m.prefix)
            logger.info("Specialized decode routes %d DSA GEMMs", len(self.routed))

    def prepare_kv(self) -> None:
        """Compiles the layer stages for the KV cache tensors the runner bound."""
        for index, stage in self.kda.items():
            stage.prepare_caches(self.layers[index].self_attn.kv_cache)

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
        b = positions.shape[-1]  # M-RoPE positions are [3, tokens]
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
        logger.info_once("GLM-5.3-Flash specialized decode captures %d-token steps", b)
        return Step(q, positions, by_layer)

    def attention(self, layer, x, step: Step):
        stage = self.kda.get(layer.layer_idx)
        if stage is not None:
            md = step.metadata[layer.self_attn.prefix]
            return stage.run(x, md, step.q)
        return layer.self_attn(hidden_states=x, positions=step.positions)

    def layer(self, layer, x, residual, post, comb, step: Step):
        """``Glm5NextDecoderLayer.forward`` for a decode step (mHC, no SP)."""
        if post is None:
            if layer.layer_idx == 0:
                x = hc_expand(x, layer.n)
            residual = x
            post, comb, x = layer.hc_pre(
                x,
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                norm_weight=layer.input_layernorm.weight.data,
                norm_eps=layer.input_layernorm.variance_epsilon,
            )
        else:
            residual, post, comb, x = layer.hc_fused_post_pre(
                x,
                residual,
                post,
                comb,
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                norm_weight=layer.input_layernorm.weight.data,
                norm_eps=layer.input_layernorm.variance_epsilon,
            )
        x = self.attention(layer, x, step)
        residual, post, comb, x = layer.hc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            norm_weight=layer.post_attention_layernorm.weight.data,
            norm_eps=layer.post_attention_layernorm.variance_epsilon,
        )
        x = layer.mlp(x)
        if layer.layer_idx == layer.num_hidden_layers - 1:
            x = layer.hc_post(x, residual, post, comb)
            return hc_contract(x, layer.n), None, None, None
        return x, residual, post, comb

    def run(self, step: Step, input_ids, inputs_embeds) -> torch.Tensor:
        """Normalized hidden states of a planned step."""
        backbone = self.backbone
        x = inputs_embeds
        if x is None:
            x = backbone.embed_input_ids(input_ids)
        residual = post = comb = None
        with planned_step():
            for layer in self.layers:
                x, residual, post, comb = self.layer(
                    layer, x, residual, post, comb, step
                )
        return backbone.norm(x)
