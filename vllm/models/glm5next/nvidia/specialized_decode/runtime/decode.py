# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode executor over the stock GLM-5.3-Flash backbone: one
routing decision per step, then every layer's stages in one loop."""

from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from vllm.config import CUDAGraphMode
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_output import MoEOutput
from vllm.model_executor.layers.fused_moe.runner.moe_runner import _unpack
from vllm.model_executor.layers.mhc import hc_contract, hc_expand
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata

from ..linear import planned_step

if TYPE_CHECKING:
    from ..all_reduce_mhc import AllReduceMhcBoundary

logger = init_logger(__name__)

# Token counts (requests x tokens per request) with a planned decode path.
MAX_TOKENS = 64
# Token counts the fused all-reduce + mHC boundary serves.
AR_MHC_MAX_TOKENS = 32
# Token counts the mega MoE kernel serves (it streams each route's experts).
MEGA_MAX_TOKENS = 2
# BF16 projections of the DSA layer and its indexer that dsa_gemm routes.
DSA_PROJECTIONS = ("fused_qkv_a_proj", "q_b_proj", "o_proj")
INDEXER_PROJECTIONS = ("wq_b", "wk_weights_proj")


def _override(obj, name: str, value, enabled: bool):
    """Sets ``obj.name`` to ``value`` for one call if ``enabled``."""
    if not enabled:
        return nullcontext()
    return _swap(obj, name, value)


@contextmanager
def _swap(obj, name: str, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


class Step(NamedTuple):
    q: int  # tokens per request
    positions: torch.Tensor
    metadata: dict[str, Any]  # by layer name


def is_decode(md, q: int, b: int) -> bool:
    """Whether one layer's metadata is a pure decode (q = 1) or a uniform verify of
    b tokens. FULL graphs pad whole requests, so every row verifies q tokens.

    The DSA indexer metadata files q-token verify rows either as decodes or as
    short prefills (the stock indexer runs them that way inside FULL verify
    graphs too); both are uniform verify steps."""
    if q > 1 and isinstance(md, DeepseekV32IndexerMetadata):
        if md.num_prefills:
            rows = md.num_prefills * q
            return not md.num_decodes and rows == md.num_prefill_tokens == b
        return md.num_decode_tokens == md.num_decodes * q == b
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
                    if m is not None and route(m):
                        self.routed.append(m.prefix)
            logger.info("Specialized decode routes %d DSA GEMMs", len(self.routed))
        if "dsa_indexer" in config.stages:
            from ..indexer import IndexerStage

            for layer in self.layers:
                if layer.layer_kind == "mla":
                    IndexerStage(layer.self_attn.indexer)
        # MoE layers whose routed experts run as one mega kernel.
        self.mega: dict[int, Any] = {}
        if "mega_moe" in config.stages:
            self.mega = self._build_mega()
        self.mhc = None
        self.ar_mhc: AllReduceMhcBoundary | None = None
        # MoE layers whose routed top-k reduction the fused boundary takes.
        self.deferred: set[int] = set()
        if "ar_mhc" in config.stages:
            self.ar_mhc = self._build_ar_mhc()
            if self.ar_mhc is not None:
                self.deferred = self._defer_moe_finalize()
        if "mhc" in config.stages:
            from ..mhc import MAX_TOKENS as MHC_MAX_TOKENS
            from ..mhc import MhcBoundary

            first = self.layers[0]
            self.mhc = MhcBoundary(first, first.hidden_size, first.hc_attn_fn.device)
            self.mhc_max_tokens = MHC_MAX_TOKENS
        if self.mhc is not None or self.ar_mhc is not None:
            # hc_fn is BF16 in the checkpoint, so the BF16 copy is exact.
            self.hc_fn = {
                id(fn): fn.detach().bfloat16()
                for layer in self.layers
                for fn in (layer.hc_attn_fn, layer.hc_ffn_fn)
            }

    def _build_ar_mhc(self):
        from ..all_reduce_mhc import AllReduceMhcBoundary

        first = self.layers[0]
        norms = {
            n.variance_epsilon
            for layer in self.layers
            for n in (layer.input_layernorm, layer.post_attention_layernorm)
        }
        if len(norms) != 1:
            logger.warning("Specialized decode: mixed norm eps, no fused boundary")
            return None
        return AllReduceMhcBoundary(
            hidden_size=first.hidden_size,
            max_num_tokens=AR_MHC_MAX_TOKENS,
            top_k=self.backbone.config.num_experts_per_tok,
            rms_eps=first.rms_norm_eps,
            hc_eps=first.hc_eps,
            post_mult=first.mhc_post_mult_value,
            sinkhorn_repeat=first.mhc_sinkhorn_iterations,
            norm_eps=norms.pop(),
            device=first.hc_attn_fn.device,
        )

    def _build_mega(self) -> dict[int, Any]:
        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

        from ..mega_moe import MegaExperts

        tokens = [b for b in self.batches if b <= MEGA_MAX_TOKENS]
        mega: dict[int, Any] = {}
        config = self.backbone.config
        if (
            config.scoring_func != "sigmoid"
            or config.n_group != 1
            or not (config.norm_topk_prob)
        ):
            logger.warning("Specialized decode: unsupported routing, no mega MoE")
            return {}
        for layer in self.layers[:-1]:
            if not layer._mlp_is_moe:
                continue
            routed = layer.mlp.experts.routed_experts
            experts = MegaExperts(
                routed.w13_weight,
                routed.w13_weight_scale_inv,
                routed.w2_weight,
                routed.w2_weight_scale_inv,
                self.backbone.config.num_experts_per_tok,
                self.backbone.config.swiglu_limit,
                layer.mlp.routed_scaling_factor,
                is_deep_gemm_e8m0_used(),
            )
            experts.prepare(tokens, layer.mlp.gate.out_dtype or torch.bfloat16)
            mega[layer.layer_idx] = experts
            device = routed.w13_weight.device
        if mega:
            top_k = self.backbone.config.num_experts_per_tok
            self.route_order = torch.arange(
                max(tokens) * top_k, dtype=torch.int32, device=device
            )
            self.shared_stream = torch.cuda.Stream(device)
        logger.info("Specialized decode runs %d MoE layers as one kernel", len(mega))
        return mega

    def _mega_moe(self, layer, x):
        """Routed experts as one kernel, left unfinalized for the boundary."""
        from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput

        moe = layer.mlp
        main = torch.cuda.current_stream()
        self.shared_stream.wait_stream(main)
        with torch.cuda.stream(self.shared_stream):
            shared = moe.shared_experts(x)
        logits, _ = moe.experts.gate(x)
        rows, ids, weights = self.mega[layer.layer_idx](
            x, logits, moe.gate.e_score_correction_bias
        )
        main.wait_stream(self.shared_stream)
        m, top_k = ids.shape
        routed = UnfinalizedMoEOutput(
            gemm2_permuted=rows,
            expert_weights=weights,
            expanded_idx_to_permuted_idx=self.route_order[: m * top_k].view(m, top_k),
        )
        return MoEOutput(routed=routed, shared_output=shared)

    def _defer_moe_finalize(self) -> set[int]:
        """MoE layers (but the last) whose TRTLLM-Gen FP8 experts can stop
        after GEMM2 for the fused boundary's finalize publish."""
        from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (
            TrtLlmFp8ExpertsModular,
            TrtLlmFp8ExpertsMonolithic,
        )

        deferred = set()
        for layer in self.layers[:-1]:
            if not layer._mlp_is_moe:
                continue
            experts = layer.mlp.experts
            method = experts.routed_experts.quant_method
            if (
                getattr(method, "experts_cls", None)
                not in (TrtLlmFp8ExpertsModular, TrtLlmFp8ExpertsMonolithic)
                or experts.routed_scaling_factor != 1.0
            ):
                logger.info_once(
                    "Specialized decode keeps the MoE finalize: %s, scale %s",
                    str(getattr(method, "experts_cls", None)),
                    str(experts.routed_scaling_factor),
                )
                continue
            config = experts.moe_config
            config.limit_deferred_moe_finalize(AR_MHC_MAX_TOKENS)
            # Planned steps turn the deferral on per call; others finalize.
            config._defer_moe_finalize = True
            usable = config.use_deferred_moe_finalize
            config._defer_moe_finalize = False
            if usable:
                deferred.add(layer.layer_idx)
        logger.info("Specialized decode defers %d MoE finalizes", len(deferred))
        return deferred

    def partial(self, tokens: int) -> bool:
        """Whether sublayers leave their output TP-partial for the fused
        all-reduce + mHC boundary."""
        return self.ar_mhc is not None and self.ar_mhc.supports(tokens)

    def boundary(self, layer, x, residual, post, comb, fn, scale, base, norm):
        """The previous sublayer's mHC post, then the next one's pre and norm."""
        if self.mhc is not None and x.shape[0] <= self.mhc_max_tokens:
            return self.mhc(
                x,
                residual.contiguous(),
                post.contiguous(),
                comb.contiguous(),
                self.hc_fn[id(fn)],
                scale,
                base,
                norm.weight.data,
                norm.variance_epsilon,
            )
        return layer.hc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            norm_weight=norm.weight.data,
            norm_eps=norm.variance_epsilon,
        )

    def reduced_boundary(self, layer, x, residual, post, comb, fn, scale, base, norm):
        """``boundary`` of a TP-partial sublayer output."""
        assert self.ar_mhc is not None
        return self.ar_mhc(
            x,
            residual.contiguous(),
            post.contiguous(),
            comb.contiguous(),
            self.hc_fn[id(fn)],
            scale,
            base,
            norm.weight.data,
        )

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
        if not torch.cuda.is_current_stream_capturing():
            return None
        if intermediate_tensors is not None or b not in self.batches:
            logger.info_once("Specialized decode skips a %d-token capture", b)
            return None
        if self.backbone.aux_hidden_state_layers:
            logger.info_once(
                "Specialized decode skips captures: aux layers %s",
                self.backbone.aux_hidden_state_layers,
            )
            return None
        if not is_forward_context_available():
            return None
        context = get_forward_context()
        if context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            logger.info_once("Specialized decode skips a PIECEWISE %d-token capture", b)
            return None
        by_layer = context.attn_metadata
        if not isinstance(by_layer, dict) or not by_layer:
            return None
        rows: list[Any] = list(by_layer.values())
        q = self.q if any(getattr(md, "num_spec_decodes", 0) for md in rows) else 1
        bad = next((md for md in rows if not is_decode(md, q, b)), None)
        if b % q or bad is not None:
            if bad is not None:
                fields = (
                    "num_prefills",
                    "num_decodes",
                    "num_decode_tokens",
                    "num_spec_decodes",
                    "num_spec_decode_tokens",
                )
                logger.info_once(
                    "Specialized decode skips a %d-token capture (q=%d): %s %s %s",
                    b,
                    q,
                    type(bad).__name__,
                    str({f: getattr(bad, f, None) for f in fields}),
                    str(
                        getattr(
                            getattr(bad, "spec_state_indices_tensor", None),
                            "shape",
                            None,
                        )
                    ),
                )
            return None
        logger.info_once("GLM-5.3-Flash specialized decode captures %d-token steps", b)
        return Step(q, positions, by_layer)

    def attention(self, layer, x, step: Step):
        """The attention output and whether it is still TP-partial."""
        partial = self.partial(x.shape[0])
        stage = self.kda.get(layer.layer_idx)
        if stage is not None:
            md = step.metadata[layer.self_attn.prefix]
            return stage.run(x, md, step.q, reduce=not partial), partial
        o_proj = layer.self_attn.o_proj
        with _override(o_proj, "reduce_results", False, partial):
            return layer.self_attn(hidden_states=x, positions=step.positions), partial

    def mlp(self, layer, x):
        """The FFN output and whether it is still TP-partial."""
        partial = self.partial(x.shape[0])
        if partial and layer.layer_idx in self.mega and x.shape[0] <= MEGA_MAX_TOKENS:
            return self._mega_moe(layer, x), True
        if partial and layer.layer_idx in self.deferred:
            experts = layer.mlp.experts
            with _swap(experts.moe_config, "_defer_moe_finalize", True):
                # The runner's custom op returns tensors only; run its body.
                shared, routed = _unpack(experts._forward_impl(x, x, x, None))
            return MoEOutput(routed=routed, shared_output=shared), True
        if layer._mlp_is_moe:
            config = layer.mlp.experts.moe_config
            unreduced = _override(config, "skip_final_all_reduce", True, partial)
        else:
            unreduced = _override(layer.mlp.down_proj, "reduce_results", False, partial)
        with unreduced:
            return layer.mlp(x), partial

    def layer(self, layer, x, partial, residual, post, comb, step: Step):
        """``Glm5NextDecoderLayer.forward`` for a decode step (mHC, no SP).
        ``partial``: whether ``x`` still needs its TP all-reduce."""
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
            boundary = self.reduced_boundary if partial else self.boundary
            residual, post, comb, x = boundary(
                layer,
                x,
                residual,
                post,
                comb,
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                layer.input_layernorm,
            )
        x, partial = self.attention(layer, x, step)
        boundary = self.reduced_boundary if partial else self.boundary
        residual, post, comb, x = boundary(
            layer,
            x,
            residual,
            post,
            comb,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            layer.post_attention_layernorm,
        )
        x, partial = self.mlp(layer, x)
        if layer.layer_idx == layer.num_hidden_layers - 1:
            if partial:
                x = tensor_model_parallel_all_reduce(x)
            if self.ar_mhc is not None:
                self.ar_mhc.join()
            x = layer.hc_post(x, residual, post, comb)
            return hc_contract(x, layer.n), False, None, None, None
        return x, partial, residual, post, comb

    def run(self, step: Step, input_ids, inputs_embeds) -> torch.Tensor:
        """Normalized hidden states of a planned step."""
        backbone = self.backbone
        x = inputs_embeds
        if x is None:
            x = backbone.embed_input_ids(input_ids)
        state = (False, None, None, None)
        with planned_step():
            for layer in self.layers:
                x, *state = self.layer(layer, x, *state, step)
        return backbone.norm(x)
