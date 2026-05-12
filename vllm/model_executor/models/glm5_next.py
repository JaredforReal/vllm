# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
from einops import rearrange
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fla.ops.kda import fused_kda_gate
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.kda import KimiDeltaAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mhc import hc_contract, hc_expand
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
    sharded_weight_loader,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLP as Glm5NextMLP
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE as Glm5NextMoE
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig

logger = init_logger(__name__)

KDA_SAFE_GATE_LOWER_BOUND = -5.0
KDA_NEG_EIGVAL_BETA_SCALE = 2.0
KDA_DEFAULT_BETA_SCALE = 1.0


def naive_kda_lowerbound_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float = KDA_SAFE_GATE_LOWER_BOUND,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    g = g.float()
    g = g + dt_bias.view(1, -1)
    g = g.view(g.shape[0], A_log.numel(), -1)
    g = lower_bound * torch.nn.functional.sigmoid(A_log.view(-1, 1).exp() * g)
    return g.to(output_dtype)


class Glm5NextMLAAttention(DeepseekV2MLAAttention):
    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, output: torch.Tensor
    ) -> None:
        output[:] = super().forward(positions, hidden_states, None)


class Glm5NextLinearAttention(KimiDeltaAttention):
    """GLM5-Next variant of KDA with safe_gate and allow_neg_eigval support"""

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        model_config: ModelConfig | None = None,
        rms_norm_eps: float = 1e-5,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__(
            layer_idx,
            hidden_size,
            quant_config,
            cache_config,
            model_config,
            rms_norm_eps,
            prefix,
        )
        self.allow_neg_eigval = self.model_config.linear_allow_neg_eigval
        self.safe_gate = self.model_config.linear_attn_config.get("safe_gate", False)

        def a_log_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor):
            if loaded_weight.dim() == 1:
                loaded_weight = loaded_weight.view([1, 1, -1, 1])
            return sharded_weight_loader(2)(param, loaded_weight)

        set_weight_attrs(self.A_log, {"weight_loader": a_log_weight_loader})

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        num_tokens = hidden_states.size(0)
        q = self.q_proj(hidden_states)[0]
        k = self.k_proj(hidden_states)[0]
        v = self.v_proj(hidden_states)[0]

        beta = self.b_proj(hidden_states)[0].float().sigmoid()
        if self.allow_neg_eigval:
            beta = beta * KDA_NEG_EIGVAL_BETA_SCALE
        beta = beta.unsqueeze(0)

        g1 = self.f_b_proj(self.f_a_proj(hidden_states)[0])[0]
        if self.safe_gate:
            g1 = naive_kda_lowerbound_gate(g1, self.A_log, self.dt_bias)
        else:
            g1 = fused_kda_gate(g1, self.A_log, self.head_dim, g_bias=self.dt_bias)
        g1 = g1.unsqueeze(0)

        g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]
        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)

        core_attn_out = torch.zeros(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.kda_attention(
            q,
            k,
            v,
            g1,
            beta,
            core_attn_out,
            self.prefix,
        )
        core_attn_out = self.o_norm(core_attn_out, g2)
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        output[:] = self.o_proj(core_attn_out)[0]


class Glm5NextDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Glm5NextConfig,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        model_config: ModelConfig | None = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_moe = config.is_moe

        if config.is_kda_layer(layer_idx):
            self.self_attn = Glm5NextLinearAttention(
                layer_idx=layer_idx,
                hidden_size=config.hidden_size,
                quant_config=quant_config,
                cache_config=cache_config,
                model_config=config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = Glm5NextMLAAttention(
                layer_idx=layer_idx,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                quant_config=quant_config,
                cache_config=cache_config,
                model_config=model_config,
                prefix=f"{prefix}.self_attn",
                config=config,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                use_nope=config.mla_use_nope,
            )

        if (
            self.is_moe
            and config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.block_sparse_moe = Glm5NextMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe",
            )
            self.mlp = self.block_sparse_moe
        else:
            self.mlp = Glm5NextMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # config.mhc = True
        n = config.mhc_num_residual_streams
        d_model = n * self.hidden_size
        mix_hc = (2 + n) * n

        # attn hc
        self.attn_hc_mapping_proj = nn.Linear(
            d_model, mix_hc, bias=False, dtype=torch.float32
        )
        self.attn_hc_bias = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.attn_hc_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        if not config.mhc_no_norm_weight:
            self.attn_hc_norm = nn.RMSNorm(d_model, eps=config.rms_norm_eps)
        else:
            self.attn_hc_norm = None

        # ffn hc
        self.ffn_hc_mapping_proj = nn.Linear(
            d_model, mix_hc, bias=False, dtype=torch.float32
        )
        self.ffn_hc_bias = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.ffn_hc_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        if not config.mhc_no_norm_weight:
            self.ffn_hc_norm = nn.RMSNorm(d_model, eps=config.rms_norm_eps)
        else:
            self.ffn_hc_norm = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        x: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # mHC start
        if self.layer_idx == 0:
            x = hc_expand(x, self.n)

        # Self Attention
        residual = x
        post, comb, x = self.hc_pre(
            x, self.attn_hc_mapping_proj.weight, self.attn_hc_scale, self.attn_hc_bias
        )
        x = self.input_layernorm(x)

        attn_output = torch.empty_like(x)
        self.self_attn(
            hidden_states=x,
            positions=positions,
            output=attn_output,
        )
        x = attn_output

        x = self.hc_post(x, residual, post, comb)

        residual = x
        post, comb, x = self.hc_pre(
            x, self.attn_hc_mapping_proj.weight, self.attn_hc_scale, self.attn_hc_bias
        )

        # Fully Connected
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)

        x = self.hc_post(x, residual, post, comb)

        # mHC end
        if self.layer_idx == self.config.num_hidden_layers - 1:
            x = hc_contract(x, self.n)

        return x

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        post_mix, res_mix, layer_input = torch.ops.vllm.mhc_pre(
            residual=x,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.hc_post_alpha,
            sinkhorn_repeat=self.hc_sinkhorn_iters,
        )
        return post_mix, res_mix, layer_input

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        return torch.ops.vllm.mhc_post(x, residual, post, comb)


@support_torch_compile
class Glm5NextModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        self.config = config

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        extra_kwargs = {}

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return Glm5NextDecoderLayer(
                config,
                layer_idx,
                cache_config,
                quant_config,
                parallel_config,
                model_config,
                prefix,
                **extra_kwargs,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, (
            "num_attention_heads must be divisible by world_size"
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for _, layer in enumerate(self.layers[self.start_layer : self.end_layer]):
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states, _ = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        if self.config.is_moe:
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.num_experts,
            )
        else:
            expert_params_mapping = []
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for args in weights:
            name, loaded_weight = args[:2]
            kwargs = args[2] if len(args) > 2 else {}
            if "rotary_emb.inv_freq" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for idx, (param_name, weight_name, expert_id, shard_id) in enumerate(
                    expert_params_mapping
                ):
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=shard_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias")
                        and name not in params_dict
                        and not self.config.is_linear_attn
                    ):  # noqa: E501
                        continue
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)
        return loaded_params


class Glm5NextForCausalLM(
    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        self.model = Glm5NextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, scale=logit_scale
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype, torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            tp_size,
            hf_config.linear_attn_config["num_heads"],
            hf_config.linear_attn_config["head_dim"],
            conv_kernel_size=hf_config.linear_attn_config["short_conv_kernel_size"],
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[
        MambaStateCopyFunc, MambaStateCopyFunc, MambaStateCopyFunc, MambaStateCopyFunc
    ]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


def get_spec_layer_idx_from_weight_name(
    config: Glm5NextConfig, weight_name: str
) -> int | None:
    if hasattr(config, "num_nextn_predict_layers") and (
        config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx + i}."):
                return layer_idx + i
    return None
