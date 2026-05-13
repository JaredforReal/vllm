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
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig

logger = init_logger(__name__)

KDA_SAFE_GATE_LOWER_BOUND = -5.0
KDA_NEG_EIGVAL_BETA_SCALE = 2.0
KDA_DEFAULT_BETA_SCALE = 1.0

# --- Temporary debug helpers ---
import os

import torch.distributed as _dist

_DEBUG_RANK = int(os.environ.get("GLM5_DEBUG_RANK", "0"))


def _is_debug_rank() -> bool:
    if not _dist.is_initialized():
        return True
    return _dist.get_rank() == _DEBUG_RANK


def _debug_print(msg: str):
    print(f"[GLM5-DEBUG] {msg}", flush=True)


# --- End debug helpers ---


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
        mla = self.mla_attn  # MultiHeadLatentAttentionWrapper

        if self.q_lora_rank is not None:
            qkv_lora = mla.fused_qkv_a_proj(hidden_states)[0]
            if _is_debug_rank():
                _debug_print(
                    f"    MLA fused_qkv_a_proj: norm={qkv_lora.float().norm():.4f}, "
                    f"shape={qkv_lora.shape}, has_inf={qkv_lora.isinf().any()}"
                )
            q_c, kv_lora = qkv_lora.split(
                [mla.q_lora_rank, mla.kv_lora_rank + mla.qk_rope_head_dim], dim=-1
            )
            q_c = mla.q_a_layernorm(q_c)
            q = mla.q_b_proj(q_c)[0]
        else:
            kv_lora = mla.kv_a_proj_with_mqa(hidden_states)[0]
            q = mla.q_proj(hidden_states)[0]

        kv_c, k_pe = kv_lora.split([mla.kv_lora_rank, mla.qk_rope_head_dim], dim=-1)
        kv_c_normed = mla.kv_a_layernorm(kv_c)

        if _is_debug_rank():
            _debug_print(
                f"    MLA q: norm={q.float().norm():.4f}, has_inf={q.isinf().any()}"
            )
            _debug_print(
                f"    MLA kv_c: norm={kv_c.float().norm():.4f}, has_inf={kv_c.isinf().any()}"
            )
            _debug_print(
                f"    MLA kv_c_normed: norm={kv_c_normed.float().norm():.4f}, has_inf={kv_c_normed.isinf().any()}"
            )
            _debug_print(
                f"    MLA k_pe: norm={k_pe.float().norm():.4f}, has_inf={k_pe.isinf().any()}"
            )

        q = q.view(-1, mla.num_heads, mla.qk_head_dim)
        k_pe = k_pe.unsqueeze(1)

        if mla.rotary_emb is not None:
            q[..., mla.qk_nope_head_dim :], k_pe = mla.rotary_emb(
                positions, q[..., mla.qk_nope_head_dim :], k_pe
            )

        attn_out = mla.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], mla.num_heads * mla.v_head_dim),
        )

        if _is_debug_rank():
            _debug_print(
                f"    MLA attn_out: norm={attn_out.float().norm():.4f}, "
                f"has_inf={attn_out.isinf().any()}, has_nan={attn_out.isnan().any()}"
            )

        result = mla.o_proj(attn_out)[0]
        if _is_debug_rank():
            _debug_print(
                f"    MLA o_proj: norm={result.float().norm():.4f}, "
                f"has_inf={result.isinf().any()}"
            )

        output[:] = result


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

        self.A_log.weight_loader = a_log_weight_loader

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        num_tokens = hidden_states.size(0)
        _dbg = _is_debug_rank()

        q = self.q_proj(hidden_states)[0]
        k = self.k_proj(hidden_states)[0]
        v = self.v_proj(hidden_states)[0]

        if _dbg:
            _debug_print(
                f"    KDA q_proj: norm={q.float().norm():.4f}, inf={q.isinf().any()}"
            )
            _debug_print(
                f"    KDA k_proj: norm={k.float().norm():.4f}, inf={k.isinf().any()}"
            )
            _debug_print(
                f"    KDA v_proj: norm={v.float().norm():.4f}, inf={v.isinf().any()}"
            )

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

        if _dbg:
            _debug_print(
                f"    KDA g1: norm={g1.float().norm():.4f}, inf={g1.isinf().any()}"
            )
            _debug_print(
                f"    KDA beta: norm={beta.float().norm():.4f}, inf={beta.isinf().any()}"
            )

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

        if _dbg:
            _debug_print(
                f"    KDA core_attn_out: norm={core_attn_out.float().norm():.4f}, "
                f"inf={core_attn_out.isinf().any()}, nan={core_attn_out.isnan().any()}"
            )

        core_attn_out = self.o_norm(core_attn_out, g2)

        if _dbg:
            _debug_print(
                f"    KDA o_norm: norm={core_attn_out.float().norm():.4f}, "
                f"inf={core_attn_out.isinf().any()}, nan={core_attn_out.isnan().any()}"
            )

        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        output[:] = self.o_proj(core_attn_out)[0]

        if _dbg:
            _debug_print(
                f"    KDA o_proj: norm={output.float().norm():.4f}, inf={output.isinf().any()}"
            )


class Glm5NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
        layer_idx: int,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_moe = config.is_moe
        self.num_hidden_layers = config.num_hidden_layers
        self.rms_norm_eps = config.rms_norm_eps
        self.num_experts = config.n_routed_experts

        # mhc config
        self.mhc_num_residual_streams = config.mhc_num_residual_streams
        self.mhc_no_norm_weight = config.mhc_no_norm_weight
        self.mhc_tau = config.mhc_tau
        self.hc_eps = config.hc_eps
        self.mhc_sinkhorn_iterations = config.mhc_sinkhorn_iterations
        self.mhc_post_mult_value = config.mhc_post_mult_value

        if config.is_kda_layer(layer_idx):
            self.self_attn = Glm5NextLinearAttention(
                layer_idx=layer_idx,
                hidden_size=config.hidden_size,
                quant_config=None,  # KDA projections are BF16 in checkpoint
                cache_config=cache_config,
                model_config=config,
                rms_norm_eps=config.rms_norm_eps,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = Glm5NextMLAAttention(
                vllm_config=vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank
                if hasattr(config, "q_lora_rank")
                else None,
                kv_lora_rank=config.kv_lora_rank
                if hasattr(config, "kv_lora_rank")
                else None,
                max_position_embeddings=config.max_position_embeddings,
                cache_config=cache_config,
                quant_config=None,  # MLA projections are BF16 in checkpoint
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
                skip_rope=getattr(config, "mla_nope", False),
            )

        if (
            self.is_moe
            and self.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
        ):
            self.mlp = Glm5NextMoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Glm5NextMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                swiglu_limit=config.swiglu_limit,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        n = config.mhc_num_residual_streams
        d_model = n * self.hidden_size
        mix_hc = (2 + n) * n

        self.n = n

        # attn hc
        self.hc_attn_fn = nn.Parameter(
            torch.empty(mix_hc, d_model, dtype=torch.float32)
        )
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

        # ffn hc
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, d_model, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def forward(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        _dbg = _is_debug_rank() and self.layer_idx in (
            0,
            1,
            2,
            10,
            20,
            30,
            self.num_hidden_layers - 1,
        )

        # mHC start
        if self.layer_idx == 0:
            if _dbg:
                _debug_print(
                    f"  L{self.layer_idx} pre hc_expand: norm={x.float().norm():.4f}, shape={x.shape}"
                )
            x = hc_expand(x, self.n)
            if _dbg:
                _debug_print(
                    f"  L{self.layer_idx} post hc_expand: norm={x.float().norm():.4f}, shape={x.shape}"
                )

        # Self Attention
        residual = x
        post, comb, x = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        if _dbg:
            _debug_print(
                f"  L{self.layer_idx} post hc_pre(attn): norm={x.float().norm():.4f}, shape={x.shape}"
            )
        x = self.input_layernorm(x)

        attn_output = torch.empty_like(x)
        self.self_attn(
            hidden_states=x,
            positions=positions,
            output=attn_output,
        )
        x = attn_output
        if _dbg:
            _debug_print(
                f"  L{self.layer_idx} post attn: norm={x.float().norm():.4f}, shape={x.shape}"
            )

        x = self.hc_post(x, residual, post, comb)

        residual = x
        post, comb, x = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )

        # Fully Connected
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        if _dbg:
            _debug_print(
                f"  L{self.layer_idx} post mlp: norm={x.float().norm():.4f}, shape={x.shape}"
            )

        x = self.hc_post(x, residual, post, comb)

        # mHC end
        if self.layer_idx == self.num_hidden_layers - 1:
            if _dbg:
                _debug_print(
                    f"  L{self.layer_idx} pre hc_contract: norm={x.float().norm():.4f}, shape={x.shape}"
                )
            x = hc_contract(x, self.n)
            if _dbg:
                _debug_print(
                    f"  L{self.layer_idx} post hc_contract: norm={x.float().norm():.4f}, shape={x.shape}"
                )

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
            hc_post_mult_value=self.mhc_post_mult_value,
            sinkhorn_repeat=self.mhc_sinkhorn_iterations,
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
        self.config = config

        self.vocab_size = config.vocab_size
        self.device = current_platform.device_type

        """
        if config.index_topk is not None:
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=self.device,
            )
        else:
        """
        topk_indices_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return Glm5NextDecoderLayer(
                vllm_config=vllm_config,
                config=config,
                layer_idx=layer_idx,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
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

        _debug = _is_debug_rank()
        if _debug:
            _debug_print(
                f"[embed] shape={hidden_states.shape}, "
                f"norm={hidden_states.float().norm():.4f}, "
                f"nan={hidden_states.isnan().any()}"
            )

        for i, layer in enumerate(self.layers[self.start_layer : self.end_layer]):
            hidden_states = layer(
                positions=positions,
                x=hidden_states,
            )
            if _debug:
                idx = self.start_layer + i
                _debug_print(
                    f"[layer {idx:2d}] shape={hidden_states.shape}, "
                    f"norm={hidden_states.float().norm():.4f}, "
                    f"nan={hidden_states.isnan().any()}"
                )

        if not get_pp_group().is_last_rank:
            # PP: intermediate tensor may be 3D [T, n, H] (after hc_expand)
            # or 2D [T, H] (before hc_expand). Layers handle both correctly
            # since hc_expand only runs at layer 0 (first PP rank) and
            # hc_contract at the last layer (last PP rank).
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            # MLA: fuse q_a_proj and kv_a_proj_with_mqa
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
        ]
        if self.config.is_moe:
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.n_routed_experts,
            )
        else:
            expert_params_mapping = []
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # GLM5-Next NoPE: checkpoint's kv_a_proj_with_mqa has only kv_lora_rank
        # rows, but the model expects kv_lora_rank + qk_rope_head_dim rows.
        # Pad the missing rope portion with zeros.
        kv_a_pad_size = 0
        if self.config.mla_nope and self.config.qk_rope_head_dim > 0:
            kv_a_pad_size = self.config.qk_rope_head_dim

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

            # Pad kv_a_proj_with_mqa for NoPE models
            if kv_a_pad_size > 0 and ".kv_a_proj_with_mqa." in name:
                pad = torch.zeros(
                    kv_a_pad_size,
                    *loaded_weight.shape[1:],
                    dtype=loaded_weight.dtype,
                    device=loaded_weight.device,
                )
                loaded_weight = torch.cat([loaded_weight, pad], dim=0)

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
                name_mapped = name.replace(weight_name, param_name)
                # QKV fusion: skip if fused module doesn't exist in model
                if param_name == ".fused_qkv_a_proj" and name_mapped not in params_dict:
                    continue
                name = name_mapped
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
            if weight_name.startswith(f"layers.{layer_idx + i}."):
                return layer_idx + i
    return None
