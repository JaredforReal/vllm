# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_image/modeling_glm_image.py
# Copyright 2025 The vLLM team.
# Copyright 2025 The ZhipuAI Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only GLM-Image model compatible with HuggingFace weights."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.glm_image.configuration_glm_image import (
    GlmImageTextConfig,
    GlmImageVQVAEConfig,
)

from vllm.attention.layer import Attention
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.conv import Conv2dLayer
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors

from .utils import (
    make_empty_intermediate_tensors_factory,
    make_layers,
)

logger = init_logger(__name__)


# === VQ-VAE Components ===


class GlmImageVQVAEVectorQuantizer(nn.Module):
    """
    Vector Quantizer module for GLM-Image VQ-VAE (Inference-optimized).

    This module quantizes continuous latent vectors into discrete codebook vectors
    using L2-normalized distance computation for better stability.

    Key differences from Chameleon's VQ-VAE:
    - GLM-Image uses L2 normalization on both input and codebook embeddings
    - Distance is computed as cosine similarity in normalized space

    Optimizations for inference (compared to transformers implementation):
    1. Uses matmul + argmax(similarity) instead of einsum + argmin(distance)
       - Mathematically equivalent: argmin(2 - 2*sim) = argmax(sim)
       - More efficient and clearer for L2-normalized vectors
    2. Removes redundant normalization (transformers normalizes twice)
    3. Removes training-only components (loss, straight-through estimator, beta)
    4. Directly returns quantized vectors without gradient preservation

    Args:
        config: GlmImageVQVAEConfig containing:
            - num_embeddings: Number of codebook vectors (typically 16384)
            - embed_dim: Dimension of each embedding vector (typically 2048)

    Mathematical Verification:
        For L2-normalized vectors where ||z|| = ||e|| = 1:
        - distance = ||z - e||^2 = 2 - 2*(z·e) = 2(1 - cosine_similarity)
        - Therefore: argmin(distance) ≡ argmax(cosine_similarity)
        This equivalence has been verified numerically (see verify_vqvae_correctness.py)
    """

    def __init__(self, config: GlmImageVQVAEConfig):
        super().__init__()
        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.embed_dim

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def forward(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize the input hidden states.

        Args:
            hidden_state: Input tensor of shape (batch, channels, height, width)

        Returns:
            Tuple of:
                - hidden_state_quant: Quantized tensor, same shape as input
                - min_encoding_indices: Codebook indices of shape
                  (batch * height * width,)
        """
        batch_size, channels, height, width = hidden_state.shape

        # Permute to (batch, height, width, channels) and flatten for processing
        hidden_state_flat = hidden_state.permute(0, 2, 3, 1).reshape(
            -1, self.embedding_dim
        )

        # L2 normalize both hidden states and embeddings
        # This is the key difference from Chameleon's implementation
        hidden_state_normalized = F.normalize(hidden_state_flat, p=2, dim=-1)
        embedding_normalized = F.normalize(self.embedding.weight, p=2, dim=-1)

        # Compute cosine similarity (since both are L2 normalized)
        # Higher similarity = closer match, so we negate for argmin
        # Using matmul for efficiency: (N, D) @ (D, K) -> (N, K)
        similarity = torch.matmul(hidden_state_normalized, embedding_normalized.t())

        # Find nearest codebook entry (highest similarity)
        min_encoding_indices = torch.argmax(similarity, dim=1)

        # Get quantized vectors using normalized embeddings
        # For inference, we directly return the quantized vectors without
        # straight-through estimator (no gradients needed)
        hidden_state_quant = embedding_normalized[min_encoding_indices]

        # Reshape back to (batch, height, width, channels)
        # then (batch, channels, height, width)
        hidden_state_quant = (
            hidden_state_quant.view(batch_size, height, width, self.embedding_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return hidden_state_quant, min_encoding_indices


class GlmImageVQVAE(nn.Module):
    """
    VQ-VAE module for GLM-Image.

    Unlike Chameleon's VQ-VAE which includes a full encoder, GLM-Image's VQ-VAE
    only contains:
    - quant_conv: Projects from latent_channels to embed_dim
    - quantize: Vector quantizer
    - post_quant_conv: Projects from embed_dim back to latent_channels

    The encoder functionality is handled by GlmImageVisionModel instead.

    This module is always in eval mode as the VQ-VAE is frozen during inference.

    Args:
        config: GlmImageVQVAEConfig
    """

    def __init__(self, config: GlmImageVQVAEConfig):
        super().__init__()
        self.config = config

        # Vector quantizer
        self.quantize = GlmImageVQVAEVectorQuantizer(config)

        # Convolutions for projecting to/from embedding space
        # Using vLLM's optimized Conv2dLayer
        self.quant_conv = Conv2dLayer(
            in_channels=config.latent_channels,
            out_channels=config.embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.post_quant_conv = Conv2dLayer(
            in_channels=config.embed_dim,
            out_channels=config.latent_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # VQ-VAE is always frozen in GLM-Image
        self.eval()

    def encode(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input features into quantized latent codes.

        Args:
            hidden_states: Input tensor of shape (batch, latent_channels, height, width)
                          This is typically the output from GlmImageVisionModel reshaped
                          into spatial format.

        Returns:
            Tuple of:
                - quant: Quantized tensor of shape (batch, embed_dim, height, width)
                - indices: Codebook indices of shape (batch * height * width,)
        """
        # Project to embedding dimension
        hidden_states = self.quant_conv(hidden_states)

        # Quantize
        quant, indices = self.quantize(hidden_states)

        return quant, indices

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the model."""
        return self.quant_conv.weight.dtype

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return self.quant_conv.weight.device


# === Placeholder classes for components to be implemented ===
# These will be fully implemented in subsequent steps


class GlmImageVisionMLP(nn.Module):
    """Placeholder - to be implemented."""

    pass


class GlmImageVisionAttention(nn.Module):
    """Placeholder - to be implemented."""

    pass


class GlmImageVisionPatchEmbed(nn.Module):
    """Placeholder - to be implemented."""

    pass


class GlmImageVisionEmbeddings(nn.Module):
    """Placeholder - to be implemented."""

    pass


class GlmImageVisionBlock(nn.Module):
    """Placeholder - to be implemented."""

    pass


class GlmImageVisionModel(nn.Module):
    """Placeholder - to be implemented."""

    pass


# === Text Model Components ===


class GlmImageTextMLP(nn.Module):
    """
    MLP module for GLM-Image text model.

    Uses SiLU activation with gated linear units (SwiGLU variant).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for GLM-Image."
            )
        # Import here to avoid circular dependency
        from vllm.model_executor.layers.activation import SiluAndMul

        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class GlmImageTextAttention(nn.Module):
    """
    Multi-headed attention for GLM-Image text model.

    Uses Grouped Query Attention (GQA) with M-RoPE position embeddings.
    """

    def __init__(
        self,
        config: GlmImageTextConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 32768,
        quant_config: QuantizationConfig | None = None,
        bias: bool = True,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # M-RoPE for 3D position encoding
        rope_parameters = getattr(config, "rope_parameters", None)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class GlmImageTextDecoderLayer(nn.Module):
    """
    Decoder layer for GLM-Image text model.

    Key difference from standard LLaMA-style decoder:
    - Uses 4 RMSNorm layers instead of 2:
      - input_layernorm: before self-attention
      - post_self_attn_layernorm: after self-attention, before residual add
      - post_attention_layernorm: before MLP
      - post_mlp_layernorm: after MLP, before residual add
    """

    def __init__(
        self,
        config: GlmImageTextConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        attention_bias = getattr(config, "attention_bias", True)

        self.self_attn = GlmImageTextAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = GlmImageTextMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=False,
            prefix=f"{prefix}.mlp",
        )

        # GLM-Image uses 4 RMSNorm layers per decoder layer
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attn_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Save residual for first add
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # Post self-attention norm and residual add
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Return hidden_states and None for residual (already added)
        return hidden_states, None


class GlmImageTextModel(nn.Module):
    """
    Text model (language backbone) for GLM-Image.

    This is the decoder-only transformer that generates discrete image tokens.
    Uses M-RoPE (3D position encoding) for multimodal position awareness.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: GlmImageTextConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        # Embedding layer
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = None

        # Decoder layers
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: GlmImageTextDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=vllm_config.quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        # Final norm
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = None

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class GlmImageModel(nn.Module):
    """Placeholder - to be implemented."""

    pass


class GlmImageForConditionalGeneration(nn.Module):
    """Placeholder - to be implemented."""

    pass
