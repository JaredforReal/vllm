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
    GlmImageVisionConfig,
    GlmImageVQVAEConfig,
)

from vllm.attention.layer import Attention
from vllm.config import CacheConfig, MultiModalConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.distributed import utils as dist_utils
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv2dLayer
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
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
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .utils import (
    make_empty_intermediate_tensors_factory,
    make_layers,
)
from .vision import get_vit_attn_backend

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


# === Vision Model Components ===


class GlmImageVisionMLP(nn.Module):
    """
    MLP module for GLM-Image vision model.

    Uses GELU activation with standard fc1 -> fc2 structure.
    Key difference from Glm4vVisionMLP: uses GELU instead of SwiGLU.
    """

    def __init__(
        self,
        config: GlmImageVisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = (
            multimodal_config.mm_encoder_tp_mode == "data"
            if multimodal_config
            else False
        )
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
            disable_tp=use_data_parallel,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
            disable_tp=use_data_parallel,
        )
        self.act_fn = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.act_fn(x)
        x, _ = self.fc2(x)
        return x


class GlmImageVisionAttention(nn.Module):
    """
    Multi-headed attention for GLM-Image vision model.

    Key differences from Glm4vVisionAttention:
    - No RoPE - uses learned position embeddings instead
    - Uses standard qkv projection (not separate q, k, v)
    - attention_bias from config controls bias in linear layers
    """

    def __init__(
        self,
        config: GlmImageVisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = (
            multimodal_config.mm_encoder_tp_mode == "data"
            if multimodal_config
            else False
        )
        self.tp_size = (
            1 if use_data_parallel else get_tensor_model_parallel_world_size()
        )

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        attention_bias = getattr(config, "attention_bias", True)

        self.num_heads_per_partition = dist_utils.divide(self.num_heads, self.tp_size)

        # QKV projection - uses bias based on config
        self.qkv = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_heads,  # No GQA in vision model
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=use_data_parallel,
        )
        self.proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=use_data_parallel,
        )

        # MMEncoderAttention for efficient vision attention
        self.attn = MMEncoderAttention(
            num_heads=self.num_heads_per_partition,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
            multimodal_config=multimodal_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        # hidden_states: [seq_len, hidden_size] (no batch dim)
        seq_len = hidden_states.shape[0]

        # QKV projection
        qkv, _ = self.qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for attention: [seq, hidden] -> [1, seq, heads, head_dim]
        q = q.view(seq_len, self.num_heads_per_partition, self.head_dim).unsqueeze(0)
        k = k.view(seq_len, self.num_heads_per_partition, self.head_dim).unsqueeze(0)
        v = v.view(seq_len, self.num_heads_per_partition, self.head_dim).unsqueeze(0)

        # No RoPE in GLM-Image vision model - position info comes from embeddings

        # Apply attention
        attn_output = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        # Reshape back: [1, seq, heads, head_dim] -> [seq, hidden]
        attn_output = attn_output.view(seq_len, -1)

        # Output projection
        output, _ = self.proj(attn_output)
        return output


class GlmImageVisionPatchEmbed(nn.Module):
    """
    Patch embedding for GLM-Image vision model.

    Key difference from Glm4vVisionPatchEmbed:
    - Uses 2D convolution (no temporal dimension)
    - GLM-Image processes single images, not videos
    """

    def __init__(self, config: GlmImageVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        # 2D convolution for patch embedding
        self.proj = Conv2dLayer(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Packed pixel values of shape
                [total_patches, in_channels * patch_size * patch_size]

        Returns:
            Patch embeddings of shape [total_patches, embed_dim]
        """
        target_dtype = self.proj.weight.dtype
        # Reshape from [N, C*P*P] to [N, C, P, P]
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.patch_size, self.patch_size
        )
        # Conv2d and flatten: [N, C, P, P] -> [N, embed_dim, 1, 1] -> [N, embed_dim]
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states


class GlmImageVisionEmbeddings(nn.Module):
    """
    Vision embeddings for GLM-Image.

    Uses learned 2D position embeddings with bilinear interpolation
    for variable resolution support.

    Key difference from Glm4vVisionEmbeddings:
    - Uses bilinear interpolation (not bicubic) for position embedding adaptation
    """

    def __init__(self, config: GlmImageVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        # GLM-Image uses bilinear, Glm4v uses bicubic
        self.interpolation_mode = "bilinear"

    def forward(
        self,
        embeddings: torch.Tensor,
        lengths: list[int] | torch.Tensor,
        image_shapes: torch.Tensor,
        h_coords: torch.Tensor,
        w_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add adapted position embeddings to patch embeddings.

        Args:
            embeddings: Patch embeddings [total_seq, embed_dim]
            lengths: Sequence length for each image
            image_shapes: [num_images, 3] with (t, h, w) for each image
            h_coords: Height coordinates for each patch [total_seq]
            w_coords: Width coordinates for each patch [total_seq]

        Returns:
            Embeddings with position encoding added [total_seq, embed_dim]
        """
        pos_embed_weight = self.position_embedding.weight
        hidden_size = pos_embed_weight.shape[1]
        total_seq = h_coords.shape[0]
        device = pos_embed_weight.device

        # Handle empty sequence case
        if total_seq == 0:
            adapted_pos_embed = torch.empty(
                0, hidden_size, device=device, dtype=pos_embed_weight.dtype
            )
        else:
            # Convert to tensors if needed
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths, device=device, dtype=torch.long)
            if not isinstance(image_shapes, torch.Tensor):
                image_shapes = torch.tensor(
                    image_shapes, device=device, dtype=torch.long
                )

            # Prepare 2D position embedding for interpolation
            orig_size_sq = pos_embed_weight.shape[0]
            orig_size = int(orig_size_sq**0.5)
            pos_embed_2d = (
                pos_embed_weight.view(orig_size, orig_size, hidden_size)
                .permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
                .unsqueeze(0)  # [1, C, H, W]
                .to(device=device, dtype=torch.float32)
            )

            # Calculate target dimensions for each patch
            target_h = torch.cat(
                [image_shapes[i, 1].repeat(lengths[i]) for i in range(len(lengths))]
            ).to(device=device, dtype=torch.float32)
            target_w = torch.cat(
                [image_shapes[i, 2].repeat(lengths[i]) for i in range(len(lengths))]
            ).to(device=device, dtype=torch.float32)

            # Normalize coordinates to [-1, 1] for grid_sample
            h_coords = h_coords.to(device=device, dtype=torch.float32)
            w_coords = w_coords.to(device=device, dtype=torch.float32)
            norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
            norm_h = ((h_coords + 0.5) / target_h) * 2 - 1

            # Create sampling grid [1, total_seq, 1, 2]
            grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)

            # Bilinear interpolation (GLM-Image uses bilinear, not bicubic)
            interpolated_embed = F.grid_sample(
                pos_embed_2d,
                grid,
                mode=self.interpolation_mode,
                align_corners=False,
                padding_mode="border",
            )

            # Reshape: [1, C, total_seq, 1] -> [total_seq, C]
            adapted_pos_embed = (
                interpolated_embed.squeeze(0).squeeze(-1).permute(1, 0)
            ).to(pos_embed_weight.dtype)

        # Add position embedding to patch embeddings
        embeddings = embeddings + adapted_pos_embed.to(embeddings.device)
        return embeddings


class GlmImageVisionBlock(nn.Module):
    """
    Transformer block for GLM-Image vision model.

    Key differences from Glm4vVisionBlock:
    - Uses LayerNorm instead of RMSNorm
    - No RoPE position embeddings (handled in GlmImageVisionEmbeddings)
    - Uses GELU MLP instead of SwiGLU
    """

    def __init__(
        self,
        config: GlmImageVisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = GlmImageVisionAttention(
            config,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = GlmImageVisionMLP(
            config,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        # Pre-norm MLP
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GlmImageVisionModel(nn.Module):
    """
    Vision encoder for GLM-Image.

    Key differences from Glm4vVisionTransformer:
    - No RoPE - uses learned position embeddings with bilinear interpolation
    - No merger, downsample, or post-processing layers
    - Uses LayerNorm instead of RMSNorm in blocks
    - No temporal dimension (images only, no video)
    """

    def __init__(
        self,
        config: GlmImageVisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.patch_size = config.patch_size
        self.spatial_merge_size = config.spatial_merge_size

        # Patch embedding
        self.patch_embed = GlmImageVisionPatchEmbed(config)

        # Position embeddings
        self.embeddings = GlmImageVisionEmbeddings(config)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                GlmImageVisionBlock(
                    config,
                    quant_config=quant_config,
                    multimodal_config=multimodal_config,
                    prefix=f"{prefix}.blocks.{i}",
                )
                for i in range(config.depth)
            ]
        )

        # Attention backend selection
        self.attn_backend = get_vit_attn_backend(
            head_size=self.head_dim,
            dtype=torch.get_default_dtype(),
            attn_backend_override=(
                multimodal_config.mm_encoder_attn_backend if multimodal_config else None
            ),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def compute_position_ids(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Compute position IDs for each patch based on grid dimensions.

        Args:
            grid_thw: [num_images, 3] with (t, h, w) for each image

        Returns:
            Position IDs [total_patches, 2] with (h_pos, w_pos) for each patch
        """
        pos_ids = []
        for t, h, w in grid_thw:
            # Create h and w position grids
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)

            # Reshape for spatial merge
            hpos_ids = (
                hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )

            wpos_ids = (
                wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )

            # Stack and repeat for temporal dimension
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        return torch.cat(pos_ids, dim=0)

    def compute_attn_mask_seqlen(
        self,
        cu_seqlens: torch.Tensor,
    ) -> int | None:
        """Compute max sequence length for flash attention."""
        if (
            self.attn_backend == AttentionBackendEnum.FLASH_ATTN
            or self.attn_backend == AttentionBackendEnum.ROCM_AITER_FA
        ):
            return (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        return None

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through vision encoder.

        Args:
            pixel_values: Packed pixel values
                [total_patches, num_channels * patch_size * patch_size]
            grid_thw: [num_images, 3] with (t, h, w) for each image

        Returns:
            Hidden states [total_patches, hidden_size]
        """
        # Patch embedding
        hidden_states = self.patch_embed(pixel_values.to(self.device, self.dtype))

        # Compute position IDs
        position_ids = self.compute_position_ids(grid_thw)

        # Compute cumulative sequence lengths for attention
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        cu_seqlens = cu_seqlens.to(self.device)

        # Get sequence lengths for position embedding
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        # Add position embeddings
        hidden_states = self.embeddings(
            hidden_states,
            seqlens,
            grid_thw,
            position_ids[:, 0].to(hidden_states.device),
            position_ids[:, 1].to(hidden_states.device),
        )

        # Compute max seqlen for flash attention
        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)

        # Transformer blocks
        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        return hidden_states


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
    """
    GLM-Image model that combines Vision Encoder, VQ-VAE, and Text Model.

    This model is used for image generation tasks:
    - Image-to-Image: Source image → Vision Encoder → VQ-VAE tokens → Text Model
    - Text-to-Image: Text tokens → Text Model → Generate image tokens

    Key components:
    - visual: GlmImageVisionModel for encoding input images
    - vqmodel: GlmImageVQVAE for tokenizing image features
    - language_model: GlmImageTextModel for text/token generation

    The model uses M-RoPE (3D position encoding) for multimodal position awareness:
    - temporal: constant for image tokens, incremental for text
    - height: row position for image tokens
    - width: column position for image tokens
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        # Vision encoder
        self.visual = GlmImageVisionModel(
            config.vision_config,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=f"{prefix}.visual" if prefix else "visual",
        )

        # VQ-VAE for image tokenization (frozen)
        self.vqmodel = GlmImageVQVAE(config.vq_config)

        # Text/Language model
        self.language_model = GlmImageTextModel(
            vllm_config=vllm_config,
            config=config.text_config,
            prefix=f"{prefix}.language_model" if prefix else "language_model",
        )

        # Store special token IDs
        self.image_token_id = config.image_token_id
        self.image_start_token_id = config.image_start_token_id
        self.image_end_token_id = config.image_end_token_id

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.language_model.get_input_embeddings()

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract image features using the vision encoder.

        Args:
            pixel_values: Packed pixel values
                [total_patches, num_channels * patch_size * patch_size]
            image_grid_thw: [num_images, 3] with (t, h, w) for each image

        Returns:
            Image features [total_patches, hidden_size]
        """
        return self.visual(pixel_values, image_grid_thw)

    def get_image_tokens(
        self,
        hidden_states: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tokenize image features into discrete tokens using VQ-VAE.

        Args:
            hidden_states: Image features [total_patches, hidden_size]
            image_grid_thw: [num_images, 3] with (t, h, w) for each image

        Returns:
            Discrete token indices [total_patches]
        """
        hidden_size = hidden_states.shape[-1]
        split_sizes = (image_grid_thw.prod(dim=-1)).tolist()
        hidden_states_list = torch.split(hidden_states, split_sizes, dim=0)

        all_image_tokens = []
        for i, hs in enumerate(hidden_states_list):
            grid_t, grid_h, grid_w = image_grid_thw[i].tolist()
            # Reshape to spatial format: [t, h, w, c] -> [t, c, h, w]
            hs = hs.view(grid_t, grid_h, grid_w, hidden_size)
            hs = hs.permute(0, 3, 1, 2).contiguous()
            # Encode with VQ-VAE
            _, indices = self.vqmodel.encode(hs)
            all_image_tokens.append(indices)

        return torch.cat(all_image_tokens, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """
        Forward pass through the GLM-Image model.

        For image-to-image generation:
        1. Encode source images with vision encoder
        2. Tokenize features with VQ-VAE
        3. Replace placeholder tokens with actual image tokens
        4. Run through language model

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            positions: Position IDs, shape (3, seq_len) for M-RoPE
            intermediate_tensors: For pipeline parallelism
            inputs_embeds: Pre-computed embeddings (optional)
            pixel_values: Source image pixels (for image-to-image)
            image_grid_thw: Grid dimensions for source images

        Returns:
            Hidden states or intermediate tensors for PP
        """
        # Handle intermediate tensors for pipeline parallelism
        if intermediate_tensors is not None:
            return self.language_model(
                input_ids=None,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=None,
            )

        # Process source images if provided (image-to-image generation)
        if pixel_values is not None and image_grid_thw is not None:
            # Encode images
            image_features = self.get_image_features(pixel_values, image_grid_thw)
            # Tokenize with VQ-VAE
            image_tokens = self.get_image_tokens(image_features, image_grid_thw)
            image_tokens = image_tokens.to(input_ids.device)

            # Replace placeholder tokens with actual image tokens
            special_image_mask = input_ids == self.image_token_id
            if special_image_mask.sum() > 0:
                input_ids = input_ids.clone()
                input_ids[special_image_mask] = image_tokens

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            input_ids = None

        # Forward through language model
        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states


class GlmImageForConditionalGeneration(nn.Module):
    """Placeholder - to be implemented."""

    pass
