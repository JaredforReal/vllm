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
    GlmImageVQVAEConfig,
)

from vllm.logger import init_logger
from vllm.model_executor.layers.conv import Conv2dLayer

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


class GlmImageTextDecoderLayer(nn.Module):
    """Placeholder - to be implemented."""

    pass


class GlmImageTextModel(nn.Module):
    """Placeholder - to be implemented."""

    pass


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


class GlmImageModel(nn.Module):
    """Placeholder - to be implemented."""

    pass


class GlmImageForConditionalGeneration(nn.Module):
    """Placeholder - to be implemented."""

    pass
