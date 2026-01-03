# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import cast

import torch
import torch.nn as nn

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)

DEFAULT_MAX_AUDIO_LEN_S = 655
DEFAULT_MERGE_FACTOR = 4
# Default convolution parameters: (padding, kernel_size, stride)
# These correspond to the two conv layers in GlmAsrEncoder
DEFAULT_CONV_PARAMS = [(1, 3, 1), (1, 3, 2)]


def _calculate_conv_output_length(
    input_length: torch.Tensor, padding: int, kernel_size: int, stride: int
) -> torch.Tensor:
    """Calculate Conv1d output length using standard formula."""
    # Standard formula: floor((input + 2*padding - kernel_size) / stride) + 1
    return (input_length + 2 * padding - kernel_size) // stride + 1


def _as_list_chunk_counts(
    chunk_counts: torch.Tensor | list[int] | list[torch.Tensor],
) -> list[int]:
    if isinstance(chunk_counts, torch.Tensor):
        return chunk_counts.tolist()
    if chunk_counts and isinstance(chunk_counts[0], torch.Tensor):
        tensor_counts = cast(list[torch.Tensor], chunk_counts)
        return [int(c.item()) for c in tensor_counts]
    return [int(c) for c in chunk_counts]


def _normalize_chunk_counts(
    chunk_counts: torch.Tensor | list[int] | list[torch.Tensor] | None,
    num_chunks: int,
) -> list[int]:
    if chunk_counts is None:
        return [1] * num_chunks
    return _as_list_chunk_counts(chunk_counts)


def _get_audio_output_lengths_from_lengths(
    audio_lengths: torch.Tensor,
    merge_factor: int,
    conv_params: list[tuple[int, int, int]],
) -> torch.Tensor:
    for padding, kernel_size, stride in conv_params:
        audio_lengths = _calculate_conv_output_length(
            audio_lengths, padding, kernel_size, stride
        )
    return (audio_lengths - merge_factor) // merge_factor + 1


def _get_audio_output_lengths_from_mask(
    mask: torch.Tensor,
    merge_factor: int,
    conv_params: list[tuple[int, int, int]],
) -> torch.Tensor:
    audio_lengths = mask.sum(-1)
    return _get_audio_output_lengths_from_lengths(
        audio_lengths, merge_factor, conv_params
    )


def _get_audio_output_lengths_for_tower(
    audio_tower: nn.Module,
    audio_lengths: torch.Tensor,
    merge_factor: int,
    conv_params: list[tuple[int, int, int]],
) -> torch.Tensor:
    if hasattr(audio_tower, "_get_feat_extract_output_lengths"):
        _, audio_output_lengths = audio_tower._get_feat_extract_output_lengths(
            audio_lengths
        )
        return audio_output_lengths
    return _get_audio_output_lengths_from_lengths(
        audio_lengths, merge_factor, conv_params
    )


def _flatten_audio_features_by_length(
    audio_features: torch.Tensor,
    audio_output_lengths: torch.Tensor,
) -> torch.Tensor:
    num_chunks, max_audio_tokens, embed_dim = audio_features.shape
    audio_output_lengths = audio_output_lengths.unsqueeze(1)
    audio_features_mask = (
        torch.arange(max_audio_tokens)
        .expand(num_chunks, max_audio_tokens)
        .to(audio_output_lengths.device)
        < audio_output_lengths
    )
    return audio_features[audio_features_mask].view(-1, embed_dim)


def _group_audio_embeddings(
    chunk_embeddings: Sequence[torch.Tensor],
    chunk_counts: Sequence[int],
) -> tuple[torch.Tensor, ...]:
    grouped_embeddings = []
    current_idx = 0
    for count in chunk_counts:
        audio_chunks = chunk_embeddings[current_idx : current_idx + count]
        grouped_embeddings.append(torch.cat(audio_chunks, dim=0))
        current_idx += count
    return tuple(grouped_embeddings)


def _normalize_to_tensor(mask: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    """Convert mask to tensor, handling both list and tensor formats."""
    if isinstance(mask, list):
        return (
            torch.stack(mask)
            if mask and isinstance(mask[0], torch.Tensor)
            else torch.tensor(mask)
        )
    return mask


def _extract_mask_for_item(
    feature_attention_mask: torch.Tensor | list[torch.Tensor],
    chunk_counts: torch.Tensor | list[int] | None,
    item_idx: int,
) -> torch.Tensor:
    """Extract attention mask for a specific audio item."""
    if chunk_counts is None:
        # Single item per audio
        mask = feature_attention_mask[item_idx]
        if isinstance(feature_attention_mask, torch.Tensor):
            return mask.unsqueeze(0)
        return _normalize_to_tensor(mask)

    # Multiple chunks per audio: calculate slice indices
    counts = _as_list_chunk_counts(chunk_counts)
    start_idx = sum(counts[:item_idx])
    end_idx = start_idx + counts[item_idx]

    # Extract slice
    if isinstance(feature_attention_mask, torch.Tensor):
        return feature_attention_mask[start_idx:end_idx]
    mask_slice = feature_attention_mask[start_idx:end_idx]
    return _normalize_to_tensor(mask_slice)


def _get_num_features_for_item(
    feature_attention_mask: torch.Tensor | None,
    chunk_counts: torch.Tensor | list[int] | None,
    item_idx: int,
    audio_embeds: list[torch.Tensor] | None,
    merge_factor: int,
    conv_params: list[tuple[int, int, int]],
) -> int:
    """Get number of features for a specific audio item."""
    if feature_attention_mask is not None:
        mask = _extract_mask_for_item(feature_attention_mask, chunk_counts, item_idx)
        audio_output_lengths = _get_audio_output_lengths_from_mask(
            mask, merge_factor, conv_params
        )
        return audio_output_lengths.sum().item()
    if audio_embeds is not None:
        return audio_embeds[item_idx].shape[0]
    raise ValueError("Either feature_attention_mask or audio_embeds must be provided")


# ============================================================================
# Optimized vLLM Native GlmAsrEncoder Implementation
# ============================================================================


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    cos = cos.unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin.unsqueeze(1)  # [bs, 1, seq_len, dim]

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class GlmAsrRotaryEmbedding(nn.Module):
    """Optimized Rotary Position Embedding for GLM-ASR encoder."""

    def __init__(self, config, device: torch.device | None = None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings

        # Compute inverse frequencies
        head_dim = config.hidden_size // config.num_attention_heads

        # Handle rope_parameters if present (for compatibility with transformers config)
        if hasattr(config, "rope_parameters") and config.rope_parameters:
            base = config.rope_parameters.get("rope_theta", 10000.0)
            partial_rotary_factor = config.rope_parameters.get(
                "partial_rotary_factor", 1.0
            )
            dim = int(head_dim * partial_rotary_factor)
        else:
            base = getattr(config, "rope_theta", 10000.0)
            dim = head_dim

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos and sin for common sequence lengths
        self._set_cos_sin_cache(config.max_position_embeddings, device)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device | None = None):
        """Pre-compute cos and sin cache for efficiency."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, num_heads, seq_len, head_dim]
            seq_len: Sequence length (optional, inferred from x if not provided)

        Returns:
            Tuple of (cos, sin) tensors for rotary embeddings
        """
        if seq_len is None:
            seq_len = x.shape[-2]

        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value tensors for Grouped Query Attention.

    Args:
        hidden_states: [batch, num_kv_heads, seq_len, head_dim]
        n_rep: Number of repetitions

    Returns:
        [batch, num_kv_heads * n_rep, seq_len, head_dim]
    """
    if n_rep == 1:
        return hidden_states

    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class GlmAsrAttention(nn.Module):
    """
    Optimized Multi-headed Grouped Query Attention for GLM-ASR.
    Uses vLLM's QKVParallelLinear for better performance.
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_rank = self.num_heads // self.tp_size
        self.num_kv_heads_per_rank = max(1, self.num_kv_heads // self.tp_size)

        # Use QKVParallelLinear for fused QKV projection
        # Note: GLM-ASR uses bias on Q and V, but not K
        # For simplicity with QKVParallelLinear, we use bias=True for all
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            position_embeddings: Tuple of (cos, sin) for RoPE

        Returns:
            [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [
                self.num_heads_per_rank * self.head_dim,
                self.num_kv_heads_per_rank * self.head_dim,
                self.num_kv_heads_per_rank * self.head_dim,
            ],
            dim=-1,
        )

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads_per_rank, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads_per_rank, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads_per_rank, self.head_dim)

        # Transpose to [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply rotary position embeddings
        cos, sin = position_embeddings
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # Handle GQA: repeat k/v if needed
        if self.num_kv_groups > 1:
            k = _repeat_kv(k, self.num_kv_groups)
            v = _repeat_kv(v, self.num_kv_groups)

        # Scaled dot-product attention (Flash Attention if available)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Output projection
        output, _ = self.o_proj(attn_output)
        return output


class GlmAsrMLP(nn.Module):
    """
    Optimized MLP for GLM-ASR encoder.
    Uses vLLM's parallel linear layers for better performance.
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.fc1 = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )

        self.act_fn = get_act_fn(config.hidden_act)

        self.fc2 = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class GlmAsrEncoderLayer(nn.Module):
    """
    Optimized Transformer encoder layer for GLM-ASR.
    Combines attention and MLP with residual connections and layer norms.
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GlmAsrAttention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        self.mlp = GlmAsrMLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        layer_norm_eps = getattr(config, "layer_norm_eps", 1e-5)
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(
            self.hidden_size, eps=layer_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            position_embeddings: Tuple of (cos, sin) for RoPE

        Returns:
            [batch_size, seq_len, hidden_size]
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GlmAsrEncoder(nn.Module):
    """
    Optimized GLM-ASR Audio Encoder with vLLM native implementation.

    This encoder processes audio features through convolutional layers
    followed by transformer layers with rotary position embeddings.
    Optimized for performance with:
    - QKVParallelLinear for fused attention projections
    - Tensor parallelism support via ColumnParallelLinear/RowParallelLinear
    - Quantization support
    - Flash Attention (SDPA)
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        # Convolutional feature extraction layers
        self.conv1 = nn.Conv1d(
            config.num_mel_bins,
            config.hidden_size,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Transformer encoder layers
        self.layers = nn.ModuleList(
            [
                GlmAsrEncoderLayer(
                    config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # Final layer norm
        layer_norm_eps = getattr(config, "layer_norm_eps", 1e-5)
        self.norm = nn.LayerNorm(config.hidden_size, eps=layer_norm_eps)

        # Rotary position embeddings
        self.rotary_emb = GlmAsrRotaryEmbedding(config)

    def _get_feat_extract_output_lengths(
        self, input_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the output length after convolutions.

        Args:
            input_lengths: Input sequence lengths [batch_size]

        Returns:
            Tuple of (output after conv1, output after conv2)
        """
        # Conv1: kernel=3, stride=1, padding=1
        output_lengths = (input_lengths + 2 * 1 - 3) // 1 + 1

        # Conv2: kernel=3, stride=2, padding=1
        output_lengths = (output_lengths + 2 * 1 - 3) // 2 + 1

        return input_lengths, output_lengths

    def forward(self, input_features: torch.Tensor):
        """
        Forward pass through the encoder.

        Args:
            input_features: [batch_size, num_mel_bins, seq_len]

        Returns:
            Object with .last_hidden_state attribute containing
            [batch_size, seq_len', hidden_size] where seq_len' is
            the sequence length after convolutions
        """
        # Apply convolutional layers with GELU activation
        hidden_states = torch.nn.functional.gelu(self.conv1(input_features))
        hidden_states = torch.nn.functional.gelu(self.conv2(hidden_states))

        # Transpose to [batch_size, seq_len, hidden_size]
        hidden_states = hidden_states.transpose(1, 2)

        batch_size, seq_len, _ = hidden_states.shape

        # Get position embeddings
        position_embeddings = self.rotary_emb(hidden_states, seq_len)

        # Apply transformer layers
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, position_embeddings)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Return in a format compatible with transformers' BaseModelOutput
        class EncoderOutput:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state

        return EncoderOutput(last_hidden_state=hidden_states)
