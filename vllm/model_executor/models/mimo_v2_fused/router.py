# SPDX-License-Identifier: Apache-2.0
"""Fused-router integration for MiMo-V2 MoE layers.

``MimoFusedRouter`` replaces the decode-path chain
    gate GEMM -> sigmoid -> bias add -> grouped topk
with :func:`fused_router_topk` when the batch is small (M <= 16, the decode
regime) and the gate weight is bf16. Larger batches and other dtypes fall
back to the default GroupedTopKRouter path unchanged.

The router ignores ``router_logits`` in the fused path and computes logits
from ``hidden_states`` directly, so the caller must skip its gate GEMM
(:func:`MiMoV2MoE.forward` does this when the fused router is active).
"""

import torch

from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

from .router_topk import fused_router_topk

# decode-only fast path
MAX_FUSED_TOKENS = 16


class MimoFusedRouter(BaseRouter):
    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        gate: torch.nn.Linear,
        renormalize: bool = True,
    ):
        super().__init__(top_k=top_k, global_num_experts=global_num_experts)
        self.gate = gate
        self.renormalize = renormalize
        self._fallback = None
        self._bias_f32 = None

    # GroupedTopKRouter-compatible fallback, resolved lazily to avoid import
    # cycles at module load time.
    def _get_fallback(self):
        if self._fallback is None:
            from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (  # noqa: E501
                GroupedTopKRouter,
            )

            self._fallback = GroupedTopKRouter(
                top_k=self.top_k,
                global_num_experts=self.global_num_experts,
                num_expert_group=1,
                topk_group=1,
                renormalize=self.renormalize,
                scoring_func="sigmoid",
                e_score_correction_bias=self.gate.e_score_correction_bias,
            )
        return self._fallback

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return self._get_fallback().routing_method_type

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.gate.weight
        if (
            hidden_states.shape[0] <= MAX_FUSED_TOKENS
            and hidden_states.dtype == torch.bfloat16
            and w.dtype == torch.bfloat16
            and w.shape[0] % 128 == 0
            and w.shape[1] % 64 == 0
        ):
            bias = self.gate.e_score_correction_bias
            if bias.dtype != torch.float32:
                if self._bias_f32 is None:
                    self._bias_f32 = bias.detach().float().contiguous()
                bias = self._bias_f32
            return fused_router_topk(
                hidden_states,
                w,
                bias,
                topk=self.top_k,
                renormalize=self.renormalize,
            )
        # fallback: router_logits must be the real logits from the gate
        return self._get_fallback()._compute_routing(
            hidden_states, router_logits, indices_type, input_ids=input_ids
        )
