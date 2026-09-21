# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 MoE-tail fusion: deferred finalize + flashinfer
``kMoEFinalizeARResidualRMSNorm`` (MoE finalize + all-reduce + residual add +
RMSNorm in one PDL-chained launch).

Replaces the decode-path tail chain per MoE layer:
    [trtllm finalize] -> [TP all-reduce] -> [residual add + RMSNorm]

The whole MoE+tail is one opaque custom op (``vllm::mimo_moe_forward_tail``),
registered per-layer via the same static-forward-context mechanism as
``moe_forward``, so torch.compile never inlines the runner internals and the
unfinalized MoE output never crosses a Tensor-typed op boundary.

- M <= TAIL_FUSION_MAX_TOKENS: deferred MoE finalize + flashinfer fused kernel
- larger M: standard finalize/AR, then identical residual+norm math in torch

Enabled with ``VLLM_MIMO_V2_FUSED_MOE_TAIL=1``.  Falls back transparently.
"""

import torch

import vllm.envs as envs
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
    get_layer_from_name,
)
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# Only use the FI fused kernel at decode-ish batch sizes (LL protocol's home);
# larger M uses the standard path plus the same residual+norm math in torch.
TAIL_FUSION_MAX_TOKENS = 128

# Process-wide TRTLLM workspace for the MoE-finalize pattern (the shared
# get_fi_ar_workspace may be MNNVL, which does not support it).
_TAIL_WORKSPACE = None


def _get_tail_workspace(tp_group, dtype, hidden_dim):
    global _TAIL_WORKSPACE
    if _TAIL_WORKSPACE is None:
        from flashinfer.comm import TRTLLMAllReduceFusionWorkspace

        _TAIL_WORKSPACE = TRTLLMAllReduceFusionWorkspace(
            tp_size=tp_group.world_size,
            tp_rank=tp_group.rank_in_group,
            max_token_num=TAIL_FUSION_MAX_TOKENS,
            hidden_dim=hidden_dim,
            dtype=dtype,
            group=tp_group.device_group,
        )
    return _TAIL_WORKSPACE


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * weight.float()).to(x.dtype)


def _mimo_moe_forward_tail_impl(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    runner = get_layer_from_name(layer_name)
    assert isinstance(runner, MiMoTailRunner) and runner.tail_fusion_enabled

    if runner.moe_config.should_defer_moe_finalize(hidden_states.shape[0]):
        # Deferred (unfinalized) MoE output, consumed locally by the fused op.
        unfinalized = runner._forward_impl(hidden_states, router_logits, None, None)
        assert isinstance(unfinalized, UnfinalizedMoEOutput), (
            f"expected UnfinalizedMoEOutput, got {type(unfinalized)}"
        )
        import flashinfer.comm as flashinfer_comm

        tp_group = get_tp_group()
        workspace = _get_tail_workspace(tp_group, residual.dtype,
                                        residual.shape[-1])
        if workspace is None:
            raise RuntimeError("FlashInfer allreduce workspace unavailable")
        norm_out = torch.empty_like(residual)
        residual_out = torch.empty_like(residual)
        flashinfer_comm.allreduce_fusion(
            input=unfinalized.gemm2_permuted,
            workspace=workspace,
            pattern=flashinfer_comm.AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
            launch_with_pdl=True,
            residual_in=residual,
            residual_out=residual_out,
            norm_out=norm_out,
            rms_gamma=norm_weight,
            rms_eps=rms_eps,
            expanded_idx_to_permuted_idx=unfinalized.expanded_idx_to_permuted_idx,
            expert_scale_factor=unfinalized.expert_weights,
            shared_expert_output=None,
        )
        return norm_out, residual_out

    # Prefill-sized M: standard finalize + AR, then the same residual+norm
    # math in plain torch (identical values, unfused).
    out = MoERunner.forward(runner, hidden_states, router_logits)
    residual_out = residual + out
    return _rms_norm(residual_out, norm_weight, rms_eps), residual_out


def _mimo_moe_forward_tail_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(residual), torch.empty_like(residual)


direct_register_custom_op(
    op_name="mimo_moe_forward_tail",
    op_func=_mimo_moe_forward_tail_impl,
    mutates_args=[],
    fake_impl=_mimo_moe_forward_tail_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def tail_fusion_supported(moe_config, quant_method) -> bool:
    """Conditions for the fused tail on this deployment (checked once)."""
    from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
        TrtLlmMxfp4ExpertsMonolithic,
    )

    experts_cls = getattr(quant_method, "experts_cls", None)
    return (
        envs.VLLM_MIMO_V2_FUSED_MOE_TAIL
        and experts_cls is TrtLlmMxfp4ExpertsMonolithic
        and moe_config.tp_size in (8, 16)
        and moe_config.dp_size == 1
        and moe_config.ep_size == 1
        and moe_config.pcp_size == 1
        and not moe_config.is_sequence_parallel
        and moe_config.hidden_dim == moe_config.hidden_dim_unpadded
        and get_tp_group().world_size > 1
    )


class MiMoTailRunner(MoERunner):
    """MoERunner that defers the decode tail for the fused finalize op."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tail_fusion_enabled = tail_fusion_supported(
            self.moe_config, self._quant_method
        )
        if self.tail_fusion_enabled:
            self.moe_config.defer_moe_finalize = True
            self.moe_config.defer_moe_finalize_max_num_tokens = (
                TAIL_FUSION_MAX_TOKENS
            )
            logger.info_once(
                "MiMo fused MoE tail (finalize+AR+residual+RMSNorm) enabled "
                "for up to %d tokens.",
                TAIL_FUSION_MAX_TOKENS,
            )

    def forward_fused_tail(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        rms_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Entry from MiMoV2MoE.forward; goes through the opaque custom op."""
        return torch.ops.vllm.mimo_moe_forward_tail(
            hidden_states,
            router_logits,
            residual,
            norm_weight,
            rms_eps,
            self.layer_name,
        )
