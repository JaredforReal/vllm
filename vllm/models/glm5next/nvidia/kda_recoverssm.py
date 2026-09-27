# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA attention backend for RecoverSSM speculative decoding.

Reuses Kimi-K3's KDA metadata builder (spec/non-spec partition, align-mode
commit plan) and swaps in GLM's commit context, whose records hold the
normalized key and decay in fp32 (see ops/recoverssm.py).
"""

from vllm.models.glm5next.nvidia.ops.recoverssm import GlmKDARecoverSSMCommitContext
from vllm.models.kimi_k3.nvidia.kda_metadata import (
    KimiK3KDAAttentionBackend,
    KimiK3KDAMetadataBuilder,
)


class Glm5NextKDARecoverSSMMetadataBuilder(KimiK3KDAMetadataBuilder):
    def _get_recoverssm_context(self) -> GlmKDARecoverSSMCommitContext:
        context = self.recoverssm_context
        if context is None:
            forward_context = self.vllm_config.compilation_config.static_forward_context
            context = GlmKDARecoverSSMCommitContext.create(
                [forward_context[name] for name in self.layer_names],
                spec_query_len=1 + self.vllm_config.num_speculative_tokens,
                max_num_reqs=self.vllm_config.scheduler_config.max_num_seqs,
            )
            self.recoverssm_context = context
        assert isinstance(context, GlmKDARecoverSSMCommitContext)
        return context


class Glm5NextKDARecoverSSMBackend(KimiK3KDAAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[Glm5NextKDARecoverSSMMetadataBuilder]:
        return Glm5NextKDARecoverSSMMetadataBuilder
