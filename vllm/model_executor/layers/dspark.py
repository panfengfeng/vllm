# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Markov and acceptance-confidence heads for DSpark drafting."""

import torch
import torch.nn as nn

from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
    nvfp4_gathered_bias,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform


class DSparkMarkovHead(nn.Module):
    """Sequential transition-bias head (low-rank V x r, r x V).

    ``markov_w1[token]`` embeds the previously sampled token (target vocab,
    ``vocab_size``); ``markov_w2`` projects it to a draft-vocab bias
    (``draft_vocab_size``) added to the base draft logits. The two sizes
    coincide for full-vocab drafts.

    Both weights are replicated because the head runs sequentially for every
    draft position. Sharding them would add an all-reduce and a full-vocab
    gather to each position.
    """

    def __init__(
        self,
        vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
        retain_weight_for_gather: bool = False,
    ) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = ParallelLMHead(
            draft_vocab_size,
            markov_rank,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "markov_w2"),
            disable_tp=True,
        )
        self.markov_w2._retain_weight_for_gather = retain_weight_for_gather
        self.markov_w2.is_w4a16_nvfp4 = False

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """r-dim Markov embedding of ``token_ids`` ([B] -> [B, r])."""
        return self.markov_w1(token_ids)

    def bias(
        self,
        markov_embed: torch.Tensor,
        logits_processor: LogitsProcessor,
    ) -> torch.Tensor:
        """Vocab-size transition bias from a Markov embedding ([B, r] -> [B, V])."""
        return logits_processor(self.markov_w2, markov_embed)

    def apply_bias_gathered(
        self,
        markov_embed: torch.Tensor,
        logits: torch.Tensor,
        values: torch.Tensor,
        index: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Apply the Markov bias only to selected rows of ``logits``.

        The caller initializes ``logits`` to ``-inf`` once for all draft
        positions. This method scatters the corrected candidate values into
        that dense buffer so the normal sampler sees the truncated proposal.
        """
        if self.markov_w2._retain_weight_for_gather and self.markov_w2.is_w4a16_nvfp4:
            # Use the fast Triton kernel for W4A16 NVFP4 weights with BF16
            # activations on CUDA. FP16 activations, non-CUDA accelerators, and
            # CPU use the slower dequantize/scatter fallback below.
            if (
                current_platform.is_cuda()
                and markov_embed.is_cuda
                and markov_embed.dtype == torch.bfloat16
                and values.dtype == torch.bfloat16
                and logits.dtype == torch.bfloat16
            ):
                nvfp4_gathered_bias(
                    markov_embed,
                    self.markov_w2._nvfp4_weight_for_gather,
                    self.markov_w2._nvfp4_weight_scale_for_gather,
                    self.markov_w2._nvfp4_weight_global_scale_for_gather,
                    values,
                    index,
                    logits,
                    scale,
                )
                return logits

            flat_index = index.reshape(-1)
            packed_weight = self.markov_w2._nvfp4_weight_for_gather.index_select(
                0, flat_index
            )
            weight_scale = self.markov_w2._nvfp4_weight_scale_for_gather.index_select(
                0, flat_index
            )
            weight = dequantize_to_dtype(
                packed_weight,
                weight_scale,
                self.markov_w2._nvfp4_weight_global_scale_for_gather,
                dtype=markov_embed.dtype,
                block_size=self.markov_w2._nvfp4_group_size_for_gather,
                swizzle=False,
            ).view(*index.shape, -1)
        else:
            weight = self.markov_w2.weight[index]
            if weight.dtype != markov_embed.dtype:
                raise RuntimeError(
                    "Gathered Markov projection requires unquantized weights or "
                    "retained ModelOpt W4A16 NVFP4 weights."
                )
        corrected = values.unsqueeze(-1).clone()
        corrected.baddbmm_(
            weight,
            markov_embed.unsqueeze(-1),
            beta=1.0,
            alpha=scale,
        )
        return logits.scatter_(1, index, corrected.squeeze(-1))


class DSparkConfidenceHead(nn.Module):
    """DSpark acceptance-confidence head."""

    def __init__(
        self,
        input_dim: int,
        prefix: str,
        bias: bool = False,
        with_markov: bool = True,
    ) -> None:
        super().__init__()
        self.with_markov = with_markov
        self.proj = ReplicatedLinear(
            input_dim,
            1,
            bias=bias,
            return_bias=False,
            params_dtype=torch.float32,
            prefix=maybe_prefix(prefix, "proj"),
        )

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        x = (
            torch.cat([hidden, markov_embed], dim=-1) if self.with_markov else hidden
        ).float()
        return self.proj(x).squeeze(-1)
