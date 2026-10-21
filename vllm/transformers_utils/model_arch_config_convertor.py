# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import final

import torch
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from transformers import PreTrainedConfig

from vllm import envs
from vllm.config.model_arch import (
    ModelArchitectureConfig,
)
from vllm.config.utils import getattr_iter
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    ConfigFormat,
    get_safetensors_params_metadata,
)
from vllm.utils.torch_utils import common_broadcastable_dtype

logger = init_logger(__name__)


class ModelArchConfigConvertorBase:
    def __init__(
        self,
        hf_config: PreTrainedConfig,
        hf_text_config: PreTrainedConfig,
        revision: str | None = None,
    ):
        self.hf_config = hf_config
        self.hf_text_config = hf_text_config
        self.revision = revision

    def get_per_layer_hf_configs(
        self,
    ) -> list[tuple[PreTrainedConfig, PreTrainedConfig]] | None:
        """`(hf_config, hf_text_config)` per layer, or `None` if homogeneous.

        This is the only place that decides whether a checkpoint is heterogeneous
        and how many layers it has, so a convertor can declare variation that
        Transformers does not express.

        Each pair is converted independently and the results are diffed, so a
        convertor that overrides this must not also collapse a varying field in
        its getters: the collapse would then be applied per layer and every layer
        would report the same value.
        """
        config_het = getattr(self.hf_config, "is_heterogeneous", False)
        text_het = getattr(self.hf_text_config, "is_heterogeneous", False)
        if not (config_het or text_het):
            return None
        # The text config decides the layer count when it is the heterogeneous one:
        # its stack is what vLLM builds attention layers for.
        source = self.hf_text_config if text_het else self.hf_config
        return [
            (
                self.hf_config.per_layer_config[i] if config_het else self.hf_config,
                self.hf_text_config.per_layer_config[i]
                if text_het
                else self.hf_text_config,
            )
            for i in range(len(source.per_layer_config))
        ]

    def get_architectures(self) -> list[str]:
        # Sometimes we get here from `vllm_config.with_hf_config(text_config)` where
        # `text_config` is a sub-config from a multi-modal model. If this is the case,
        # the sub-config will not have `architectures` and it will explicitly be `None`
        return getattr(self.hf_config, "architectures", None) or []

    def get_num_hidden_layers(self) -> int:
        return getattr(self.hf_text_config, "num_hidden_layers", 0)

    def get_total_num_attention_heads(self) -> int:
        return getattr(self.hf_text_config, "num_attention_heads", 0)

    def get_vocab_size(self) -> int:
        return getattr(self.hf_text_config, "vocab_size", 0)

    def get_hidden_size(self) -> int:
        return getattr(self.hf_text_config, "hidden_size", 0)

    def get_head_size(self) -> int:
        if self.is_deepseek_mla():
            # special case for deepseek_v4
            if hasattr(self.hf_text_config, "compress_ratios"):
                return self.hf_text_config.head_dim
            qk_rope_head_dim = self._get_qk_rope_head_dim()
            if not envs.VLLM_MLA_DISABLE:
                return self.hf_text_config.kv_lora_rank + qk_rope_head_dim
            else:
                qk_nope_head_dim = getattr(self.hf_text_config, "qk_nope_head_dim", 0)
                if qk_rope_head_dim and qk_nope_head_dim:
                    return qk_rope_head_dim + qk_nope_head_dim

        # NOTE: Some config classes may set head_dim=None or materialize a missing
        # head_dim as 0 (for example, DeepseekVLV2TextConfig).
        if (
            head_dim := getattr(self.hf_text_config, "head_dim", None)
        ) is not None and head_dim > 0:
            return head_dim

        # NOTE: Some models (such as PLaMo2.1) use `hidden_size_per_head`
        if getattr(self.hf_text_config, "hidden_size_per_head", None) is not None:
            return self.hf_text_config.hidden_size_per_head

        if (total_num_attention_heads := self.get_total_num_attention_heads()) == 0:
            return 0
        # FIXME(woosuk): This may not be true for all models.
        return self.get_hidden_size() // total_num_attention_heads

    def _get_qk_rope_head_dim(self) -> int:
        """Get qk_rope_head_dim, fixing the transformers v5.4+ attribute_map bug."""
        cfg = self.hf_text_config
        qk_rope_head_dim = getattr(cfg, "qk_rope_head_dim", 0)
        qk_nope_head_dim = getattr(cfg, "qk_nope_head_dim", 0)

        # In valid MLA configs, qk_rope_head_dim != qk_nope_head_dim.
        if qk_rope_head_dim == 0 or qk_rope_head_dim != qk_nope_head_dim:
            return qk_rope_head_dim  # not corrupted

        # Read the correct value from raw config.json.
        from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

        model_path = self.hf_config.name_or_path
        if not model_path:
            return qk_rope_head_dim
        raw = get_hf_file_to_dict("config.json", model_path, self.revision)
        if raw and "qk_rope_head_dim" in raw:
            correct = raw["qk_rope_head_dim"]
            if correct != qk_rope_head_dim:
                logger.info(
                    "Fixing qk_rope_head_dim: %d -> %d "
                    "(transformers v5.4+ attribute_map bug)",
                    qk_rope_head_dim,
                    correct,
                )
                # Patch the config so downstream model layers also get
                # the correct value.
                cfg.qk_rope_head_dim = correct
                return correct
        return qk_rope_head_dim

    def get_total_num_kv_heads(self) -> int:
        attributes = [
            # For Falcon:
            "n_head_kv",
            "num_kv_heads",
            # For LLaMA-2:
            "num_key_value_heads",
            # For ChatGLM:
            "multi_query_group_num",
            # For Step3p5:
            "num_attention_groups",
        ]
        # For non-grouped-query attention models, the number of KV heads is
        # equal to the number of attention heads.
        default_factory = self.get_total_num_attention_heads
        return getattr_iter(
            self.hf_text_config, attributes, default_factory=default_factory
        )

    def get_num_experts_from_block_configs(self) -> int:
        """Check block_configs for heterogeneous models (e.g., NemotronH).

        For heterogeneous models with varying expert counts per layer,
        returns the MAX to ensure all expert weights can be loaded.
        """
        max_experts = 0
        block_configs = getattr(self.hf_text_config, "block_configs", None)
        if block_configs:
            for block in block_configs:
                if isinstance(block, dict):
                    if block.get("block_type", "") == "moe":
                        max_experts = max(max_experts, block.get("n_routed_experts", 0))
                else:
                    if getattr(block, "block_type", "") == "moe":
                        max_experts = max(
                            max_experts, getattr(block, "n_routed_experts", 0)
                        )
        return max_experts

    def get_num_experts(self) -> int:
        """Returns the number of experts in the model."""
        num_expert_names = [
            "num_experts",  # Jamba
            "moe_num_experts",  # Dbrx
            "n_routed_experts",  # DeepSeek
            "num_local_experts",  # Mixtral
        ]

        num_experts = getattr_iter(self.hf_text_config, num_expert_names, 0)
        if isinstance(num_experts, list):
            # Ernie VL's remote code uses list[int]...
            # The values are always the same so we just take the first one.
            return num_experts[0]

        if not num_experts:
            num_experts = self.get_num_experts_from_block_configs()
        return num_experts

    def get_num_experts_per_token(self) -> int:
        names = [
            "num_experts_per_tok",
            "num_experts_per_token",
            "top_k_experts",
            "moe_topk",
            "moe_top_k",
        ]
        num_experts_per_token = getattr_iter(self.hf_text_config, names, 0)
        if isinstance(num_experts_per_token, list):
            return max(num_experts_per_token, default=0)
        return num_experts_per_token or 0

    @final
    @classmethod
    def get_torch_dtype(
        cls,
        hf_config: PreTrainedConfig,
        model_id: str,
        revision: str | None,
        config_format: str | ConfigFormat,
    ):
        # NOTE: getattr(config, "dtype", torch.float32) is not correct
        # because config.dtype can be None.
        config_dtype = getattr(hf_config, "dtype", None)

        # Fallbacks for multi-modal models if the root config
        # does not define dtype
        if config_dtype is None:
            config_dtype = getattr(hf_config.get_text_config(), "dtype", None)
        if config_dtype is None and hasattr(hf_config, "vision_config"):
            config_dtype = getattr(hf_config.vision_config, "dtype", None)
        if config_dtype is None and hasattr(hf_config, "encoder_config"):
            config_dtype = getattr(hf_config.encoder_config, "dtype", None)

        # Try to read the dtype of the weights if they are in safetensors format
        if config_dtype is None:
            param_mt = get_safetensors_params_metadata(model_id, revision=revision)

            if param_mt:
                param_dtypes: set[torch.dtype] = {
                    _SAFETENSORS_TO_TORCH_DTYPE[dtype]
                    for info in param_mt.values()
                    if (dtype := info.get("dtype", None))
                    and dtype in _SAFETENSORS_TO_TORCH_DTYPE
                }

                if param_dtypes:
                    return common_broadcastable_dtype(param_dtypes)

        if config_dtype is None:
            config_dtype = torch.float32

        return config_dtype

    def _normalize_quantization_config(self, config: PreTrainedConfig):
        quant_cfg = getattr(config, "quantization_config", None)
        if quant_cfg is None:
            # compressed-tensors uses a "compression_config" key
            quant_cfg = getattr(config, "compression_config", None)

        else:
            # Set quant_method for ModelOpt models.
            producer_name = quant_cfg.get("producer", {}).get("name")
            modelopt_quant_cfg = quant_cfg.get("quantization", {})
            is_legacy_modelopt = (
                isinstance(modelopt_quant_cfg, dict)
                and "modelopt_quant_config" in modelopt_quant_cfg
            )
            if producer_name == "modelopt" or is_legacy_modelopt:
                quant_algo = modelopt_quant_cfg.get("quant_algo")
                if quant_algo is not None:
                    quant_algo_upper = str(quant_algo).upper()
                    if quant_algo_upper in {
                        "FP8",
                        "FP8_PER_CHANNEL_PER_TOKEN",
                        "FP8_PB_WO",
                    }:
                        quant_cfg["quant_method"] = "modelopt"
                    elif quant_algo_upper == "NVFP4":
                        quant_cfg["quant_method"] = "modelopt_fp4"
                    else:
                        raise ValueError(f"Unknown ModelOpt quant algo: {quant_algo}")

        if quant_cfg is not None:
            # Use the community standard 'quant_method'
            quant_method = quant_cfg.get("quant_method", "").lower()

            # Normalize library names
            quant_method = quant_method.replace(
                "compressed_tensors", "compressed-tensors"
            )

            quant_cfg["quant_method"] = quant_method

        return quant_cfg

    def get_quantization_config(self):
        quant_cfg = self._normalize_quantization_config(self.hf_config)
        if quant_cfg is None and (
            text_config := getattr(self.hf_config, "text_config", None)
        ):
            # Check the text config as well for multi-modal models.
            quant_cfg = self._normalize_quantization_config(text_config)
        return quant_cfg

    def is_deepseek_mla(self) -> bool:
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        elif self.hf_text_config.model_type in (
            "axk1",
            "deepseek_v2",
            "deepseek_v3",
            "deepseek_v32",
            "deepseek_v4",
            "deepseek_v41",
            "dots3_note",
            "deepseek_mtp",
            "k3_dspark",
            "glm5_next",
            "glm5_next_text",
            "glm_moe_dsa",
            "glm4_moe_lite",
            "glm4_moe_lite_mtp",
            "kimi_k2",
            "kimi_linear",
            "longcat_flash",
            "pangu_ultra_moe",
            "pangu_ultra_moe_mtp",
            "bailing_hybrid",
            "bailing_hybrid_mtp",
            "bailing_hybrid_v3_mtp",
            "hy_v4",
            "hy_v4_mtp",
        ):
            # check is deepseek_v4 model
            if hasattr(self.hf_text_config, "compress_ratios"):
                return getattr(self.hf_text_config, "head_dim", None) is not None
            else:
                return getattr(self.hf_text_config, "kv_lora_rank", None) is not None
        elif self.hf_text_config.model_type == "eagle":
            # if the model is an EAGLE module, check for the
            # underlying architecture
            return (
                self.hf_text_config.model.model_type
                in (
                    "axk1",
                    "deepseek_v2",
                    "deepseek_v3",
                    "deepseek_v32",
                    "deepseek_mtp",
                )
                and getattr(self.hf_text_config, "kv_lora_rank", None) is not None
            )
        return False

    def is_mm_prefix_lm(self, supports_multimodal: bool = True) -> bool:
        """Whether to use bidirectional attention for mm positions.

        ``supports_multimodal`` is False when the deployment is configuration-
        disabled for multimodal inputs (text-only serving). In that case
        mm_prefix is unnecessary and must stay off so attention backends
        without ``supports_mm_prefix()`` remain eligible.
        """
        if not supports_multimodal:
            return False
        if hasattr(self.hf_config, "is_mm_prefix_lm"):
            return bool(self.hf_config.is_mm_prefix_lm)
        # fallback to list of known models
        MM_PREFIX_LM_MODELS = (
            "bagel",
            "gemma3",
            "molmo2",
            "moondream3",
            "paligemma",
        )
        if not hasattr(self.hf_config, "model_type"):
            return False
        return self.hf_config.model_type in MM_PREFIX_LM_MODELS

    def rswa_window(self) -> int | None:
        value = getattr(self.hf_config, "rswa_window", None)
        if value is None:
            return None
        return int(value)

    def derive_max_model_len_and_key(self) -> tuple[float, str | None]:
        derived_max_model_len = float("inf")
        possible_keys = [
            # OPT
            "max_position_embeddings",
            # GPT-2
            "n_positions",
            # MPT
            "max_seq_len",
            # ChatGLM2
            "seq_length",
            # Command-R
            "model_max_length",
            # Whisper
            "max_target_positions",
            # Others
            "max_sequence_length",
            "max_seq_length",
            "seq_len",
        ]
        # Choose the smallest "max_length" from the possible keys
        max_len_key = None
        for key in possible_keys:
            max_len = getattr(self.hf_text_config, key, None)
            if max_len is not None:
                if max_len < derived_max_model_len:
                    max_len_key = key
                derived_max_model_len = min(derived_max_model_len, max_len)

        # For Command-R / Cohere, Cohere2 models
        if tmp_max_len := getattr(self.hf_text_config, "model_max_length", None):
            max_len_key = "model_max_length"
            derived_max_model_len = tmp_max_len
        return derived_max_model_len, max_len_key

    def convert(self, supports_multimodal: bool = True) -> ModelArchitectureConfig:
        if (per_layer := self.get_per_layer_hf_configs()) is None:
            return self.convert_layer(supports_multimodal)

        if self.is_deepseek_mla():
            raise NotImplementedError(
                "Heterogeneous MLA models are not supported: `get_head_size` "
                "patches the config in place for them, which a per-layer "
                "conversion would apply to a throwaway copy."
            )
        # Convert each layer and let the result work out which fields differ.
        # Reading a varying attribute off the global config would raise, so the
        # whole-model config has to be built up from the layers.
        return ModelArchitectureConfig.from_layers(
            [
                type(self)(*configs).convert_layer(supports_multimodal)
                for configs in per_layer
            ]
        )

    def convert_layer(
        self, supports_multimodal: bool = True
    ) -> ModelArchitectureConfig:
        """Convert one homogeneous config, without resolving per-layer values.

        `convert` calls this once per layer, so it must not recurse back into
        per-layer resolution: the configs it receives are already layer-specific,
        and they can still carry the attributes that made the model look
        heterogeneous in the first place.
        """
        model_arch_config = ModelArchitectureConfig(
            architectures=self.get_architectures(),
            model_type=self.hf_config.model_type,
            text_model_type=getattr(self.hf_text_config, "model_type", None),
            hidden_size=self.get_hidden_size(),
            total_num_hidden_layers=self.get_num_hidden_layers(),
            total_num_attention_heads=self.get_total_num_attention_heads(),
            head_size=self.get_head_size(),
            vocab_size=self.get_vocab_size(),
            total_num_kv_heads=self.get_total_num_kv_heads(),
            num_experts=self.get_num_experts(),
            num_experts_per_token=self.get_num_experts_per_token(),
            quantization_config=self.get_quantization_config(),
            is_deepseek_mla=self.is_deepseek_mla(),
            is_mm_prefix_lm=self.is_mm_prefix_lm(supports_multimodal),
            rswa_window=self.rswa_window(),
            derived_max_model_len_and_key=self.derive_max_model_len_and_key(),
        )

        return model_arch_config


class DeepSeekMTPModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_num_hidden_layers(self) -> int:
        return getattr(self.hf_text_config, "num_nextn_predict_layers", 0)


# V4.1 uses the base convertor; bundled DSpark initialization temporarily sets
# model_type="deepseek_mtp" before selecting DSparkV41DraftModel.
MODEL_ARCH_CONFIG_CONVERTORS = {
    "deepseek_mtp": DeepSeekMTPModelArchConfigConvertor,
}
