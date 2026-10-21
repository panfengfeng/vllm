# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise config dispatch with real V4/V4.1 configs and stubbed HF I/O.

These CPU tests load the complete production config module. They do not
validate Transformers validators, remote repositories or model execution.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "vllm/transformers_utils/config.py"


@pytest.fixture
def hf_config(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM_E8M0", "0")

    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    class PreTrainedConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        @classmethod
        def get_config_dict(cls, model, **kwargs):
            return json.loads((Path(model) / "config.json").read_text()), kwargs

        def update(self, values):
            self.__dict__.update(values)

        def get_text_config(self):
            return self

        def standardize_rope_params(self):
            pass

        def validate_rope(self):
            pass

    registered = {}
    calls = []

    class AutoConfig:
        @staticmethod
        def register(model_type, config_class, **kwargs):
            registered[model_type] = config_class

        @staticmethod
        def from_pretrained(model, **kwargs):
            calls.append(kwargs)
            values, _ = PreTrainedConfig.get_config_dict(model)
            return registered[values["model_type"]](**values)

    class DraftConfig(PreTrainedConfig):
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            values, _ = cls.get_config_dict(model)
            return cls(**values)

    no_op = lambda *a, **kw: None
    stub("torch")
    constants = stub("huggingface_hub.constants", HF_HUB_OFFLINE=True)
    stub("huggingface_hub", constants=constants)
    stub("safetensors", __path__=[])
    stub("safetensors.torch", _TYPES={})
    stub(
        "transformers",
        __path__=[],
        AutoConfig=AutoConfig,
        GenerationConfig=object,
        PreTrainedConfig=PreTrainedConfig,
    )
    stub("transformers.configuration_utils", ALLOWED_LAYER_TYPES=set())
    stub(
        "transformers.models.auto.image_processing_auto",
        get_image_processor_config=no_op,
    )
    stub("transformers.models.auto.modeling_auto", MODEL_MAPPING_NAMES={})
    stub("transformers.models.auto.tokenization_auto", get_tokenizer_config=no_op)
    stub("transformers.utils", CONFIG_NAME="config.json")
    envs = SimpleNamespace(VLLM_USE_MODELSCOPE=False, is_set=lambda name: False)
    stub("vllm", __path__=[], envs=envs)
    logger = SimpleNamespace(
        **{
            name: no_op
            for name in ("debug", "info", "info_once", "warning", "warning_once")
        }
    )
    stub("vllm.logger", init_logger=lambda name: logger)
    stub("vllm.transformers_utils", __path__=[])
    stub("vllm.transformers_utils.config_parser_base", ConfigParserBase=object)
    stub(
        "vllm.transformers_utils.repo_utils",
        is_mistral_model_repo=lambda **kw: False,
        file_or_path_exists=lambda model, config_name, revision=None: (
            Path(model) / config_name
        ).is_file(),
        get_hf_file_to_dict=no_op,
        hf_api=no_op,
        list_repo_files=no_op,
        try_get_local_file=no_op,
        with_retry=lambda fn, *a, **kw: fn(),
    )
    stub(
        "vllm.transformers_utils.utils",
        parse_safetensors_file_metadata=no_op,
        without_trust_remote_code=lambda kw: {
            k: v for k, v in kw.items() if k != "trust_remote_code"
        },
    )
    stub("vllm.utils.torch_utils", common_broadcastable_dtype=no_op)
    stub(
        "vllm.config.utils",
        getattr_iter=lambda obj, names, default, **kw: next(
            (getattr(obj, name) for name in names if hasattr(obj, name)), default
        ),
    )
    configs = stub("vllm.transformers_utils.configs", __path__=[])
    for version in ("v4", "v41"):
        module = load(
            f"vllm.transformers_utils.configs.deepseek_{version}",
            ROOT / f"vllm/transformers_utils/configs/deepseek_{version}.py",
        )
        name = f"Deepseek{version.capitalize()}Config"
        setattr(configs, name, getattr(module, name))
    configs.EAGLEConfig = configs.SpeculatorsConfig = DraftConfig
    module = load("vllm.transformers_utils.config", CONFIG_PATH)

    def checkpoint(values):
        (tmp_path / "config.json").write_text(json.dumps(values))
        return tmp_path

    return SimpleNamespace(
        module=module, checkpoint=checkpoint, registered=registered, calls=calls
    )


@pytest.mark.parametrize("vision_layers", [0, 2])
def test_v41_checkpoint_flattens_text_and_vision_before_rope_patch(
    hf_config, vision_layers
):
    path = hf_config.checkpoint(
        {
            "model_type": "deepseek_v41",
            "architectures": ["DeepseekV41ForCausalLM"],
            "text_config": {
                "hidden_size": 128,
                "rope_scaling": {"type": "yarn", "factor": 4},
            },
            "vision_config": {"num_hidden_layers": vision_layers, "hidden_size": 64},
            "quantization_config": {
                "quant_method": "fp8",
                "expert_dtype": "fp4",
                "scale_fmt": "ue8m0",
            },
        }
    )
    config = hf_config.module.get_config(path, trust_remote_code=True)
    assert config.hidden_size == 128
    assert (config.vision_n_layers, config.vision_dim) == (vision_layers, 64)
    assert config.rope_parameters == {"type": "yarn", "rope_type": "yarn", "factor": 4}
    assert config.expert_dtype == "fp4"
    assert config.quantization_config["scale_fmt"] == "ue8m0"
    assert hf_config.module.os.environ["VLLM_USE_DEEP_GEMM_E8M0"] == "1"
    assert hf_config.registered["deepseek_v41"] is type(config)
    assert not hf_config.calls[0]["trust_remote_code"]


def test_dspark_callable_override_runs_after_native_v41_config_construction(hf_config):
    path = hf_config.checkpoint(
        {
            "model_type": "deepseek_v41",
            "architectures": ["DeepseekV41ForCausalLM"],
            "text_config": {"num_nextn_predict_layers": 3, "dspark_block_size": 5},
        }
    )

    def override(config):
        if config.model_type == "deepseek_v41":
            config.model_type = "deepseek_mtp"
            config.n_predict = config.num_nextn_predict_layers
        return config

    config = hf_config.module.get_config(
        path, trust_remote_code=False, hf_overrides_fn=override
    )
    assert config.model_type == "deepseek_mtp"
    assert (config.n_predict, config.dspark_block_size) == (3, 5)
    assert config.architectures == ["DeepseekV41ForCausalLM"]
    assert set(hf_config.registered) == {"deepseek_v41"}


@pytest.mark.parametrize("model_type", ["deepseek_v4", "eagle", "speculators"])
def test_shared_config_entries_keep_their_loading_path(hf_config, model_type):
    values = {"model_type": model_type, "architectures": ["test"]}
    if model_type == "speculators":
        values.pop("model_type")
        values["speculators_config"] = {}
    _, config = hf_config.module.HFConfigParser().parse(
        hf_config.checkpoint(values), trust_remote_code=False
    )
    assert config.architectures == ["test"]
    assert bool(hf_config.calls) == (model_type == "deepseek_v4")
