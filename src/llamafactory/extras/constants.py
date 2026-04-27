# Copyright 2025 the LlamaFactory team.
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

import os
from collections import OrderedDict, defaultdict
from enum import StrEnum, unique

from peft.utils import SAFETENSORS_WEIGHTS_NAME as SAFE_ADAPTER_WEIGHTS_NAME
from peft.utils import WEIGHTS_NAME as ADAPTER_WEIGHTS_NAME
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME


AUDIO_PLACEHOLDER = os.getenv("AUDIO_PLACEHOLDER", "<audio>")

CHECKPOINT_NAMES = {
    SAFE_ADAPTER_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
}

CHOICES = ["A", "B", "C", "D"]

DATA_CONFIG = "dataset_info.json"

DEFAULT_TEMPLATE = defaultdict(str)

FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}

IGNORE_INDEX = -100

IMAGE_PLACEHOLDER = os.getenv("IMAGE_PLACEHOLDER", "<image>")

LAYERNORM_NAMES = {"norm", "ln"}

LLAMABOARD_CONFIG = "llamaboard_config.yaml"

MCA_SUPPORTED_MODELS = {
    "deepseek_v3",
    "glm4_moe",
    "llama",
    "mistral",
    "mixtral",
    "qwen2",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3",
    "qwen3_moe",
    "qwen3_next",
    "qwen3_5",
    "qwen3_5_moe",
}

METHODS = ["full", "freeze", "lora", "oft"]

MOD_SUPPORTED_MODELS = {"bloom", "falcon", "gemma", "llama", "mistral", "mixtral", "phi", "starcoder2"}

MROPE_MODELS = {
    "glm4v",
    "glm_ocr",
    "Keye",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen2_5_omni_thinker",
    "qwen3_omni_moe_thinker",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_5",
    "qwen3_5_moe",
}

MULTIMODAL_SUPPORTED_MODELS = set()

PEFT_METHODS = {"lora", "oft"}

RUNNING_LOG = "running_log.txt"

SUBJECTS = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]

SUPPORTED_MODELS = OrderedDict()

TRAINER_LOG = "trainer_log.jsonl"

TRAINING_ARGS = "training_args.yaml"

TRAINING_STAGES = {
    "Supervised Fine-Tuning": "sft",
    "Reward Modeling": "rm",
    "PPO": "ppo",
    "DPO": "dpo",
    "KTO": "kto",
    "Pre-Training": "pt",
}

STAGES_USE_PAIR_DATA = {"rm", "dpo"}

SUPPORTED_CLASS_FOR_S2ATTN = {"llama"}

SWANLAB_CONFIG = "swanlab_public_config.json"

VIDEO_PLACEHOLDER = os.getenv("VIDEO_PLACEHOLDER", "<video>")

V_HEAD_WEIGHTS_NAME = "value_head.bin"

V_HEAD_SAFE_WEIGHTS_NAME = "value_head.safetensors"


class AttentionFunction(StrEnum):
    AUTO = "auto"
    DISABLED = "disabled"
    SDPA = "sdpa"
    FA2 = "fa2"
    FA3 = "fa3"


class EngineName(StrEnum):
    HF = "huggingface"
    VLLM = "vllm"
    SGLANG = "sglang"
    KT = "ktransformers"


class DownloadSource(StrEnum):
    DEFAULT = "hf"
    MODELSCOPE = "ms"
    OPENMIND = "om"


@unique
class QuantizationMethod(StrEnum):
    r"""Borrowed from `transformers.utils.quantization_config.QuantizationMethod`."""

    BNB = "bnb"
    GPTQ = "gptq"
    AWQ = "awq"
    AQLM = "aqlm"
    QUANTO = "quanto"
    EETQ = "eetq"
    HQQ = "hqq"
    MXFP4 = "mxfp4"
    FP8 = "fp8"


class RopeScaling(StrEnum):
    LINEAR = "linear"
    DYNAMIC = "dynamic"
    YARN = "yarn"
    LLAMA3 = "llama3"



# Model registrations removed for blank-slate focus.
# Use --model_name_or_path to specify your custom model or configuration.
