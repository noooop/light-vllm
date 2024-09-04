import enum
import json
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, ClassVar, List, Optional, Tuple, Type, Union

import torch
from transformers import PretrainedConfig

import light_vllm.envs as envs
from light_vllm.logger import init_logger
from light_vllm.layers.quantization import QUANTIZATION_METHODS
from light_vllm.models.transformers_utils.config import get_config, get_hf_text_config
from light_vllm.utils import (cuda_device_count_stateless, get_cpu_memory, is_cpu,
                              is_hip, is_neuron, is_openvino, is_tpu, is_xpu,
                              print_warning_once)
from light_vllm.task.base.config import EngineConfig, CacheConfig, ModelConfig

logger = init_logger(__name__)

_GB = 1 << 30

_PP_SUPPORTED_MODELS = [
    "AquilaModel",
    "AquilaForCausalLM",
    "DeepseekV2ForCausalLM",
    "InternLMForCausalLM",
    "LlamaForCausalLM",
    "LLaMAForCausalLM",
    "MistralForCausalLM",
    "Phi3ForCausalLM",
    "GPT2LMHeadModel",
    "MixtralForCausalLM",
    "NemotronForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2MoeForCausalLM",
    "QWenLMHeadModel",
]


class ChatModelConfig(ModelConfig):
    """Configuration for the model.

    Args:
        model: Name or path of the huggingface model to use.
            It is also used as the content for `model_name` tag in metrics 
            output when `served_model_name` is not specified. 
        tokenizer: Name or path of the huggingface tokenizer to use.
        tokenizer_mode: Tokenizer mode. "auto" will use the fast tokenizer if
            available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        dtype: Data type for model weights and activations. The "auto" option
            will use FP16 precision for FP32 and FP16 models, and BF16 precision
            for BF16 models.
        seed: Random seed for reproducibility.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id. If unspecified, will use the default
            version.
        code_revision: The specific revision to use for the model code on
            Hugging Face Hub. It can be a branch name, a tag name, or a
            commit id. If unspecified, will use the default version.
        rope_scaling: Dictionary containing the scaling configuration for the
            RoPE embeddings. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id. If unspecified, will use
            the default version.
        max_model_len: Maximum length of a sequence (including prompt and
            output). If None, will be derived from the model.
        quantization: Quantization method that was used to quantize the model
            weights. If None, we assume the model weights are not quantized.
        quantization_param_path: Path to JSON file containing scaling factors.
            Used to load KV cache scaling factors into the model when KV cache
            type is FP8_E4M3 on ROCm (AMD GPU). In the future these will also
            be used to load activation and weight scaling factors when the
            model dtype is FP8_E4M3 on ROCm.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode (DEPRECATED. Use max_seq_len_to_capture instead).
        max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode
        disable_sliding_window: Whether to disable sliding window. If True,
            we will disable the sliding window functionality of the model.
            If the model does not support sliding window, this argument is
            ignored.
        skip_tokenizer_init: If true, skip initialization of tokenizer and
            detokenizer.
        served_model_name: The model name used in metrics tag `model_name`,
            matches the model name exposed via the APIs. If multiple model 
            names provided, the first name will be used. If not specified, 
            the model name will be the same as `model`.
    """

    def __init__(
            self,
            model: str,
            tokenizer: str,
            tokenizer_mode: str,
            trust_remote_code: bool,
            dtype: Union[str, torch.dtype],
            seed: int,
            revision: Optional[str] = None,
            code_revision: Optional[str] = None,
            rope_scaling: Optional[dict] = None,
            rope_theta: Optional[float] = None,
            tokenizer_revision: Optional[str] = None,
            max_model_len: Optional[int] = None,
            quantization: Optional[str] = None,
            quantization_param_path: Optional[str] = None,
            enforce_eager: bool = True,
            max_context_len_to_capture: Optional[int] = None,
            max_seq_len_to_capture: Optional[int] = None,
            max_logprobs: int = 20,
            disable_sliding_window: bool = False,
            skip_tokenizer_init: bool = False,
            served_model_name: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(
            model,
            tokenizer,
            tokenizer_mode,
            trust_remote_code,
            dtype,
            seed,
            revision,
            code_revision,
            tokenizer_revision,
            quantization,
            quantization_param_path,
            enforce_eager,
            skip_tokenizer_init,
            served_model_name)

        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta

        self.enforce_eager = enforce_eager
        if max_context_len_to_capture is not None:
            raise ValueError("`max_context_len_to_capture` is deprecated. "
                             "Use `max_seq_len_to_capture` instead.")
        self.max_seq_len_to_capture = max_seq_len_to_capture
        self.max_logprobs = max_logprobs
        self.disable_sliding_window = disable_sliding_window
        self.skip_tokenizer_init = skip_tokenizer_init

        for key, value in [("rope_scaling", rope_scaling),
                           ("rope_theta", rope_theta)]:
            if value is not None:
                logger.info("Updating %s from %r to %r", key,
                            getattr(self.hf_config, key, None), value)
                self.hf_config.update({key: value})

        if (not self.disable_sliding_window
                and self.hf_text_config.model_type == "gemma2"
                and self.hf_text_config.sliding_window is not None):
            print_warning_once(
                "Gemma 2 uses sliding window attention for every odd layer, "
                "which is currently not supported by vLLM. Disabling sliding "
                "window and capping the max length to the sliding window size "
                f"({self.hf_text_config.sliding_window}).")
            self.disable_sliding_window = True

        self.max_model_len = _get_and_verify_max_len(
            hf_config=self.hf_text_config,
            max_model_len=max_model_len,
            disable_sliding_window=self.disable_sliding_window,
            sliding_window_len=self.get_hf_config_sliding_window())

        self._verify_cuda_graph()

    def _verify_cuda_graph(self) -> None:
        if self.max_seq_len_to_capture is None:
            self.max_seq_len_to_capture = self.max_model_len
        self.max_seq_len_to_capture = min(self.max_seq_len_to_capture,
                                          self.max_model_len)

    def get_hf_config_sliding_window(self) -> Optional[int]:
        """Get the sliding window size, or None if disabled."""

        # Some models, like Qwen2 and Qwen1.5, use `use_sliding_window` in
        # addition to sliding window size. We check if that field is present
        # and if it's False, return None.
        if (hasattr(self.hf_text_config, "use_sliding_window")
                and not self.hf_text_config.use_sliding_window):
            return None
        return getattr(self.hf_text_config, "sliding_window", None)

    def get_sliding_window(self) -> Optional[int]:
        """Get the sliding window size, or None if disabled.
        """
        # If user disables sliding window, return None.
        if self.disable_sliding_window:
            return None
        # Otherwise get the value from the hf config.
        return self.get_hf_config_sliding_window()


def _get_and_verify_max_len(
    hf_config: PretrainedConfig,
    max_model_len: Optional[int],
    disable_sliding_window: bool,
    sliding_window_len: Optional[int],
) -> int:
    """Get and verify the model's maximum length."""
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
        # Others
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    # Choose the smallest "max_length" from the possible keys.
    max_len_key = None
    for key in possible_keys:
        max_len = getattr(hf_config, key, None)
        if max_len is not None:
            max_len_key = key if max_len < derived_max_model_len \
                else max_len_key
            derived_max_model_len = min(derived_max_model_len, max_len)

    # If sliding window is manually disabled, max_length should be less
    # than the sliding window length in the model config.
    if disable_sliding_window and sliding_window_len is not None:
        max_len_key = "sliding_window" \
            if sliding_window_len < derived_max_model_len else max_len_key
        derived_max_model_len = min(derived_max_model_len, sliding_window_len)

    # If none of the keys were found in the config, use a default and
    # log a warning.
    if derived_max_model_len == float("inf"):
        if max_model_len is not None:
            # If max_model_len is specified, we use it.
            return max_model_len

        default_max_len = 2048
        logger.warning(
            "The model's config.json does not contain any of the following "
            "keys to determine the original maximum length of the model: "
            "%s. Assuming the model's maximum length is %d.", possible_keys,
            default_max_len)
        derived_max_model_len = default_max_len

    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if rope_scaling is not None:
        if "type" in rope_scaling:
            rope_type = rope_scaling["type"]
        elif "rope_type" in rope_scaling:
            rope_type = rope_scaling["rope_type"]
        else:
            raise ValueError(
                "rope_scaling must have a 'type' or 'rope_type' key.")

        # The correct one should be "longrope", kept "su" here
        # to be backward compatible
        if rope_type not in ("su", "longrope", "llama3"):
            if disable_sliding_window:
                # TODO(robertgshaw): Find a model that supports rope_scaling
                # with sliding window to see if this case should be allowed.
                raise NotImplementedError(
                    "Disabling sliding window is not supported for models "
                    "with rope_scaling. Please raise an issue so we can "
                    "investigate.")

            assert "factor" in rope_scaling
            scaling_factor = rope_scaling["factor"]
            if rope_type == "yarn":
                derived_max_model_len = rope_scaling[
                    "original_max_position_embeddings"]
            derived_max_model_len *= scaling_factor

    # If the user specified a max length, make sure it is smaller than the
    # derived length from the HF model config.
    if max_model_len is None:
        max_model_len = int(derived_max_model_len)
    elif max_model_len > derived_max_model_len:
        # Some models might have a separate key for specifying model_max_length
        # that will be bigger than derived_max_model_len. We compare user input
        # with model_max_length and allow this override when it's smaller.
        model_max_length = getattr(hf_config, "model_max_length", None)
        if model_max_length is not None and max_model_len <= model_max_length:
            if disable_sliding_window:
                # TODO(robertgshaw): Find a model that has model_max_length
                # with sliding window to see if this case should be allowed.
                raise NotImplementedError(
                    "Disabling sliding window is not supported for models "
                    "model_max_length in the config. Please raise an issue "
                    "so we can investigate.")
        else:
            msg = (
                f"User-specified max_model_len ({max_model_len}) is greater "
                f"than the derived max_model_len ({max_len_key}="
                f"{derived_max_model_len} or model_max_length="
                f"{model_max_length} in model's config.json). This may lead "
                "to incorrect model outputs or CUDA errors.")
            if envs.VLLM_ALLOW_LONG_MAX_MODEL_LEN:
                logger.warning(
                    "%s Make sure the value is correct and within the "
                    "model context size.", msg)
            else:
                raise ValueError(
                    f"{msg} To allow overriding this maximum, set "
                    "the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN=1")
    return int(max_model_len)


class SchedulerConfig:
    """Scheduler configuration.

    Args:
        max_num_batched_tokens: Maximum number of tokens to be processed in
            a single iteration.
        max_num_seqs: Maximum number of sequences to be processed in a single
            iteration.
        max_model_len: Maximum length of a sequence (including prompt
            and generated text).
        use_v2_block_manager: Whether to use the BlockSpaceManagerV2 or not.
        num_lookahead_slots: The number of slots to allocate per sequence per
            step, beyond the known token ids. This is used in speculative
            decoding to store KV activations of tokens which may or may not be
            accepted.
        delay_factor: Apply a delay (of delay factor multiplied by previous
            prompt latency) before scheduling next prompt.
        enable_chunked_prefill: If True, prefill requests can be chunked based
            on the remaining max_num_batched_tokens.
        preemption_mode: Whether to perform preemption by swapping or 
            recomputation. If not specified, we determine the mode as follows:
            We use recomputation by default since it incurs lower overhead than
            swapping. However, when the sequence group has multiple sequences
            (e.g., beam search), recomputation is not currently supported. In
            such a case, we use swapping instead.
    """

    def __init__(self,
                 max_num_batched_tokens: Optional[int],
                 max_num_seqs: int,
                 max_model_len: int,
                 use_v2_block_manager: bool = False,
                 num_lookahead_slots: int = 0,
                 delay_factor: float = 0.0,
                 enable_chunked_prefill: bool = False,
                 preemption_mode: Optional[str] = None) -> None:
        if max_num_batched_tokens is not None:
            self.max_num_batched_tokens = max_num_batched_tokens
        else:
            if enable_chunked_prefill:
                # It is the values that have the best balance between ITL
                # and TTFT on A100. Note it is not optimized for throughput.
                self.max_num_batched_tokens = 512
            else:
                # If max_model_len is too short, use 2048 as the default value
                # for higher throughput.
                self.max_num_batched_tokens = max(max_model_len, 2048)
        if enable_chunked_prefill:
            logger.info(
                "Chunked prefill is enabled with max_num_batched_tokens=%d.",
                self.max_num_batched_tokens)

        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.use_v2_block_manager = use_v2_block_manager
        self.num_lookahead_slots = num_lookahead_slots
        self.delay_factor = delay_factor
        self.chunked_prefill_enabled = enable_chunked_prefill
        self.preemption_mode = preemption_mode
        self._verify_args()

    def _verify_args(self) -> None:
        if (self.max_num_batched_tokens < self.max_model_len
                and not self.chunked_prefill_enabled):
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({self.max_model_len}). "
                "This effectively limits the maximum sequence length to "
                "max_num_batched_tokens and makes vLLM reject longer "
                "sequences. Please increase max_num_batched_tokens or "
                "decrease max_model_len.")

        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_num_seqs "
                f"({self.max_num_seqs}).")

        if self.num_lookahead_slots < 0:
            raise ValueError(
                "num_lookahead_slots "
                f"({self.num_lookahead_slots}) must be greater than or "
                "equal to 0.")


@dataclass(frozen=True)
class ChatEngineConfig(EngineConfig):
    """Dataclass which contains all engine-related configuration. This
    simplifies passing around the distinct configurations in the codebase.
    """

    model_config: ChatModelConfig
    cache_config: Optional[CacheConfig]
    scheduler_config: SchedulerConfig

    def to_dict(self):
        """Return the configs as a dictionary, for use in **kwargs.
        """
        return dict(
            (field.name, getattr(self, field.name)) for field in fields(self))
