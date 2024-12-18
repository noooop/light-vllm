from dataclasses import dataclass, fields
from typing import List, Optional, Union

import torch

from light_vllm.core.config import CacheConfig, EngineConfig, ModelConfig
from light_vllm.logger import init_logger

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
        max_logprobs: int = 20,
        disable_sliding_window: bool = False,
        skip_tokenizer_init: bool = False,
        served_model_name: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(model, tokenizer, tokenizer_mode, trust_remote_code,
                         dtype, seed, revision, code_revision, rope_scaling,
                         rope_theta, tokenizer_revision, max_model_len,
                         quantization, quantization_param_path,
                         disable_sliding_window, skip_tokenizer_init,
                         served_model_name)
        self.max_logprobs = max_logprobs


class SchedulerConfig:
    supported_scheduling = ["sync", "simple_async", "async", "double_buffer"]

    def __init__(self,
                 max_num_batched_tokens: Optional[int],
                 max_num_seqs: int,
                 max_model_len: int,
                 use_v2_block_manager: bool = False,
                 num_lookahead_slots: int = 0,
                 delay_factor: float = 0.0,
                 enable_chunked_prefill: bool = False,
                 preemption_mode: Optional[str] = None,
                 max_num_on_the_fly: Optional[int] = None,
                 scheduling: str = "async") -> None:
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

        if max_num_on_the_fly is None:
            if scheduling == "double_buffer":
                self.max_num_on_the_fly = 3
            else:
                self.max_num_on_the_fly = 2
        else:
            self.max_num_on_the_fly = max_num_on_the_fly

        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.use_v2_block_manager = use_v2_block_manager
        self.num_lookahead_slots = num_lookahead_slots
        self.delay_factor = delay_factor
        self.chunked_prefill_enabled = enable_chunked_prefill
        self.preemption_mode = preemption_mode
        self.scheduling = scheduling
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

        if self.max_num_on_the_fly < 2:
            raise ValueError(
                f"max_num_on_the_fly {self.max_num_on_the_fly} must "
                "be greater than 1")


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

    def log_config(self):
        from light_vllm.version import __version__ as VLLM_VERSION
        logger.info(
            "Initializing an LLM engine (v%s) with config: "
            "model=%r, tokenizer=%r, "
            "skip_tokenizer_init=%s, tokenizer_mode=%s, revision=%s, "
            "rope_scaling=%r, rope_theta=%r, tokenizer_revision=%s, "
            "trust_remote_code=%s, dtype=%s, max_seq_len=%d, "
            "download_dir=%r, load_format=%s, "
            "quantization=%s, kv_cache_dtype=%s, "
            "quantization_param_path=%s, device_config=%s, "
            "seed=%d, served_model_name=%s, use_v2_block_manager=%s, "
            "enable_prefix_caching=%s, scheduling=%s)",
            VLLM_VERSION,
            self.model_config.model,
            self.model_config.tokenizer,
            self.model_config.skip_tokenizer_init,
            self.model_config.tokenizer_mode,
            self.model_config.revision,
            self.model_config.rope_scaling,
            self.model_config.rope_theta,
            self.model_config.tokenizer_revision,
            self.model_config.trust_remote_code,
            self.model_config.dtype,
            self.model_config.max_model_len,
            self.load_config.download_dir,
            self.load_config.load_format,
            self.model_config.quantization,
            self.cache_config.cache_dtype,
            self.model_config.quantization_param_path,
            self.device_config.device,
            self.model_config.seed,
            self.model_config.served_model_name,
            self.scheduler_config.use_v2_block_manager,
            self.cache_config.enable_prefix_caching,
            self.scheduler_config.scheduling,
        )
