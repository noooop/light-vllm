from dataclasses import dataclass
from typing import List, Optional, Union

from light_vllm.core.arg_utils import EngineArgs
from light_vllm.core.config import DeviceConfig, LoadConfig
from light_vllm.decoding.config import (CacheConfig, ChatEngineConfig,
                                        ChatModelConfig, EngineConfig,
                                        SchedulerConfig)
from light_vllm.logger import init_logger

logger = init_logger(__name__)


def nullable_str(val: str):
    if not val or val == "None":
        return None
    return val


@dataclass
class ChatEngineArgs(EngineArgs):
    """Arguments for vLLM chat engine."""
    kv_cache_dtype: str = 'auto'
    quantization_param_path: Optional[str] = None

    max_model_len: Optional[int] = None

    block_size: int = 16
    enable_prefix_caching: bool = False
    disable_sliding_window: bool = False
    use_v2_block_manager: bool = False
    swap_space: int = 4  # GiB
    cpu_offload_gb: int = 0  # GiB
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    max_logprobs: int = 20  # Default value for OpenAI Chat Completions API

    max_num_on_the_fly: Optional[int] = None
    scheduling: str = "simple_async"

    revision: Optional[str] = None
    code_revision: Optional[str] = None
    rope_scaling: Optional[dict] = None
    rope_theta: Optional[float] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None

    device: str = 'auto'
    num_gpu_blocks_override: Optional[int] = None
    model_loader_extra_config: Optional[dict] = None
    ignore_patterns: Optional[Union[str, List[str]]] = None
    preemption_mode: Optional[str] = None

    scheduler_delay_factor: float = 0.0
    enable_chunked_prefill: Optional[bool] = None

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model

    def create_engine_config(self, ) -> EngineConfig:
        assert self.cpu_offload_gb >= 0, (
            "CPU offload space must be non-negative"
            f", but got {self.cpu_offload_gb}")

        device_config = DeviceConfig(device=self.device)
        model_config = ChatModelConfig(
            model=self.model,
            tokenizer=self.tokenizer,
            tokenizer_mode=self.tokenizer_mode,
            trust_remote_code=self.trust_remote_code,
            dtype=self.dtype,
            seed=self.seed,
            revision=self.revision,
            code_revision=self.code_revision,
            rope_scaling=self.rope_scaling,
            rope_theta=self.rope_theta,
            tokenizer_revision=self.tokenizer_revision,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            quantization_param_path=self.quantization_param_path,
            max_logprobs=self.max_logprobs,
            disable_sliding_window=self.disable_sliding_window,
            skip_tokenizer_init=self.skip_tokenizer_init,
            served_model_name=self.served_model_name)
        cache_config = CacheConfig(
            block_size=self.block_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            swap_space=self.swap_space,
            cache_dtype=self.kv_cache_dtype,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            sliding_window=model_config.get_sliding_window(),
            enable_prefix_caching=self.enable_prefix_caching,
            cpu_offload_gb=self.cpu_offload_gb,
        )

        max_model_len = model_config.max_model_len
        use_long_context = max_model_len > 32768
        if self.enable_chunked_prefill is None:
            # If not explicitly set, enable chunked prefill by default for
            # long context (> 32K) models. This is to avoid OOM errors in the
            # initial memory profiling phase.
            if use_long_context:
                self.enable_chunked_prefill = True
                logger.warning(
                    "Chunked prefill is enabled by default for models with "
                    "max_model_len > 32K. Currently, chunked prefill might "
                    "not work with some features or models. If you "
                    "encounter any issues, please disable chunked prefill "
                    "by setting --enable-chunked-prefill=False.")
            if self.enable_chunked_prefill is None:
                self.enable_chunked_prefill = False

        if not self.enable_chunked_prefill and use_long_context:
            logger.warning(
                "The model has a long context length (%s). This may cause OOM "
                "errors during the initial memory profiling phase, or result "
                "in low performance due to small KV cache space. Consider "
                "setting --max-model-len to a smaller value.", max_model_len)

        scheduler_config = SchedulerConfig(
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            max_model_len=model_config.max_model_len,
            use_v2_block_manager=self.use_v2_block_manager,
            delay_factor=self.scheduler_delay_factor,
            enable_chunked_prefill=self.enable_chunked_prefill,
            preemption_mode=self.preemption_mode,
            scheduling=self.scheduling)

        load_config = LoadConfig(
            load_format=self.load_format,
            download_dir=self.download_dir,
            model_loader_extra_config=self.model_loader_extra_config,
            ignore_patterns=self.ignore_patterns,
        )

        if (model_config.get_sliding_window() is not None
                and scheduler_config.chunked_prefill_enabled
                and not scheduler_config.use_v2_block_manager):
            raise ValueError(
                "Chunked prefill is not supported with sliding window. "
                "Set --disable-sliding-window to disable sliding window.")

        return ChatEngineConfig(model_config=model_config,
                                cache_config=cache_config,
                                scheduler_config=scheduler_config,
                                device_config=device_config,
                                load_config=load_config,
                                parallel_config=None)
