from dataclasses import dataclass
from typing import List, Optional, Union

from light_vllm.logger import init_logger
from light_vllm.wde.core.arg_utils import EngineArgs
from light_vllm.wde.core.config import (DeviceConfig, LoadConfig,
                                        filter_unexpected_fields)
from light_vllm.wde.decode_only.config import (
    DecodeOnlyEmbeddingSchedulerConfig, DecodeOnlyEngineConfig,
    DecodeOnlyModelConfig, DecodeOnlySchedulerConfig,
    PrefillOnlyParallelConfig)

logger = init_logger(__name__)


def nullable_str(val: str):
    if not val or val == "None":
        return None
    return val


@filter_unexpected_fields
@dataclass
class DecodeOnlyEngineArgs(EngineArgs):
    """Arguments for vLLM engine."""
    model: str
    served_model_name: Optional[Union[List[str]]] = None
    tokenizer: Optional[str] = None
    skip_tokenizer_init: bool = False
    tokenizer_mode: str = 'auto'
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    load_format: str = 'auto'
    dtype: str = 'auto'
    kv_cache_dtype: str = 'auto'
    quantization_param_path: Optional[str] = None
    disable_sliding_window: bool = False
    seed: int = 0
    max_model_len: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None

    output_last_hidden_states: bool = False
    enable_bidirectional: bool = False

    max_num_seqs: int = 256
    max_num_on_the_fly: int = 3
    scheduling: str = "async"

    data_parallel_size: int = 0

    disable_log_stats: bool = False
    revision: Optional[str] = None
    code_revision: Optional[str] = None
    rope_scaling: Optional[dict] = None
    rope_theta: Optional[float] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None
    disable_custom_all_reduce: bool = False
    device: str = 'auto'
    model_loader_extra_config: Optional[dict] = None
    ignore_patterns: Optional[Union[str, List[str]]] = None

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model

    def create_engine_config(self) -> DecodeOnlyEngineConfig:
        device_config = DeviceConfig(device=self.device)
        model_config = DecodeOnlyModelConfig(
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
            disable_sliding_window=self.disable_sliding_window,
            skip_tokenizer_init=self.skip_tokenizer_init,
            served_model_name=self.served_model_name,
            output_last_hidden_states=self.output_last_hidden_states,
            enable_bidirectional=self.enable_bidirectional)

        if model_config.output_last_hidden_states:
            scheduler_config = DecodeOnlyEmbeddingSchedulerConfig(
                max_num_batched_tokens=self.max_num_batched_tokens,
                max_num_seqs=self.max_num_seqs,
                max_model_len=model_config.max_model_len,
                max_num_on_the_fly=self.max_num_on_the_fly,
                scheduling=self.scheduling)
        else:
            scheduler_config = DecodeOnlySchedulerConfig()

        if (model_config.output_last_hidden_states
                and self.data_parallel_size > 0):
            parallel_config = PrefillOnlyParallelConfig(
                data_parallel_size=self.data_parallel_size)
        else:
            parallel_config = None

        load_config = LoadConfig(
            load_format=self.load_format,
            download_dir=self.download_dir,
            model_loader_extra_config=self.model_loader_extra_config,
            ignore_patterns=self.ignore_patterns,
        )

        return DecodeOnlyEngineConfig(model_config=model_config,
                                      scheduler_config=scheduler_config,
                                      device_config=device_config,
                                      load_config=load_config,
                                      parallel_config=parallel_config)
