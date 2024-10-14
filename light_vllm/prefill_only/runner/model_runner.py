import torch
import torch.nn as nn

from light_vllm.backends.attention import AttentionBackend
from light_vllm.core.config import DeviceConfig, LoadConfig, ModelConfig
from light_vllm.core.schema.execute_io import ExecuteOutput
from light_vllm.logger import init_logger
from light_vllm.prefill_only.config import PrefillOnlySchedulerConfig
from light_vllm.prefill_only.schema.execute_io import ModelInputForGPU
from light_vllm.utils import CudaMemoryProfiler, is_pin_memory_available

logger = init_logger(__name__)


class ModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: PrefillOnlySchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        attn_backend: AttentionBackend,
    ):
        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.load_config = load_config
        self.attn_backend = attn_backend
        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        # Lazy initialization
        self.model: nn.Module  # Set after load_model

    def load_model(self) -> None:
        from light_vllm.core.loader.loader import (get_model_loader,
                                                   initialize_model)

        logger.info("Starting to load model %s...", self.model_config.model)
        with CudaMemoryProfiler() as m:
            loader = get_model_loader(self.load_config)
            self.model = initialize_model(model_config=self.model_config,
                                          load_config=self.load_config,
                                          device_config=self.device_config,
                                          attn_backend=self.attn_backend)

            loader.load_model(self.model,
                              model_config=self.model_config,
                              device_config=self.device_config)

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPU,
    ) -> ExecuteOutput:
        return self.model(**model_input.to_dict())
