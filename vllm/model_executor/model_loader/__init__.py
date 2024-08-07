from typing import Optional

from torch import nn

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig,
                         ModelConfig, SchedulerConfig)
from vllm.model_executor.model_loader.loader import (BaseModelLoader,
                                                     get_model_loader)
from vllm.model_executor.model_loader.utils import (
    get_architecture_class_name, get_model_architecture)


def get_model(*, model_config: ModelConfig, load_config: LoadConfig,
              device_config: DeviceConfig,
              scheduler_config: SchedulerConfig,
              cache_config: CacheConfig) -> nn.Module:
    loader = get_model_loader(load_config)
    return loader.load_model(model_config=model_config,
                             device_config=device_config,
                             scheduler_config=scheduler_config,
                             cache_config=cache_config)


__all__ = [
    "get_model", "get_model_loader", "BaseModelLoader",
    "get_architecture_class_name", "get_model_architecture"
]
