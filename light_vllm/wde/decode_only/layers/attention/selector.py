import enum
from typing import Optional

import torch

from light_vllm.logger import init_logger
from light_vllm.wde.core.layers.attention.abstract import AttentionType
from light_vllm.wde.core.llm_engine import LLMEngine

logger = init_logger(__name__)


class _Backend(enum.Enum):
    FLASH_ATTN = enum.auto()
    XFORMERS = enum.auto()
    ROCM_FLASH = enum.auto()
    TORCH_SDPA = enum.auto()
    OPENVINO = enum.auto()
    FLASHINFER = enum.auto()
    PALLAS = enum.auto()
    IPEX = enum.auto()

    @staticmethod
    def backend_name_to_enum(backend_name: str) -> "_Backend":
        assert backend_name is not None

        backend_members = _Backend.__members__
        if backend_name not in backend_members:
            raise ValueError(
                f"Invalid attention backend '{backend_name}'. "
                f"Available backends: {', '.join(backend_members)} "
                "(case-sensitive).")

        return _Backend[backend_name]


class AttnBackend:

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        model_config = engine.engine_config.model_config
        num_heads = model_config.get_num_attention_heads()
        head_size = model_config.get_head_size()
        num_kv_heads = model_config.get_num_kv_heads()
        sliding_window = model_config.get_sliding_window()
        dtype = model_config.dtype

        backend = cls.which_attn_to_use(num_heads, head_size, num_kv_heads,
                                        sliding_window, dtype)

        backend_cls = cls.get_backend_cls(backend)

        attn_type = AttentionType.attn_type_name_to_enum(
            engine.workflow.attn_type)

        return backend_cls(attn_type)

    @staticmethod
    def get_backend_cls(backend):
        if backend == _Backend.FLASH_ATTN:
            logger.info("Using FLASH ATTN backend.")
            from light_vllm.wde.decode_only.layers.attention.backends.flash_attn import (  # noqa: E501
                DecodeOnlyFlashAttentionBackend)
            return DecodeOnlyFlashAttentionBackend
        else:
            raise ValueError("Invalid attention backend.")

    @classmethod
    def which_attn_to_use(cls, num_heads: int, head_size: int,
                          num_kv_heads: int, sliding_window: Optional[int],
                          dtype: torch.dtype):
        # Default case.
        selected_backend = _Backend.FLASH_ATTN

        return selected_backend
