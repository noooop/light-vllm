import enum
from functools import lru_cache
from typing import Optional, Type

import torch

from light_vllm.logger import init_logger
from light_vllm.wde.core.llm_engine import LLMEngine
from light_vllm.wde.encode_only.layers.attention.backends.abstract import (
    EncodeOnlyAttentionBackend)

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


@lru_cache(maxsize=None)
def get_attn_backend(
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    kv_cache_dtype: Optional[str],
    block_size: int,
    is_blocksparse: bool = False,
) -> Type[EncodeOnlyAttentionBackend]:
    """Selects which attention backend to use and lazily imports it."""

    from light_vllm.wde.encode_only.layers.attention.backends.flash_attn import (
        EncodeOnlyFlashAttentionBackend)
    return EncodeOnlyFlashAttentionBackend


class GetAttnBackend:

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        from light_vllm.wde.encode_only.layers.attention.backends.flash_attn import (
            EncodeOnlyFlashAttentionBackend)
        return EncodeOnlyFlashAttentionBackend
