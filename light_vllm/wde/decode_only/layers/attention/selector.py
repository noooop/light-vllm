import enum
from functools import lru_cache
from typing import Optional, Type

import torch

import light_vllm.envs as envs
from light_vllm.wde.core.llm_engine import LLMEngine

from light_vllm.wde.decode_only.layers.attention.backends.abstract import AttentionBackend
from light_vllm.logger import init_logger
from light_vllm.platforms import current_platform
from light_vllm.utils import is_cpu

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
) -> Type[AttentionBackend]:
    """Selects which attention backend to use and lazily imports it."""

    backend = which_attn_to_use(num_heads, head_size, num_kv_heads,
                                sliding_window, dtype, kv_cache_dtype,
                                block_size)
    if backend == _Backend.FLASH_ATTN:
        from light_vllm.wde.decode_only.layers.attention.backends.flash_attn import (  # noqa: F401
            DecodeOnlyFlashAttentionBackend)
        return DecodeOnlyFlashAttentionBackend
    if backend == _Backend.XFORMERS:
        logger.info("Using XFormers backend.")
        from light_vllm.wde.decode_only.layers.attention.backends.xformers import (  # noqa: F401
            DecodeOnlyXFormersBackend)
        return DecodeOnlyXFormersBackend
    elif backend == _Backend.TORCH_SDPA:
        assert is_cpu(), RuntimeError(
            "Torch SDPA backend is only used for the CPU device.")
        logger.info("Using Torch SDPA backend.")
        from light_vllm.wde.decode_only.layers.attention.backends.torch_sdpa import DecodeOnlyTorchSDPABackend
        return DecodeOnlyTorchSDPABackend
    else:
        raise ValueError("Invalid attention backend.")


def which_attn_to_use(
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    kv_cache_dtype: Optional[str],
    block_size: int,
) -> _Backend:
    """Returns which flash attention backend to use."""
    # Default case.
    selected_backend = _Backend.FLASH_ATTN

    # Check the environment variable and override if specified
    backend_by_env_var: Optional[str] = envs.VLLM_ATTENTION_BACKEND
    if backend_by_env_var is not None:
        backend_members = _Backend.__members__
        if backend_by_env_var not in backend_members:
            raise ValueError(
                f"Invalid attention backend '{backend_by_env_var}'. "
                f"Available backends: {', '.join(backend_members)} "
                "(case-sensitive).")
        selected_backend = _Backend[backend_by_env_var]

    if is_cpu():
        if selected_backend != _Backend.TORCH_SDPA:
            logger.info("Cannot use %s backend on CPU.", selected_backend)
        return _Backend.TORCH_SDPA

    # FlashAttn in NVIDIA GPUs.
    if selected_backend == _Backend.FLASH_ATTN:
        if current_platform.get_device_capability()[0] < 8:
            # Volta and Turing NVIDIA GPUs.
            logger.info(
                "Cannot use FlashAttention-2 backend for Volta and Turing "
                "GPUs.")
            selected_backend = _Backend.XFORMERS
        elif dtype not in (torch.float16, torch.bfloat16):
            logger.info(
                "Cannot use FlashAttention-2 backend for dtype other than "
                "torch.float16 or torch.bfloat16.")
            selected_backend = _Backend.XFORMERS
        elif kv_cache_dtype is not None and kv_cache_dtype.startswith("fp8"):
            logger.info(
                "Cannot use FlashAttention-2 backend for FP8 KV cache.")
            selected_backend = _Backend.XFORMERS
        elif block_size % 16 != 0:
            logger.info(
                "Cannot use FlashAttention-2 backend for block size not "
                "divisible by 16.")
            selected_backend = _Backend.XFORMERS
        elif sliding_window is not None:
            logger.info(
                "Cannot use FlashAttention-2 backend due to sliding window.")
            selected_backend = _Backend.XFORMERS

    # FlashAttn is valid for the model, checking if the package is installed.
    if selected_backend == _Backend.FLASH_ATTN:
        try:
            import vllm_flash_attn  # noqa: F401
            from light_vllm.wde.decode_only.layers.attention.backends.flash_attn import (  # noqa: F401
                DecodeOnlyFlashAttentionBackend)

            supported_sizes = DecodeOnlyFlashAttentionBackend.get_supported_head_sizes()
            if head_size not in supported_sizes:
                logger.info(
                    "Cannot use FlashAttention-2 backend for head size %d.",
                    head_size)
                selected_backend = _Backend.XFORMERS
        except ImportError:
            logger.info(
                "Cannot use FlashAttention-2 backend because the "
                "vllm_flash_attn package is not found. "
                "`pip install vllm-flash-attn` for better performance.")
            selected_backend = _Backend.XFORMERS

    return selected_backend


class GetAttnBackend:
    @classmethod
    def from_engine(cls, engine: LLMEngine):
        from light_vllm.wde.decode_only.layers.attention.backends.flash_attn import (  # noqa: F401
            DecodeOnlyFlashAttentionBackend)
        return DecodeOnlyFlashAttentionBackend