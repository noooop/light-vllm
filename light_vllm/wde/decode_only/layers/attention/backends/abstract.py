from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import (Any, Dict, Generic, List, Optional, Set, Tuple, Type,
                    TypeVar)

import torch

from light_vllm.wde.core.layers.attention.abstract import (
    AttentionBackend, AttentionImpl, AttentionMetadata,
    AttentionMetadataBuilder, AttentionType)


class DecodeOnlyAttentionBackend(AttentionBackend, ABC):
    """Abstract class for attention backends."""

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> Type["DncodeOnlyAttentionImpl"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_metadata_cls() -> Type["DncodeOnlyAttentionMetadata"]:
        raise NotImplementedError

    @classmethod
    def make_metadata(cls, *args, **kwargs) -> "DncodeOnlyAttentionMetadata":
        return cls.get_metadata_cls()(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def get_builder_cls() -> Type["DncodeOnlyAttentionMetadataBuilder"]:
        raise NotImplementedError

    @classmethod
    def make_metadata_builder(
            cls, *args, **kwargs) -> "DncodeOnlyAttentionMetadataBuilder":
        return cls.get_builder_cls()(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        raise NotImplementedError


@dataclass
class DecodeOnlyAttentionMetadata(AttentionMetadata):
    """Attention metadata for prefill and decode batched together."""
    # Total number of prefill requests.
    num_prefills: int
    # Number of prefill tokens.
    num_prefill_tokens: int
    # Number of decode tokens. Note that it is equivalent to the number of
    # decode requests.
    num_decode_tokens: int
    # (num_tokens,). The indices of the token slots that input tokens will be
    # stored into. E.g., if `slot_mapping` is [35, 2, 17] and the block size
    # is 16, the three tokens are stored in the 3rd slot in block 2, 2nd slot
    # in block 0, and 1st slot in block 1, respectively.
    slot_mapping: torch.Tensor

    @property
    @abstractmethod
    def prefill_metadata(self) -> Optional["DecodeOnlyAttentionMetadata"]:
        """Return the attention metadata that's required to run prefill
        attention."""
        pass

    @property
    @abstractmethod
    def decode_metadata(self) -> Optional["DecodeOnlyAttentionMetadata"]:
        """Return the attention metadata that's required to run decode
        attention."""
        pass

    def asdict_zerocopy(self,
                        skip_fields: Optional[Set[str]] = None
                        ) -> Dict[str, Any]:
        """Similar to dataclasses.asdict, but avoids deepcopying."""
        if skip_fields is None:
            skip_fields = set()
        # Note that if we add dataclasses as fields, they will need
        # similar handling.
        return {
            field.name: getattr(self, field.name)
            for field in fields(self) if field.name not in skip_fields
        }


T = TypeVar("T", bound=DecodeOnlyAttentionMetadata)


class DecodeOnlyAttentionMetadataBuilder(AttentionMetadataBuilder, ABC,
                                         Generic[T]):
    """Abstract class for attention metadata builders."""

    @abstractmethod
    def __init__(self, input_builder: "ModelRunnerInputBuilderBase") -> None:
        raise NotImplementedError

    @abstractmethod
    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int) -> T:
        """Build attention metadata with on-device tensors."""
        raise NotImplementedError


class DecodeOnlyAttentionImpl(AttentionImpl, ABC, Generic[T]):

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        raise NotImplementedError
