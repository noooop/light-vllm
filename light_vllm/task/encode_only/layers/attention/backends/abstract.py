from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from enum import Enum, auto
from typing import (TYPE_CHECKING, Any, Dict, Generic, List, Optional, Set,
                    Tuple, Type, TypeVar)

import torch


class AttentionType(Enum):
    DECODER = auto()  # Decoder attention between previous layer Q/K/V
    ENCODER = auto()  # Encoder attention between previous layer Q/K/V
    ENCODER_DECODER = auto()  # Attention between dec. Q and enc. K/V


class EncodeOnlyAttentionBackend(ABC):
    """Abstract class for attention backends."""

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> Type["EncodeOnlyAttentionImpl"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_metadata_cls() -> Type["EncodeOnlyAttentionMetadata"]:
        raise NotImplementedError

    @classmethod
    def make_metadata(cls, *args, **kwargs) -> "EncodeOnlyAttentionMetadata":
        return cls.get_metadata_cls()(*args, **kwargs)


@dataclass
class EncodeOnlyAttentionMetadata:
    # Maximum query length in the batch.
    max_seq_len: int
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)

        return self


T = TypeVar("T", bound=EncodeOnlyAttentionMetadata)


class EncodeOnlyAttentionImpl(ABC, Generic[T]):

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
        attn_metadata: T,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.ENCODER,
    ) -> torch.Tensor:
        raise NotImplementedError
