from abc import ABC, abstractmethod
from typing import List, Optional, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.config import TokenizerPoolConfig

AnyTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class BaseTokenizerGroup(ABC):
    """A group of tokenizers that can be used for LoRA adapters."""

    @classmethod
    @abstractmethod
    def from_config(cls, tokenizer_pool_config: Optional[TokenizerPoolConfig],
                    **init_kwargs) -> "BaseTokenizerGroup":
        pass

    @abstractmethod
    def ping(self) -> bool:
        """Check if the tokenizer group is alive."""
        pass

    @abstractmethod
    def get_max_input_len(self,
                          ) -> Optional[int]:
        """Get the maximum input length for the LoRA request."""
        pass

    @abstractmethod
    def encode(self,
               prompt: str,
               request_id: Optional[str] = None) -> List[int]:
        """Encode a prompt using the tokenizer group."""
        pass

    def check_health(self):
        """Raise exception if the tokenizer group is unhealthy."""
        return
