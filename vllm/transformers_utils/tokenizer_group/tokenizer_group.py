from typing import List, Optional

from vllm.config import TokenizerPoolConfig
from vllm.transformers_utils.tokenizer import (get_tokenizer)
from vllm.utils import LRUCache

from .base_tokenizer_group import AnyTokenizer, BaseTokenizerGroup


class TokenizerGroup(BaseTokenizerGroup):
    """A group of tokenizers that can be used for LoRA adapters."""

    def __init__(self, tokenizer_id: str, max_num_seqs: int,
                 max_input_length: Optional[int], **tokenizer_config):
        self.tokenizer_id = tokenizer_id
        self.tokenizer_config = tokenizer_config
        self.max_input_length = max_input_length
        self.tokenizer = get_tokenizer(self.tokenizer_id, **tokenizer_config)

    @classmethod
    def from_config(cls, **init_kwargs) -> "TokenizerGroup":
        return cls(**init_kwargs)

    def ping(self) -> bool:
        """Check if the tokenizer group is alive."""
        return True

    def get_max_input_len(self) -> Optional[int]:
        """Get the maximum input length for the LoRA request."""
        return self.max_input_length

    def _raise_if_input_too_long(self,
                                 encoded_tokens: List[int]):
        input_length = len(encoded_tokens)

        max_input_length = self.max_input_length
        if max_input_length is not None and input_length > max_input_length:
            raise ValueError("Input too long.", input_length, max_input_length)

    def encode(self,
               prompt: str,
               request_id: Optional[str] = None) -> List[int]:
        tokenizer = self.get_tokenizer()
        ret = tokenizer.encode(prompt)
        self._raise_if_input_too_long(ret)
        return ret

    def get_tokenizer(
        self) -> AnyTokenizer:
        return self.tokenizer
