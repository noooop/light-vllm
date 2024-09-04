from dataclasses import dataclass

from typing import List, Union, Optional


class Params:
    pass


class Prompt:
    pass


@dataclass
class Request:
    request_id: str
    arrival_time: float


@dataclass
class TextPrompt(Prompt):
    """Schema for a text prompt."""

    prompt: str
    """The input text to be tokenized before passing to the model."""


@dataclass
class TokensPrompt(Prompt):
    """Schema for a tokenized prompt."""

    prompt_token_ids: List[int]
    """A list of token IDs to pass to the model."""


PromptInput = Union[str, TextPrompt, TokensPrompt]
"""
The inputs to the LLM, which can take one of the following forms:

- A text prompt (:class:`str` or :class:`TextPrompt`)
- A tokenized prompt (:class:`TokensPrompt`)
"""


@dataclass
class TextOnlyInput:
    """
    The inputs in :class:`~vllm.LLMEngine` before they are
    passed to the model executor.
    """
    prompt_token_ids: List[int]
    """The token IDs of the prompt."""

    prompt: Optional[str]
    """
    The original prompt text corresponding to the token IDs, if available.
    """