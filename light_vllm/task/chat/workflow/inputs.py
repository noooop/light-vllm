from typing import (TYPE_CHECKING, List, Literal, Optional, Sequence,
                    TypedDict, Union, cast, overload)


from light_vllm.layers.sampling_params import SamplingParams


class TextPrompt(TypedDict):
    """Schema for a text prompt."""

    prompt: str
    """The input text to be tokenized before passing to the model."""


class TokensPrompt(TypedDict):
    """Schema for a tokenized prompt."""

    prompt_token_ids: List[int]
    """A list of token IDs to pass to the model."""


PromptInput = Union[str, TextPrompt, TokensPrompt]
"""
The inputs to the LLM, which can take one of the following forms:

- A text prompt (:class:`str` or :class:`TextPrompt`)
- A tokenized prompt (:class:`TokensPrompt`)
"""


class ChatInput(TypedDict):
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


class ChatRequest(TypedDict):
    request_id: str
    input: Union[PromptInput, ChatInput]
    sampling_params: SamplingParams


if __name__ == '__main__':
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    for request_id, prompt in enumerate(prompts):
        request = ChatRequest(request_id=str(request_id), input=prompt, sampling_params=sampling_params)
        print(request)
