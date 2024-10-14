from dataclasses import dataclass
from typing import Dict, List, Optional, Union


class Params:
    pass


class Inputs:
    pass


@dataclass
class TextPrompt(Inputs):
    """Schema for a text prompt."""

    prompt: str
    """The input text to be tokenized before passing to the model."""


@dataclass
class TokensPrompt(Inputs):
    """Schema for a tokenized prompt."""

    prompt_token_ids: List[int]
    """A list of token IDs to pass to the model."""


@dataclass
class TextOnlyInputs(Inputs):
    prompt_token_ids: List[int]
    """The token IDs of the prompt."""

    prompt: Optional[str] = None
    """
    The original prompt text corresponding to the token IDs, if available.
    """


PromptInput = Union[str, Dict, TextPrompt, TokensPrompt, TextOnlyInputs]


@dataclass
class Request:
    request_id: str
    arrival_time: float


@dataclass
class TextRequest(Request):
    inputs: Dict
    params: Optional[Params]


class ValidationError(ValueError):
    pass


class SchedulableRequest(Request):

    @property
    def num_new_tokens(self):
        raise NotImplementedError


@dataclass
class TextSchedulableRequest(SchedulableRequest):
    inputs: TextOnlyInputs
    params: Optional[Params]

    @property
    def num_new_tokens(self):
        return len(self.inputs.prompt_token_ids)


@dataclass
class SchedulerOutput:
    pass


@dataclass
class RequestOutput(Request):
    finished: bool
