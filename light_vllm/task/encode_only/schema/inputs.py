from typing import Union

from dataclasses import dataclass
from light_vllm.task.base.schema.inputs import Request, PromptInput, TextOnlyInput


@dataclass
class EncodeOnlyInput(TextOnlyInput):
    pass


@dataclass
class EncodeOnlyRequest(Request):
    input: Union[PromptInput, EncodeOnlyInput]