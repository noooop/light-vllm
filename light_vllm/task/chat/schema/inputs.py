from typing import Union

from dataclasses import dataclass
from light_vllm.layers.sampling_params import SamplingParams
from light_vllm.task.base.schema.inputs import Request, PromptInput, TextOnlyInput


class ChatInput(TextOnlyInput):
    pass


@dataclass
class ChatRequest(Request):
    input: Union[PromptInput, ChatInput]
    sampling_params: SamplingParams
