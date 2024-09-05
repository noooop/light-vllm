from typing import Union

from dataclasses import dataclass
from light_vllm.task.base.schema.inputs import Request, PromptInput, TextOnlyInput


@dataclass
class EncodeOnlyInput(TextOnlyInput):
    pass


@dataclass
class EncodeOnlyRequest(Request):
    input: Union[PromptInput, EncodeOnlyInput]

    @property
    def num_new_tokens(self):
        if isinstance(self.input, EncodeOnlyInput):
            return len(self.input.prompt_token_ids)