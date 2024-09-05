from typing import Any, Dict, Optional

import time
from light_vllm.inputs.tokenizer import Tokenizer
from light_vllm.task.encode_only.schema.inputs import PromptInput, EncodeOnlyInput, EncodeOnlyRequest
from light_vllm.task.base.processor.input_processor import InputProcessor, RequestProcessor


class EncodeOnlyModelInputProcessor(InputProcessor):
    @classmethod
    def from_engine(cls, engine):
        return cls()

    def __call__(self,
                 request_id: str,
                 prompt: PromptInput,
                 params,
                 arrival_time: Optional[float] = None) -> EncodeOnlyRequest:
        if not arrival_time:
            arrival_time = time.time()
        request = EncodeOnlyRequest(request_id=str(request_id),
                                    input=prompt,
                                    arrival_time=arrival_time)
        return request


class EncodeOnlyModelRequestProcessor(RequestProcessor):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_engine(cls, engine):
        return cls(engine.tokenizer)

    def __call__(self, request: EncodeOnlyRequest) -> EncodeOnlyRequest:
        if not isinstance(request.input, EncodeOnlyInput):
            input = request.input
            if isinstance(request.input, str):
                input = {"prompt": input}

            if "prompt_token_ids" not in input:
                tokenizer = self.tokenizer

                prompt_token_ids = tokenizer.encode(input["prompt"])
            else:
                prompt_token_ids = input["prompt_token_ids"]

            input = EncodeOnlyInput(prompt_token_ids=prompt_token_ids,
                                    prompt=input.get("prompt"))
            request.input = input

        return request
