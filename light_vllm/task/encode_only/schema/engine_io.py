
from dataclasses import dataclass
from typing import List
import torch
from light_vllm.task.base.schema.engine_io import Request, PromptInput, TextOnlyInputs, SchedulableRequest, RequestOutput


@dataclass
class EncodeOnlyInput(TextOnlyInputs):
    pass


@dataclass
class EncodeOnlyRequest(Request):
    inputs: PromptInput


@dataclass
class EncodeOnlySchedulableRequest(SchedulableRequest):
    inputs: TextOnlyInputs

    @property
    def num_new_tokens(self):
        return len(self.inputs.prompt_token_ids)


class EncodeOnlyRequestOutput(RequestOutput):
    """
    The output data of an embedding request to the LLM.

    Args:
        request_id (str): A unique identifier for the embedding request.
        outputs (EmbeddingOutput): The embedding results for the given input.
        prompt_token_ids (List[int]): A list of token IDs used in the prompt.
        finished (bool): A flag indicating whether the embedding is completed.
    """

    def __init__(self,
                 request_id: str,
                 outputs: torch.Tensor,
                 prompt_token_ids: List[int],
                 finished: bool):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.finished = finished
        self.outputs = outputs

    def __repr__(self):
        """
        Returns a string representation of an EmbeddingRequestOutput instance.

        The representation includes the request_id and the number of outputs,
        providing a quick overview of the embedding request's results.

        Returns:
            str: A string representation of the EmbeddingRequestOutput instance.
        """
        return (f"EmbeddingRequestOutput(request_id='{self.request_id}', "
                f"outputs={repr(self.outputs)}, "
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"finished={self.finished})")