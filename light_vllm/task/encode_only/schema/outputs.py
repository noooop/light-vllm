
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (TYPE_CHECKING, Dict, List, Mapping, Optional, Set, Tuple,
                    Union)

from light_vllm.task.base.schema.outputs import RequestOutput


class EncodeOnlyRequestOutput(RequestOutput):
    """
    The output data of an embedding request to the LLM.

    Args:
        request_id (str): A unique identifier for the embedding request.
        outputs (EmbeddingOutput): The embedding results for the given input.
        prompt_token_ids (List[int]): A list of token IDs used in the prompt.
        finished (bool): A flag indicating whether the embedding is completed.
    """

    def __init__(self, request_id: str, outputs: "EmbeddingOutput",
                 prompt_token_ids: List[int], finished: bool):
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