
from dataclasses import dataclass
from typing import Sequence, Union, List
from light_vllm.wde.core.schema.engine_io import Request
from light_vllm.wde.core.schema.engine_io import Inputs, RequestOutput


@dataclass
class Pairs(Inputs):
    query: str
    passage: str


RerankerInputs = Union[Sequence, Pairs]


@dataclass
class RerankerRequest(Request):
    inputs: Pairs


class RerankerRequestOutput(RequestOutput):

    def __init__(self,
                 request_id: str,
                 score: float,
                 prompt_token_ids: List[int],
                 finished: bool):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.finished = finished
        self.score = score

    def __repr__(self):
        return (f"RerankerRequestOutput(request_id='{self.request_id}', "
                f"score={repr(self.score)}, "
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"finished={self.finished})")
