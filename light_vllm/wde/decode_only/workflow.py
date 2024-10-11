from typing import Dict

from light_vllm.wde.core.workflow import Workflow
from light_vllm.wde.decode_only.output_last_hidden_states.workflow import (
    DecodeOnlyOutputLastHiddenStatesWorkflow)
from light_vllm.wde.decoding.workflow import DecodeDecodingOnlyWorkflow


class DecodeOnlyWorkflow(Workflow):

    @classmethod
    def from_engine_args(cls, engine_args: Dict):
        if engine_args.get("output_last_hidden_states", False):
            return DecodeOnlyOutputLastHiddenStatesWorkflow
        else:
            return DecodeDecodingOnlyWorkflow
