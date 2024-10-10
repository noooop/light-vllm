from typing import Dict

from light_vllm.wde.chat.workflow import ChatWorkflow
from light_vllm.wde.core.workflow import Workflow
from light_vllm.wde.prefill_only.workflow import PrefillOnlyWorkflow


class DecodeOnlyOutputLastHiddenStatesWorkflow(Workflow):
    EngineArgs: str = ("light_vllm.wde.decode_only.arg_utils:"
                       "DecodeOnlyEngineArgs")
    attn_type: str = "DECODER"

    @classmethod
    def from_engine(cls, engine):
        workflow = PrefillOnlyWorkflow.from_engine(engine)

        if engine.engine_config.model_config.enable_bidirectional:
            workflow.attn_type = "ENCODER"
        else:
            workflow.attn_type = "DECODER"

        workflow.OutputProcessor = ("light_vllm.wde.decode_only.processor."
                                    "output_processor:"
                                    "DecodeOnlyHiddenStatesOutputProcessor")
        return workflow


class DecodeOnlyWorkflow(Workflow):

    @classmethod
    def workflow_cls_from_engine_args(cls, engine_args: Dict):
        if engine_args.get("output_last_hidden_states", False):
            return DecodeOnlyOutputLastHiddenStatesWorkflow
        else:
            return ChatWorkflow
