

from dataclasses import dataclass
from typing import List, Tuple

import torch


from light_vllm.task.base.schema.execute_io import ExecuteModelInput, WorkerInput, ModelInput, ExecuteInput
from light_vllm.task.base.processor.model_pre_processor import ModelPreProcessor
from light_vllm.task.encode_only.scheduler import SchedulerOutputs
from light_vllm.task.encode_only.layers.attention import EncodeOnlyAttention, EncodeOnlyAttentionMetadata
from light_vllm.task.encode_only.schema.execute_io import ModelInputForGPU


from light_vllm.utils import is_pin_memory_available

pin_memory = is_pin_memory_available()


class EncodeOnlyModelPreProcessor(ModelPreProcessor):
    def __init__(self):
        pass

    @classmethod
    def from_engine(cls, engine):
        return cls()

    def __call__(self,  scheduler_outputs: SchedulerOutputs) -> ExecuteInput:
        input_tokens = []
        input_positions = []
        seq_lens = []
        max_seq_len = 0
        for request in scheduler_outputs.scheduled_requests:
            prompt_token_ids = request.input.prompt_token_ids
            n_tokens = len(prompt_token_ids)
            input_tokens.extend(prompt_token_ids)
            input_positions.extend(list(range(1, n_tokens+1)))
            seq_lens.append(n_tokens)
            max_seq_len = max(max_seq_len, n_tokens)

        input_ids = torch.tensor(input_tokens, dtype=torch.long, pin_memory=pin_memory, device="cpu")
        positions = torch.tensor(input_positions, dtype=torch.long, pin_memory=pin_memory, device="cpu")
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.long, pin_memory=pin_memory, device="cpu")
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                      dtype=torch.int32,
                                      device="cpu")
        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])

        attn_metadata = EncodeOnlyAttentionMetadata(max_seq_len=max_seq_len,
                                                    seq_start_loc=seq_start_loc)

        model_input = ModelInputForGPU(input_ids=input_ids,
                                       positions=positions,
                                       attn_metadata=attn_metadata)

        return ExecuteInput(worker_input=None,
                            model_input=model_input)

