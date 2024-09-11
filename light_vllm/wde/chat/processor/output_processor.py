
import time
from typing import List

from light_vllm.wde.chat.scheduler import SchedulerOutputs
from light_vllm.wde.core.schema.sequence import SequenceGroup, SequenceGroupMetadata
from light_vllm.wde.chat.schema.execute_io import SequenceGroupOutput, SamplerOutput
from light_vllm.wde.chat.schema.outputs import ChatModelRequestOutput
from light_vllm.wde.core.processor.output_processor import OutputProcessor


class ChatModelOutputProcessor(OutputProcessor):
    def __init__(self, scheduler_config, scheduler, tokenizer, seq_counter):
        from light_vllm.wde.chat.processor.utils.stop_checker import StopChecker
        from light_vllm.wde.chat.processor.utils.single_step import SingleStepOutputProcessor
        self.scheduler = scheduler

        self.output_processor = SingleStepOutputProcessor(
            tokenizer,
            seq_counter,
            stop_checker=StopChecker(
                scheduler_config.max_model_len,
                tokenizer
            ),
            max_model_len=scheduler_config.max_model_len,
        )

    @classmethod
    def from_engine(cls, engine):
        return cls(engine.engine_config.scheduler_config,
                   engine.scheduler,
                   engine.tokenizer,
                   engine.seq_counter)

    def __call__(self, scheduler_outputs: SchedulerOutputs, execute_output) -> List[ChatModelRequestOutput]:
        now = time.time()

        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups
        ignored_seq_groups = scheduler_outputs.ignored_seq_groups
        seq_group_metadata_list = scheduler_outputs.seq_group_metadata_list

        # Organize outputs by [sequence group][step] instead of
        # [step][sequence group].
        output_by_sequence_group: List[List[SequenceGroupOutput]] = [
            [] for _ in range(len(execute_output[0]))
        ]
        for step in execute_output:
            for i, sequence_group_output in enumerate(step):
                output_by_sequence_group[i].append(sequence_group_output)

        # Update the scheduled sequence groups with the model outputs.
        for scheduled_seq_group, outputs, seq_group_meta in zip(
                scheduled_seq_groups, output_by_sequence_group,
                seq_group_metadata_list):
            seq_group = scheduled_seq_group.seq_group
            seq_group.update_num_computed_tokens(
                scheduled_seq_group.token_chunk_size)

            self.output_processor.process_prompt_logprob(seq_group, outputs)
            if seq_group_meta.do_sample:
                seq_need_fork, seq_need_free = self.output_processor.process_outputs(seq_group, outputs)

                for parent, seq in seq_need_fork:
                    self.scheduler.fork_seq(parent, seq)
                for seq in seq_need_free:
                    self.scheduler.free_seq(seq)

        # Create the outputs.
        request_outputs = []
        for scheduled_seq_group in scheduled_seq_groups:
            seq_group = scheduled_seq_group.seq_group
            seq_group.maybe_set_first_token_time(now)
            request_output = ChatModelRequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)
        for seq_group in ignored_seq_groups:
            request_output = ChatModelRequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)
        return request_outputs