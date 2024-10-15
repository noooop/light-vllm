import time
from typing import List

from light_vllm.core.processor.output_processor import OutputProcessor
from light_vllm.decoding.backends.sampler import (get_logprobs,
                                                  get_pythonized_sample_results
                                                  )
from light_vllm.decoding.scheduler import SchedulerOutput
from light_vllm.decoding.schema.engine_io import ChatModelRequestOutput
from light_vllm.decoding.schema.execute_io import (
    CompletionSequenceGroupOutput, SamplerOutput, SequenceOutput)


class ChatModelOutputProcessor(OutputProcessor):

    def __init__(self, scheduler_config, scheduler, tokenizer, seq_counter):
        from light_vllm.decoding.processor.utils.single_step import (
            SingleStepOutputProcessor)
        from light_vllm.decoding.processor.utils.stop_checker import (
            StopChecker)
        self.scheduler = scheduler

        self.output_processor = SingleStepOutputProcessor(
            tokenizer,
            seq_counter,
            stop_checker=StopChecker(scheduler_config.max_model_len,
                                     tokenizer),
            max_model_len=scheduler_config.max_model_len,
        )

    @classmethod
    def from_engine(cls, engine):
        return cls(engine.engine_config.scheduler_config, engine.scheduler,
                   engine.tokenizer, engine.seq_counter)

    def __call__(
            self, scheduler_output: SchedulerOutput,
            execute_output: SamplerOutput) -> List[ChatModelRequestOutput]:
        now = time.time()

        scheduled_seq_groups = scheduler_output.scheduled_seq_groups
        ignored_seq_groups = scheduler_output.ignored_seq_groups
        seq_group_metadata_list = scheduler_output.seq_group_metadata_list

        # sampler GPU -> CPU & pythonized
        deferred_sample_results_args = execute_output.deferred_sample_results_args
        sampled_token_ids_tensor = execute_output.sampled_token_ids
        sampling_metadata = deferred_sample_results_args.sampling_metadata
        logprobs = execute_output.logprobs

        maybe_deferred_sample_results, maybe_sampled_tokens_tensor = get_pythonized_sample_results(
            deferred_sample_results_args), sampled_token_ids_tensor

        prompt_logprobs, sample_logprobs = get_logprobs(
            logprobs, sampling_metadata, maybe_deferred_sample_results)

        sampler_output: List[List[CompletionSequenceGroupOutput]] = []

        for (seq_group, sample_result, group_prompt_logprobs,
             group_sample_logprobs) in zip(sampling_metadata.seq_groups,
                                           maybe_deferred_sample_results,
                                           prompt_logprobs, sample_logprobs):
            seq_ids = seq_group.seq_ids
            next_token_ids, parent_ids = sample_result
            seq_outputs: List[SequenceOutput] = []
            for parent_id, next_token_id, logprobs in zip(
                    parent_ids, next_token_ids, group_sample_logprobs):
                seq_outputs.append(
                    SequenceOutput(seq_ids[parent_id], next_token_id,
                                   logprobs))
            sampler_output.append([
                CompletionSequenceGroupOutput(seq_outputs,
                                              group_prompt_logprobs)
            ])

        # sampler GPU -> CPU & pythonized end

        # Update the scheduled sequence groups with the model outputs.
        for scheduled_seq_group, outputs, seq_group_meta in zip(
                scheduled_seq_groups, sampler_output, seq_group_metadata_list):
            seq_group = scheduled_seq_group.seq_group
            seq_group.update_num_computed_tokens(
                scheduled_seq_group.token_chunk_size)

            self.output_processor.process_prompt_logprob(seq_group, outputs)
            if seq_group_meta.do_sample:
                seq_need_fork, seq_need_free = self.output_processor.process_outputs(
                    seq_group, outputs)

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
