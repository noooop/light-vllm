import dataclasses
from typing import List

import torch

from light_vllm.core.config import DeviceConfig
from light_vllm.core.processor.model_input_builder import ModelInputBuilder
from light_vllm.decoding.backends.sampling_metadata import SamplingMetadata
from light_vllm.decoding.config import (CacheConfig, ModelConfig,
                                        SchedulerConfig)
from light_vllm.decoding.processor.model_input_builder2 import (
    ModelInputForGPUBuilder)
from light_vllm.decoding.scheduler import SchedulerOutput
from light_vllm.decoding.schema.execute_io import (
    DecodingExecuteModelInput, DecodingModelInputForGPUWithSamplingMetadata,
    DecodingWorkerInputForGPU, ExecuteInput)
from light_vllm.decoding.schema.sequence import SequenceGroupMetadata
from light_vllm.utils import is_pin_memory_available


class ChatModelPreProcessor(ModelInputBuilder):

    def __init__(
        self,
        device_config: DeviceConfig,
        model_config: ModelConfig,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        attn_backend,
        cuda_graph,
    ):
        self.device = device_config.device
        self.pin_memory = is_pin_memory_available()

        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.attn_backend = attn_backend
        self.cuda_graph = cuda_graph

    @classmethod
    def from_engine(cls, engine):
        return cls(
            engine.engine_config.device_config,
            engine.engine_config.model_config,
            engine.engine_config.scheduler_config,
            engine.engine_config.cache_config,
            attn_backend=engine.executor.worker.model_runner.attn_backend,
            cuda_graph=engine.executor.worker.model_runner.cuda_graph)

    def _prepare_model_input_tensors(
        self, seq_group_metadata_list: List[SequenceGroupMetadata]
    ) -> DecodingModelInputForGPUWithSamplingMetadata:
        """Helper method to prepare the model input based on a given sequence
        group. Prepares metadata needed for the base model forward pass but not
        metadata for possible additional steps, e.g., sampling.

        The API assumes seq_group_metadata_list is sorted by prefill -> decode.

        The result tensors and data structure also batches input in prefill
        -> decode order. For example,

        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.

        If cuda graph is required, this API automatically pads inputs.
        """
        builder = ModelInputForGPUBuilder(
            model_config=self.model_config,
            scheduler_config=self.scheduler_config,
            cache_config=self.cache_config,
            attn_backend=self.attn_backend,
            cuda_graph=self.cuda_graph,
            device=self.device)
        for seq_group_metadata in seq_group_metadata_list:
            builder.add_seq_group(seq_group_metadata)
        return builder.build()  # type: ignore

    def prepare_model_input(
        self, seq_group_metadata_list: List[SequenceGroupMetadata]
    ) -> DecodingModelInputForGPUWithSamplingMetadata:
        model_input = self._prepare_model_input_tensors(
            seq_group_metadata_list)

        sampling_metadata = SamplingMetadata.prepare(seq_group_metadata_list,
                                                     model_input.seq_lens,
                                                     model_input.query_lens,
                                                     self.device,
                                                     self.pin_memory)

        is_prompt = (seq_group_metadata_list[0].is_prompt
                     if seq_group_metadata_list else None)
        return dataclasses.replace(model_input,
                                   sampling_metadata=sampling_metadata,
                                   is_prompt=is_prompt)

    @torch.inference_mode()
    def prepare_worker_input(
        self, execute_model_input: DecodingExecuteModelInput
    ) -> DecodingWorkerInputForGPU:
        num_seq_groups = len(execute_model_input.seq_group_metadata_list)
        # `blocks_to_swap_in` and `blocks_to_swap_out` are cpu tensors.
        # they contain parameters to launch cudamemcpyasync.
        blocks_to_swap_in = torch.tensor(execute_model_input.blocks_to_swap_in,
                                         device="cpu",
                                         dtype=torch.int64).view(-1, 2)
        blocks_to_swap_out = torch.tensor(
            execute_model_input.blocks_to_swap_out,
            device="cpu",
            dtype=torch.int64).view(-1, 2)
        # `blocks_to_copy` is a gpu tensor. The src and tgt of
        # blocks to copy are in the same device, and `blocks_to_copy`
        # can be used directly within cuda kernels.
        blocks_to_copy = torch.tensor(execute_model_input.blocks_to_copy,
                                      device="cpu",
                                      dtype=torch.int64).view(-1, 2)

        return DecodingWorkerInputForGPU(num_seq_groups=num_seq_groups,
                                         blocks_to_swap_in=blocks_to_swap_in,
                                         blocks_to_swap_out=blocks_to_swap_out,
                                         blocks_to_copy=blocks_to_copy)

    def __call__(self, scheduler_output: SchedulerOutput) -> ExecuteInput:
        seq_group_metadata_list = scheduler_output.seq_group_metadata_list

        execute_model_input = DecodingExecuteModelInput(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_output.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_output.blocks_to_swap_out,
            blocks_to_copy=scheduler_output.blocks_to_copy)

        worker_input: DecodingWorkerInputForGPU = self.prepare_worker_input(
            execute_model_input)
        model_input: DecodingModelInputForGPUWithSamplingMetadata = self.prepare_model_input(
            execute_model_input.seq_group_metadata_list)

        return ExecuteInput(worker_input=worker_input, model_input=model_input)
