import warnings
from typing import (List, Optional)
import torch
import torch.distributed
import torch.nn as nn
import light_vllm.envs as envs
from light_vllm.layers.sampling_params import SamplingParams

from light_vllm.wde.core.config import DeviceConfig, LoadConfig
from light_vllm.wde.chat.config import CacheConfig, ModelConfig, SchedulerConfig

from light_vllm.wde.core.runner.cuda_graph_util import CUDAGraph

from light_vllm.wde.core.inputs import INPUT_REGISTRY
from light_vllm.logger import init_logger
from light_vllm.models.utils import set_cpu_offload_max_bytes

from light_vllm.wde.core.schema.sequence import SequenceGroupMetadata
from light_vllm.wde.core.schema.execute_io import ExecuteOutput
from light_vllm.wde.decode_only.layers.attention import DecodeOnlyAttentionBackend

from light_vllm.utils import (CudaMemoryProfiler, is_hip,
                              is_pin_memory_available)
from light_vllm.wde.chat.schema.execute_io import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class GPUModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        attn_backend: DecodeOnlyAttentionBackend,
        kv_cache_dtype: Optional[str] = "auto",
    ):
        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.load_config = load_config
        self.attn_backend = attn_backend

        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.cuda_graph = CUDAGraph(model_config, cache_config, scheduler_config)

        # Lazy initialization
        self.model: nn.Module  # Set after load_model

        set_cpu_offload_max_bytes(
            int(self.cache_config.cpu_offload_gb * 1024**3))

    def load_model(self) -> None:
        from light_vllm.wde.core.loader.loader import get_model_loader, initialize_model

        logger.info("Starting to load model %s...", self.model_config.model)
        with CudaMemoryProfiler() as m:
            loader = get_model_loader(self.load_config)
            self.model = initialize_model(
                model_config=self.model_config,
                load_config=self.load_config,
                device_config=self.device_config,
                cache_config=self.cache_config,
                attn_backend=self.attn_backend)

            loader.load_model(
                self.model,
                model_config=self.model_config,
                device_config=self.device_config)

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

        if self.kv_cache_dtype == "fp8" and is_hip():
            # Currently only ROCm accepts kv-cache scaling factors
            # via quantization_param_path and this will be deprecated
            # in the future.
            if self.model_config.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    warnings.warn(
                        "Loading kv cache scaling factor from JSON is "
                        "deprecated and will be removed. Please include "
                        "kv cache scaling factors in the model checkpoint.",
                        FutureWarning,
                        stacklevel=2)
                    self.model.load_kv_cache_scales(
                        self.model_config.quantization_param_path)
                    logger.info("Loaded KV cache scaling factors from %s",
                                self.model_config.quantization_param_path)
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.",
                        self.model.__class__)
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors "
                    "provided. Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!")

        if envs.VLLM_TEST_DYNAMO_GRAPH_CAPTURE:
            self.model = torch.compile(self.model,
                                       fullgraph=True,
                                       backend="eager")

    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        # This represents the maximum number of different requests

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []
        # Additional GPU memory may be needed for vision encoding, which needs
        # to be accounted for when calculating the GPU blocks for
        # vLLM blocker manager.
        # To exercise the worst scenario for GPU memory consumption,
        # the number of seqs (batch_size) is chosen to maximize the number
        # of images processed.
        model_config = self.model_config

        batch_size = 0
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            batch_size += seq_len

            seq_data, dummy_multi_modal_data = INPUT_REGISTRY \
                .dummy_data_for_profiling(model_config, seq_len)

            # Having more tokens is over-conservative but otherwise fine
            assert len(seq_data.prompt_token_ids) >= seq_len, (
                f"Expected at least {seq_len} dummy tokens for profiling, "
                f"but got: {len(seq_data.prompt_token_ids)}")

            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None
            )
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers()
        kv_caches = [None] * num_layers
        model_input = self.prepare_model_input(seqs)
        self.execute_model(model_input, kv_caches)
        torch.cuda.synchronize()
        return

    def capture_model(self, kv_caches: List[torch.Tensor]) -> None:
        self.cuda_graph.capture_model(self.model, self.attn_backend, kv_caches)

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
    ) -> Optional[List[ExecuteOutput]]:

        # Currently cuda graph is only supported by the decode phase.
        assert model_input.attn_metadata is not None
        prefill_meta = model_input.attn_metadata.prefill_metadata
        decode_meta = model_input.attn_metadata.decode_metadata
        if prefill_meta is None and decode_meta.use_cuda_graph:
            model_executable = self.cuda_graph.get_graph_runner(model_input)
        else:
            model_executable = self.model

        hidden_states = model_executable(
            input_ids=model_input.input_tokens,
            positions=model_input.input_positions,
            kv_caches=kv_caches,
            attn_metadata=model_input.attn_metadata)

        logits = self.model.compute_logits(hidden_states,
                                           model_input.sampling_metadata)

        # Sample the next token.
        output: ExecuteOutput = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )

        return [output]


