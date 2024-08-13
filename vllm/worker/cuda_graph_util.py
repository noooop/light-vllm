import time
import numpy as np
from typing import (TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type,
                    TypeVar, Union)
from dataclasses import dataclass
import gc
import torch
import torch.nn as nn
from contextlib import contextmanager
from vllm.layers.attention import AttentionMetadata, get_attn_backend
from vllm.utils import make_tensor_with_pad

from vllm.logger import init_logger

logger = init_logger(__name__)

_PAD_SLOT_ID = -1
_BATCH_SIZE_ALIGNMENT = 8
# Capture graphs for token size 1, 2, 4, 8, 16, 24, 32, 40, ..., 256.
# NOTE: _get_graph_batch_size needs to be updated if this list is changed.
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [
    _BATCH_SIZE_ALIGNMENT * i for i in range(1, 33)
]
_NUM_WARMUP_ITERS = 2


class CUDAGraph:
    def __init__(self, model_config, cache_config, scheduler_config):
        self.model_config = model_config
        self.scheduler_config = scheduler_config

        self.max_seq_len_to_capture = model_config.max_seq_len_to_capture
        self.has_seqlen_agnostic = model_config.contains_seqlen_agnostic_layers()
        self.block_size = cache_config.block_size

        self.graph_runners: Dict[int, CUDAGraphRunner] = {}
        self.graph_memory_pool: Optional[Tuple[
            int, int]] = None  # Set during graph capture.

        # When using CUDA graph, the input block tables must be padded to
        # max_seq_len_to_capture. However, creating the block table in
        # Python can be expensive. To optimize this, we cache the block table
        # in numpy and only copy the actual input content at every iteration.
        # The shape of the cached block table will be
        # (max batch size to capture, max context len to capture / block size).
        self.graph_block_tables = np.zeros(
            (max(_BATCH_SIZES_TO_CAPTURE), self.get_max_block_per_batch()),
            dtype=np.int32)

    def get_max_block_per_batch(self) -> int:
        block_size = self.block_size
        return (self.max_seq_len_to_capture + block_size - 1) // block_size

    def capture_model(self, model, attn_backend, kv_caches: List[torch.Tensor]):
        graph_runners, graph_memory_pool = capture_model(
            model=model,
            model_config=self.model_config,
            kv_caches=kv_caches,
            graph_block_tables=self.graph_block_tables,
            max_num_seqs=self.scheduler_config.max_num_seqs,
            max_seq_len_to_capture=self.max_seq_len_to_capture,
            attn_backend=attn_backend,
            has_seqlen_agnostic=self.has_seqlen_agnostic)
        self.graph_runners = graph_runners
        self.graph_memory_pool = graph_memory_pool

    def model_input_for_gpu_builder_maybe_pad(self, builder, input_tokens, input_positions, seq_lens, max_decode_seq_len):
        batch_size = len(input_tokens)
        use_captured_graph = determine_use_captured_graph(
            decode_only=builder.decode_only,
            enforce_eager=self.model_config.enforce_eager,
            batch_size=batch_size,
            max_decode_seq_len=max_decode_seq_len,
            max_seq_len_to_capture=self.max_seq_len_to_capture
        )

        # If cuda graph can be used, pad tensors accordingly.
        # See `capture_model` API for more details.
        # vLLM uses cuda graph only for decoding requests.
        cuda_graph_pad_size = -1
        if use_captured_graph:
            graph_batch_size = get_graph_batch_size(batch_size)
            assert graph_batch_size >= batch_size
            cuda_graph_pad_size = graph_batch_size - batch_size
            batch_size = graph_batch_size

        # Tokens and positions.
        input_tokens.extend([0] * cuda_graph_pad_size)
        input_positions.extend([0] * cuda_graph_pad_size)
        input_tokens_tensor = torch.tensor(input_tokens,
                                           dtype=torch.long,
                                           device=builder.runner.device)
        input_positions_tensor = torch.tensor(input_positions,
                                              dtype=torch.long,
                                              device=builder.runner.device)

        # Sequence and query lengths.
        seq_lens.extend([1] * cuda_graph_pad_size)

        return input_tokens, input_positions, input_tokens_tensor, input_positions_tensor, seq_lens, cuda_graph_pad_size, batch_size

    def attention_metadata_builder_maybe_pad(self, builder, cuda_graph_pad_size, num_decode_tokens, batch_size, device):
        use_captured_graph = cuda_graph_pad_size != -1

        if use_captured_graph:
            builder.slot_mapping.extend([_PAD_SLOT_ID] * cuda_graph_pad_size)
            builder.block_tables.extend([] * cuda_graph_pad_size)
            num_decode_tokens = batch_size

            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            input_block_tables = self.graph_block_tables[:batch_size]
            for i, block_table in enumerate(builder.block_tables):
                if block_table:
                    input_block_tables[i, :len(block_table)] = block_table
            block_tables = torch.tensor(input_block_tables, device=device)
        else:
            block_tables = make_tensor_with_pad(
                builder.block_tables,
                pad=0,
                dtype=torch.int,
                device=device,
            )
        return num_decode_tokens, block_tables

    def get_graph_runner(self, model_input):
        assert model_input.input_tokens is not None
        graph_batch_size = model_input.input_tokens.shape[0]
        model_executable = self.graph_runners[graph_batch_size]
        return model_executable


class CUDAGraphRunner:

    def __init__(self, model: nn.Module, backend_name: str):
        self.model = model
        self.backend_name = backend_name

        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

        self._graph: Optional[torch.cuda.CUDAGraph] = None

    @property
    def graph(self):
        assert self._graph is not None
        return self._graph

    def capture(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            memory_pool: Optional[Tuple[int, int]],
            stream: torch.cuda.Stream,
            **kwargs,
    ) -> torch.Tensor:
        assert self._graph is None
        # Run the model a few times without capturing the graph.
        # This is to make sure that the captured graph does not include the
        # kernel launches for initial benchmarking (e.g., Triton autotune).
        # Note one iteration is not enough for torch.jit.script
        for _ in range(_NUM_WARMUP_ITERS):
            self.model(
                input_ids,
                positions,
                kv_caches,
                attn_metadata,
                **kwargs,
            )
        torch.cuda.synchronize()

        # Capture the graph.
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool=memory_pool, stream=stream):
            output_hidden_states = self.model(
                input_ids,
                positions,
                kv_caches,
                attn_metadata,
                **kwargs,
            )

            hidden_states = (
                output_hidden_states)

            del output_hidden_states
            # make sure `output_hidden_states` is deleted
            # in the graph's memory pool
            gc.collect()
        torch.cuda.synchronize()

        # Save the input and output buffers.
        self.input_buffers = {
            "input_ids": input_ids,
            "positions": positions,
            "kv_caches": kv_caches,
            "slot_mapping": attn_metadata.slot_mapping,
            "seq_lens_tensor": attn_metadata.decode_metadata.seq_lens_tensor,
            "block_tables": attn_metadata.decode_metadata.block_tables,
            **kwargs,
        }

        self.output_buffers = {
            "hidden_states": hidden_states
        }

        return hidden_states

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            **kwargs,
    ) -> torch.Tensor:
        # KV caches are fixed tensors, so we don't need to copy them.
        del kv_caches

        # Copy the input tensors to the input buffers.
        self.input_buffers["input_ids"].copy_(input_ids, non_blocking=True)
        self.input_buffers["positions"].copy_(positions, non_blocking=True)
        self.input_buffers["slot_mapping"].copy_(attn_metadata.slot_mapping,
                                                 non_blocking=True)

        self.input_buffers["seq_lens_tensor"].copy_(
            attn_metadata.decode_metadata.seq_lens_tensor,
            non_blocking=True)
        self.input_buffers["block_tables"].copy_(
            attn_metadata.decode_metadata.block_tables, non_blocking=True)

        if "seqlen_agnostic_capture_inputs" in self.input_buffers:
            self.model.copy_inputs_before_cuda_graphs(self.input_buffers,
                                                      **kwargs)
        # Run the graph.
        self.graph.replay()
        if "seqlen_agnostic_capture_inputs" in self.input_buffers:
            self.model.copy_outputs_after_cuda_graphs(self.input_buffers,
                                                      **kwargs)
        return self.output_buffers["hidden_states"]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def determine_use_captured_graph(decode_only: bool,
                                 enforce_eager: bool,
                                 batch_size: int,
                                 max_decode_seq_len: int,
                                 max_seq_len_to_capture: int) -> bool:
    return (decode_only and not enforce_eager
            and batch_size <= _BATCH_SIZES_TO_CAPTURE[-1]
            and max_decode_seq_len <= max_seq_len_to_capture)


@dataclass
class GraphCaptureContext:
    stream: torch.cuda.Stream


@contextmanager
def graph_capture():
    stream = torch.cuda.Stream()
    graph_capture_context = GraphCaptureContext(stream)

    curr_stream = torch.cuda.current_stream()
    if curr_stream != stream:
        stream.wait_stream(curr_stream)

    with torch.cuda.stream(stream):
        yield graph_capture_context


def get_graph_batch_size(batch_size: int) -> int:
    """Returns the padded batch size given actual batch size.

    Batch sizes are 1, 2, 4, _BATCH_SIZE_ALIGNMENT,
    2*_BATCH_SIZE_ALIGNMENT, 3*_BATCH_SIZE_ALIGNMENT...
    """
    if batch_size <= 2:
        return batch_size
    elif batch_size <= 4:
        return 4
    else:
        return ((batch_size + _BATCH_SIZE_ALIGNMENT - 1) //
                _BATCH_SIZE_ALIGNMENT * _BATCH_SIZE_ALIGNMENT)


@torch.inference_mode()
def capture_model(model,
                  model_config,
                  kv_caches: List[torch.Tensor],
                  graph_block_tables,
                  max_num_seqs,
                  max_seq_len_to_capture,
                  attn_backend,
                  has_seqlen_agnostic):
    """Cuda graph capture a model.

    Note that CUDA graph's performance gain is negligible if number
    of batched tokens are larger than 200. And since CUDA graph
    requires fixed sized tensors, supporting large/variable batch
    size requires high GPU memory overhead. Thus, vLLM only captures
    decoding requests. Mixed batch (chunked prefill + decoding) or
    prefill requests are not captured.

    Since it is used for decoding-only, it assumes there's only 1 token
    per sequence in the batch.
    """
    assert not model_config.enforce_eager
    logger.info("Capturing the model for CUDA graphs. This may lead to "
                "unexpected consequences if the model is not static. To "
                "run the model in eager mode, set 'enforce_eager=True' or "
                "use '--enforce-eager' in the CLI.")
    logger.info("CUDA graphs can take additional 1~3 GiB memory per GPU. "
                "If you are running out of memory, consider decreasing "
                "`gpu_memory_utilization` or enforcing eager mode. "
                "You can also reduce the `max_num_seqs` as needed "
                "to decrease memory usage.")
    start_time = time.perf_counter()

    # Prepare dummy inputs. These will be reused for all batch sizes.
    max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
    input_tokens = torch.zeros(max_batch_size, dtype=torch.long).cuda()
    input_positions = torch.zeros(max_batch_size, dtype=torch.long).cuda()
    slot_mapping = torch.empty(max_batch_size, dtype=torch.long).cuda()
    slot_mapping.fill_(_PAD_SLOT_ID)
    seq_lens = torch.ones(max_batch_size, dtype=torch.int32).cuda()
    block_tables = torch.from_numpy(graph_block_tables).cuda()

    graph_batch_size = get_graph_batch_size(max_num_seqs)
    batch_size_capture_list = [
        bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= graph_batch_size
    ]

    graph_runners: Dict[int, CUDAGraphRunner] = {}
    graph_memory_pool: Optional[Tuple[int, int]] = None  # Set during graph capture.

    with graph_capture() as graph_capture_context:
        # NOTE: Capturing the largest batch size first may help reduce the
        # memory usage of CUDA graph.
        for batch_size in reversed(batch_size_capture_list):
            attn_metadata = attn_backend.make_metadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decode_tokens=batch_size,
                slot_mapping=slot_mapping[:batch_size],
                seq_lens=None,
                seq_lens_tensor=seq_lens[:batch_size],
                max_query_len=None,
                max_prefill_seq_len=0,
                max_decode_seq_len=max_seq_len_to_capture,
                query_start_loc=None,
                seq_start_loc=None,
                context_lens_tensor=None,
                block_tables=block_tables[:batch_size],
                use_cuda_graph=True,
            )

            graph_runner = CUDAGraphRunner(
                model, attn_backend.get_name())

            capture_inputs = {
                "input_ids": input_tokens[:batch_size],
                "positions": input_positions[:batch_size],
                "kv_caches": kv_caches,
                "attn_metadata": attn_metadata,
                "memory_pool": graph_memory_pool,
                "stream": graph_capture_context.stream
            }
            if has_seqlen_agnostic:
                # Only used by Mamba-based models CUDA graph atm (Jamba)
                capture_inputs.update({
                    "seqlen_agnostic_capture_inputs":
                        model.get_seqlen_agnostic_capture_inputs(
                            batch_size)
                })
            graph_runner.capture(**capture_inputs)
            graph_runners[batch_size] = graph_runner
            graph_memory_pool = graph_runner.graph.pool()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # This usually takes < 10 seconds.
    logger.info("Graph capturing finished in %.0f secs.", elapsed_time)

    return graph_runners, graph_memory_pool
