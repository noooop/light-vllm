
import torch
import os
import random
import numpy as np
import time


def pitch():
    import gc
    from vllm.worker.worker import Worker
    from typing import List, Optional, Set, Tuple, Type

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        self.free_gpu_memory = free_gpu_memory
        self.total_gpu_memory = total_gpu_memory

        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        peak_memory = self.init_gpu_memory - free_gpu_memory
        assert peak_memory > 0, (
            "Error in memory profiling. "
            f"Initial free memory {self.init_gpu_memory}, current free memory"
            f" {free_gpu_memory}. This happens when the GPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")

        cache_block_size = self.get_cache_block_size_bytes()
        num_gpu_blocks = int(
            (total_gpu_memory * self.cache_config.gpu_memory_utilization -
             peak_memory) // cache_block_size)
        num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                             cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        gc.collect()
        torch.cuda.empty_cache()
        return num_gpu_blocks, num_cpu_blocks

    Worker.determine_num_available_blocks = determine_num_available_blocks


def benchmark(args):
    random.seed(args.seed)

    os.environ["VLLM_NO_USAGE_STATS"] = "True"

    try:
        import light_vllm
        from light_vllm import LLMEngine, EngineArgs, SamplingParams, TextPrompt
        print("light_vllm:", light_vllm.__version__)

    except Exception:
        import vllm
        from vllm import LLMEngine, EngineArgs, SamplingParams, TextPrompt
        pitch()
        print("vllm:", vllm.__version__)

    engine_args = EngineArgs(
        model=args.model,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        kv_cache_dtype=args.kv_cache_dtype,
        quantization_param_path=args.quantization_param_path,
        device=args.device,
        enable_prefix_caching=args.enable_prefix_caching,
        download_dir=args.download_dir,
        enable_chunked_prefill=args.enable_chunked_prefill,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        distributed_executor_backend=args.distributed_executor_backend,
        disable_log_stats=True
    )
    engine = LLMEngine.from_engine_args(engine_args)


    print(f"init_gpu_memory {engine.model_executor.driver_worker.init_gpu_memory / float(2 ** 30): .4f} GB")
    print(f"free_gpu_memory {engine.model_executor.driver_worker.free_gpu_memory / float(2 ** 30): .4f} GB")
    print(f"total_gpu_memory {engine.model_executor.driver_worker.total_gpu_memory / float(2 ** 30): .4f} GB")
    print(engine.cache_config.block_size, engine.cache_config.num_gpu_blocks, engine.cache_config.num_cpu_blocks)


if __name__ == '__main__':
    from easydict import EasyDict as edict
    args = edict()

    args.dataset = None
    args.input_len = 512
    args.output_len = 512

    args.model = "Qwen/Qwen2-7B-Instruct"
    args.trust_remote_code = False
    args.tokenizer = args.model
    args.quantization = None
    args.quantization_param_path = None
    args.tensor_parallel_size = 1
    args.seed = 0
    args.n = 1
    args.use_beam_search = False
    args.num_prompts = 1000
    args.dtype = 'auto'
    args.max_model_len = 5000
    args.enforce_eager = True
    args.kv_cache_dtype = "auto"
    args.device = "cuda"
    args.enable_prefix_caching = False
    args.gpu_memory_utilization = 0.9
    args.output_json = None
    args.distributed_executor_backend = None
    args.download_dir = None

    import sys
    from concurrent.futures import ProcessPoolExecutor

    def run(args):
        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(benchmark, args)
            f.result()

    if "full" in sys.argv:
        max_num_seqs_list = [1024, 768, 512, 384, 256, 128, 64, 32]
    else:
        max_num_seqs_list = [256, 128]

    for enable_chunked_prefill in [True, False]:
        args.enable_chunked_prefill = False
        args.max_num_batched_tokens = None
        args.max_num_seqs = 512
        run(args)