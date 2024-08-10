"""Benchmark offline inference throughput."""

import os

os.environ["VLLM_USE_MODELSCOPE"] = "True"

import numpy as np
import random
import time
from typing import List, Optional, Tuple


def run_vllm(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    quantization_param_path: Optional[str],
    device: str,
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    distributed_executor_backend: Optional[str],
    gpu_memory_utilization: float = 0.9,
    download_dir: Optional[str] = None,
) -> (float, float):
    from vllm import LLMEngine, EngineArgs, SamplingParams, TextPrompt

    import logging
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(logging.ERROR)

    engine_args = EngineArgs(
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        quantization_param_path=quantization_param_path,
        device=device,
        enable_prefix_caching=enable_prefix_caching,
        download_dir=download_dir,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        distributed_executor_backend=distributed_executor_backend,
    )
    engine = LLMEngine.from_engine_args(engine_args)

    # Add the requests to the engine.
    for request_id, (prompt, _, output_len) in enumerate(requests):
        inputs = TextPrompt(prompt=prompt)
        sampling_params = SamplingParams(
                n=n,
                temperature=0.0 if use_beam_search else 1.0,
                top_p=1.0,
                use_beam_search=use_beam_search,
                ignore_eos=True,
                max_tokens=output_len,
            )

        engine.add_request(str(request_id), inputs, sampling_params)

    out = []
    start = time.perf_counter()
    while engine.has_unfinished_requests():
        request_outputs = engine.step()
        out.append((time.perf_counter(), request_outputs))
    end = time.perf_counter()

    timestamp = {}
    for t, rs in out:
        for r in rs:
            request_id = r.request_id
            if request_id not in timestamp:
                timestamp[request_id] = []
            timestamp[request_id].append(t)

    delay = []
    for v in timestamp.values():
        dd = [v[i]-v[i-1] for i in range(1, len(v))]
        delay.extend(dd)

    delay = np.mean(delay)

    return end - start, delay


def main(args):
    random.seed(args.seed)

    prompt = "hi" * (args.input_len - 1)
    requests = [(prompt, args.input_len, args.output_len)
                for _ in range(args.num_prompts)]

    elapsed_time, delay = run_vllm(
        requests, args.model, args.tokenizer, args.quantization,
        args.tensor_parallel_size, args.seed, args.n, args.use_beam_search,
        args.trust_remote_code, args.dtype, args.max_model_len,
        args.enforce_eager, args.kv_cache_dtype,
        args.quantization_param_path, args.device,
        args.enable_prefix_caching, args.enable_chunked_prefill,
        args.max_num_batched_tokens, args.max_num_seqs, args.distributed_executor_backend,
        args.gpu_memory_utilization, args.download_dir)

    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len in requests)
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s, "
          f"Delay {delay*1000:0.2f} ms")


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
    args.max_model_len = 10000
    args.enforce_eager = False
    args.kv_cache_dtype = "auto"
    args.device = "cuda"
    args.enable_prefix_caching = False
    args.enable_chunked_prefill = True
    args.gpu_memory_utilization = 0.9
    args.output_json = None
    args.distributed_executor_backend = None
    args.download_dir = None

    import sys
    from concurrent.futures import ProcessPoolExecutor

    def run(args):
        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(main, args)
            f.result()

    if "full" in sys.argv:
        for enforce_eager in [True, False]:
            args.enforce_eager = enforce_eager
            print("enforce_eager", enforce_eager)

            print()
            print("enable_chunked_prefill = False")
            for max_num_seqs in [1024, 768, 512, 384, 256, 128, 64, 32]:
                print("max_num_seqs", max_num_seqs)
                args.enable_chunked_prefill = False
                args.max_num_batched_tokens = None
                args.max_num_seqs = max_num_seqs
                run(args)

            print()
            print("enable_chunked_prefill = True")
            for max_num_seqs in [1024, 768, 512, 384, 256, 128, 64, 32]:
                print("max_num_seqs", max_num_seqs)
                args.enable_chunked_prefill = True
                args.max_num_seqs = max_num_seqs
                args.max_num_batched_tokens = args.max_num_seqs
                run(args)
    else:
        args.enforce_eager = True
        args.max_num_seqs = 128
        args.enable_chunked_prefill = False
        args.max_num_batched_tokens = None
        run(args)

        args.enforce_eager = False
        args.max_num_seqs = 256
        args.enable_chunked_prefill = False
        args.max_num_batched_tokens = None
        run(args)

        args.enforce_eager = True
        args.max_num_seqs = 128
        args.enable_chunked_prefill = True
        args.max_num_batched_tokens = args.max_num_seqs
        run(args)

        args.enforce_eager = False
        args.max_num_seqs = 256
        args.enable_chunked_prefill = True
        args.max_num_batched_tokens = args.max_num_seqs
        run(args)
