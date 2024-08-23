
import os
import random
import numpy as np
import time


def profiler(cls, method):
    metrics = {"count": 0., "total_elapsed_time": 0.}

    if method == "__call__":
        m = type(cls).__call__

        def func(self, *args, **kwargs):
            start = time.perf_counter()
            o = m(self, *args, **kwargs)
            end = time.perf_counter()
            metrics["total_elapsed_time"] += end - start
            metrics["count"] += 1
            return o

        type(cls).__call__ = func
    else:
        m = getattr(cls, method)

        def func(*args, **kwargs):
            start = time.perf_counter()
            o = m(*args, **kwargs)
            end = time.perf_counter()
            metrics["total_elapsed_time"] += end - start
            metrics["count"] += 1
            return o

        setattr(cls, method, func)
    return metrics


def benchmark(args):
    random.seed(args.seed)

    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
    os.environ["VLLM_NO_USAGE_STATS"] = "True"

    import light_vllm
    from light_vllm import LLMEngine, EngineArgs, SamplingParams, TextPrompt
    print("light_vllm:", light_vllm.__version__)

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

    request_processor_profiler = profiler(engine.request_processor, "__call__")
    model_pre_processor_profiler = profiler(engine.model_pre_processor, "__call__")
    executor_profiler = profiler(engine.executor, "execute_model")
    execute_worker_profiler = profiler(engine.executor.driver_worker, "execute_worker")
    execute_model_profiler = profiler(engine.executor.driver_worker.model_runner, "execute_model")
    model_output_processor_profiler = profiler(engine.output_processor, "__call__")
    scheduler_profiler = profiler(engine.scheduler, "schedule")

    prompt = "hi" * (args.input_len - 1)
    requests = [(prompt, args.input_len, args.output_len)
                for _ in range(args.num_prompts)]

    start = time.perf_counter()
    for request_id, (prompt, _, output_len) in enumerate(requests):
        inputs = TextPrompt(prompt=prompt)
        sampling_params = SamplingParams(
            n=args.n,
            temperature=0.0 if args.use_beam_search else 1.0,
            top_p=1.0,
            use_beam_search=args.use_beam_search,
            ignore_eos=True,
            max_tokens=output_len,
        )
        engine.add_request(str(request_id), inputs, sampling_params)

    out = []
    n_step = 0
    while engine.has_unfinished_requests():
        n_step += 1
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

    tpot = []
    for v in timestamp.values():
        dd = [v[i]-v[i-1] for i in range(1, len(v))]
        tpot.extend(dd)

    tpot = np.mean(tpot)
    elapsed_time = end - start

    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len in requests)

    print(f"Throughput: {len(requests) / elapsed_time:.4f} requests/s, "
          f"{total_num_tokens / elapsed_time:.4f} tokens/s, "
          f"Delay {tpot*1000:0.4f} ms, n_step {n_step}, elapsed_time: {elapsed_time}")
    print("scheduler_profiler", scheduler_profiler["total_elapsed_time"],
          scheduler_profiler["count"])
    print("request_processor_profiler", request_processor_profiler["total_elapsed_time"],
          request_processor_profiler["count"])
    print("model_pre_processor_profiler", model_pre_processor_profiler["total_elapsed_time"],
          model_pre_processor_profiler["count"])
    print("executor_profiler", executor_profiler["total_elapsed_time"],
          executor_profiler["count"])
    print("execute_worker_profiler", execute_worker_profiler["total_elapsed_time"],
          execute_worker_profiler["count"])
    print("execute_model_profiler", execute_model_profiler["total_elapsed_time"],
          execute_model_profiler["count"])
    print("model_output_processor_profiler", model_output_processor_profiler["total_elapsed_time"],
          model_output_processor_profiler["count"])


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

    print()
    print("enable_chunked_prefill = False")
    for max_num_seqs in max_num_seqs_list:
        print("max_num_seqs", max_num_seqs)
        args.enable_chunked_prefill = False
        args.max_num_batched_tokens = None
        args.max_num_seqs = max_num_seqs
        run(args)

    print()
    print("enable_chunked_prefill = True")
    for max_num_seqs in max_num_seqs_list:
        print("max_num_seqs", max_num_seqs)
        args.enable_chunked_prefill = True
        args.max_num_seqs = max_num_seqs
        args.max_num_batched_tokens = args.max_num_seqs
        run(args)
