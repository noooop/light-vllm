import os

os.environ["VLLM_USE_MODELSCOPE"] = "True"


import time
from concurrent.futures import ProcessPoolExecutor


def benchmark(args):
    from vllm import LLMEngine, EngineArgs, SamplingParams, TextPrompt

    import logging
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(logging.ERROR)

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
    )
    engine = LLMEngine.from_engine_args(engine_args)

    for n in range(1, args.num_prompts+2):
        prompt = "hi" * (args.input_len - 1)
        requests = [(prompt, args.input_len, args.output_len)
                    for _ in range(n)]

        for request_id, (prompt, _, output_len) in enumerate(requests):
            inputs = TextPrompt(prompt=prompt)
            sampling_params = SamplingParams(
                    n=n,
                    temperature=0.0 if args.use_beam_search else 1.0,
                    top_p=1.0,
                    use_beam_search=args.use_beam_search,
                    ignore_eos=True,
                    max_tokens=output_len,
                )
            engine.add_request(str(request_id), inputs, sampling_params)

        # prefill & Warming up
        for _ in range(10):
            engine.step()

        start = time.perf_counter()
        for _ in range(128):
            engine.step()
        end = time.perf_counter()

        elapsed_time = end - start
        print(n, f"{elapsed_time:.2f}")

        for request_id, (prompt, _, output_len) in enumerate(requests):
            engine.abort_request(str(request_id))

        while engine.has_unfinished_requests():
            engine.step()


def run(args):
    with ProcessPoolExecutor(1) as executor:
        f = executor.submit(benchmark, args)
        f.result()


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
    args.dtype = 'auto'
    args.max_model_len = 10000
    args.enforce_eager = False
    args.kv_cache_dtype = "auto"
    args.device = "cuda"
    args.enable_prefix_caching = False
    args.gpu_memory_utilization = 0.9
    args.output_json = None
    args.distributed_executor_backend = None
    args.download_dir = None
    args.enable_chunked_prefill = False
    args.max_num_batched_tokens = None
    args.max_num_seqs = 512

    args.num_prompts = 128

    for enforce_eager in [True, False]:
        print("enforce_eager", enforce_eager)
        args.enforce_eager = enforce_eager
        run(args)