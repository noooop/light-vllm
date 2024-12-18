import random
import time

import numpy as np


def benchmark(args):
    random.seed(args.seed)

    from light_vllm import LLMEngine, SamplingParams
    from light_vllm.decoding.arg_utils import ChatEngineArgs as EngineArgs

    engine_args = EngineArgs(
        model=args.model,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=args.kv_cache_dtype,
        quantization_param_path=args.quantization_param_path,
        device=args.device,
        enable_prefix_caching=args.enable_prefix_caching,
        download_dir=args.download_dir,
        enable_chunked_prefill=args.enable_chunked_prefill,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        scheduling=args.scheduling)

    engine = LLMEngine.from_engine_args(engine_args)

    prompt = "hi" * (args.input_len - 1)
    requests = [(prompt, args.input_len, args.output_len)
                for _ in range(args.num_prompts)]

    start = time.perf_counter()
    for request_id, (prompt, _, output_len) in enumerate(requests):
        inputs = prompt
        sampling_params = SamplingParams(
            n=args.n,
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=output_len,
        )
        engine.add_request(str(request_id), inputs, sampling_params)

    out = []
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

    tpot = []
    for v in timestamp.values():
        dd = [v[i] - v[i - 1] for i in range(1, len(v))]
        tpot.extend(dd)

    tpot = np.mean(tpot)
    elapsed_time = end - start

    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len in requests)

    print(f"Throughput: {len(requests) / elapsed_time:.4f} requests/s, "
          f"{total_num_tokens / elapsed_time:.4f} tokens/s, "
          f"Delay {tpot*1000:0.2f} ms")


if __name__ == '__main__':
    from easydict import EasyDict as edict
    args = edict()

    args.dataset = None
    args.input_len = 256
    args.output_len = 16

    args.model = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    args.trust_remote_code = False
    args.tokenizer = args.model
    args.quantization = None
    args.quantization_param_path = None
    args.seed = 0
    args.n = 1
    args.num_prompts = 1000
    args.dtype = 'auto'
    args.max_model_len = 1000

    args.kv_cache_dtype = "auto"
    args.device = "cuda"
    args.enable_prefix_caching = False
    args.gpu_memory_utilization = 0.9
    args.output_json = None
    args.download_dir = None

    import sys
    from concurrent.futures import ProcessPoolExecutor

    def run(args):
        try:
            with ProcessPoolExecutor(1) as executor:
                f = executor.submit(benchmark, args)
                f.result()
        except Exception:
            pass

    if "full" in sys.argv:
        max_num_seqs_list1 = [768, 512, 384, 256, 128, 64, 32, 16, 8]
        max_num_seqs_list2 = [1536, 1024, 768, 512, 384, 256, 128, 64, 32]
    else:
        max_num_seqs_list1 = [256, 128]
        max_num_seqs_list2 = max_num_seqs_list1

    for scheduling in ["sync", "simple_async", "async", "double_buffer"]:
        print(f"scheduling: {scheduling}")
        args.scheduling = scheduling

        print()
        print("enable_chunked_prefill = True")
        for max_num_seqs in max_num_seqs_list2:
            print("max_num_seqs", max_num_seqs)
            args.enable_chunked_prefill = True
            args.max_num_seqs = max_num_seqs
            args.max_num_batched_tokens = max_num_seqs
            run(args)
