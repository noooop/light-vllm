import os
import random


def pitch():
    from vllm.worker.model_runner import ModelRunner

    o_prepare_model_input = ModelRunner.prepare_model_input

    log = []

    def p_prepare_model_input(self, *args, **kwargs):
        out = o_prepare_model_input(self, *args, **kwargs)
        log.append((out.attn_metadata.num_prefills,
                    out.attn_metadata.num_prefill_tokens,
                    out.attn_metadata.num_decode_tokens))
        return out

    ModelRunner.prepare_model_input = p_prepare_model_input

    return log


def benchmark(args):
    random.seed(args.seed)

    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
    os.environ["VLLM_NO_USAGE_STATS"] = "True"

    log = pitch()

    import vllm
    from vllm import EngineArgs, LLMEngine, SamplingParams, TextPrompt

    print(vllm.__version__)

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
        disable_log_stats=True)
    engine = LLMEngine.from_engine_args(engine_args)

    prompt = "hi" * (args.input_len - 1)
    requests = [(prompt, args.input_len, args.output_len)
                for _ in range(args.num_prompts)]

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

    i = 0
    while engine.has_unfinished_requests():
        #for i in range(10):
        request_outputs = engine.step()
        print("=" * 80)
        print(i)

        print()
        print(log[-1])
        print([r.request_id for r in request_outputs])

        for seq_group in engine.scheduler[0].running:
            data = seq_group.get_seqs()[0].data
            print(seq_group.request_id, data.get_num_computed_tokens(),
                  data.get_num_uncomputed_tokens())

        i += 1


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

    from concurrent.futures import ProcessPoolExecutor

    def run(args):
        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(benchmark, args)
            f.result()

    max_num_seqs_list = [256]

    print()
    print("enable_chunked_prefill = True")
    for max_num_seqs in max_num_seqs_list:
        print("max_num_seqs", max_num_seqs)
        args.enable_chunked_prefill = True
        args.max_num_seqs = max_num_seqs
        args.max_num_batched_tokens = args.max_num_seqs
        run(args)
