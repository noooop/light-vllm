import time
import random

from light_vllm.utils import STR_DTYPE_TO_TORCH_DTYPE


def benchmark_hf(args):
    random.seed(args.seed)

    import torch
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(args.model, use_fp16=True)

    prompt = "if" * args.input_len
    requests = [prompt for _ in range(args.num_prompts)]

    with torch.no_grad():
        for batchsize in args.batchsize:
            start = time.perf_counter()
            n_step = 0
            for i in range(0, len(requests), batchsize):
                batch = requests[i:i + batchsize]
                output = model.encode(batch, batch_size=batchsize)
                n_step += 1
            end = time.perf_counter()

            elapsed_time = end - start
            delay = elapsed_time / n_step

            print(f"Batchsize {batchsize}, Throughput: {len(requests) / elapsed_time:.4f} requests/s, "
                  f"Delay {delay * 1000:0.2f} ms, n_step {n_step}")


def benchmark_vllm(args):
    random.seed(args.seed)

    from light_vllm import LLMEngine
    from light_vllm.task.encode_only.arg_utils import EncodeOnlyEngineArgs as EngineArgs

    prompt = "if" * args.input_len
    requests = [prompt for _ in range(args.num_prompts)]

    engine_args = EngineArgs(
        model=args.model,
        tokenizer=args.tokenizer,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        device=args.device,
        max_num_seqs=32,
    )

    engine = LLMEngine.from_engine_args(engine_args)

    for batchsize in args.batchsize:
        engine.engine_config.scheduler_config.set_args(max_num_seqs=batchsize)

        start = time.perf_counter()
        for request_id, prompt in enumerate(requests):
            engine.add_request(str(request_id), prompt)

        n_step = 0
        while engine.has_unfinished_requests():
            engine.step()
            n_step += 1
        end = time.perf_counter()

        elapsed_time = end - start
        delay = elapsed_time / n_step

        print(f"Batchsize {batchsize}, Throughput: {len(requests) / elapsed_time:.4f} requests/s, "
              f"Delay {delay * 1000:0.2f} ms, n_step {n_step}")


if __name__ == '__main__':
    from easydict import EasyDict as edict
    args = edict()

    args.input_len = 256
    args.num_prompts = 10000

    args.model = 'BAAI/bge-m3'

    args.trust_remote_code = False
    args.tokenizer = args.model
    args.seed = 0
    args.max_model_len = None
    args.dtype = "half"
    args.device = "cuda"
    args.batchsize = [1, 2, 4, 8, 16, 32, 64]


    from concurrent.futures import ProcessPoolExecutor

    def run_hf(args):
        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(benchmark_hf, args)
            f.result()

    #run_hf(args)

    def run_vllm(args):
        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(benchmark_vllm, args)
            f.result()

    run_vllm(args)