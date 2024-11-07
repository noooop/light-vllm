import time

import vllm
from vllm import LLM, SamplingParams

input_len = 256
output_len = 16
num_prompts = 1000

prompt = "hi" * (input_len - 1)
prompts = [prompt for _ in range(num_prompts)]

sampling_params = SamplingParams(
    n=1,
    temperature=1.0,
    top_p=1.0,
    ignore_eos=True,
    max_tokens=output_len,
)

llm = LLM(model="Qwen/Qwen2-7B-Instruct",
          quantization="fp8",
          max_num_seqs=512,
          max_num_batched_tokens=512,
          enable_chunked_prefill=True)

start = time.perf_counter()
llm.generate(prompts, sampling_params)
end = time.perf_counter()
elapsed_time = end - start

print(
    f"vllm: {vllm.__version__} Throughput: {num_prompts / elapsed_time:.4f} requests/s"
)
"""
0.6.1 Throughput: 59.0771 requests/s
vllm: 0.6.1.post1 59.0011 requests/s
vllm: 0.6.1.dev238+ge2c6e0a82 Throughput: 41.5246 requests/s
vllm: 0.6.3 Throughput: 41.3583 requests/s
vllm: 0.6.3.post1 Throughput: 58.6271 requests/s
"""
