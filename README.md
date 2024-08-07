# light-vllm
Does not support multiple machines and multiple GPUs

# Simplify
## step 1
大刀阔斧的删除以下模块
- distributed ray
- adapter_commons prompt_adapter lora 
- multimodal
- spec_decode guided_decoding
- async
- usage metrics

大概删除568个文件，修改47个文件后，终于又可以跑起来最简单的llm推理

benchmarks

Chunked prefill is enabled with max_num_batched_tokens=64

python -m benchmarks.benchmark_chunked_prefill_throughput

简化前
Throughput: 2.45 requests/s, 2504.18 tokens/s

简化后
Throughput: 2.48 requests/s, 2538.07 tokens/s

对贡献这些代码的提交者表示深深的歉意



# Warning
Not rigorously tested.
For research and experimentation only.

Use [vllm](https://github.com/vllm-project/vllm) for production environment


# LICENSE
vllm is licensed under Apache-2.0.
