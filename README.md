# light-vllm
Does not support multiple machines and multiple GPUs

# step 1 Simplify
大刀阔斧的删除以下模块
- distributed ray
- adapter_commons prompt_adapter lora 
- multimodal
- spec_decode guided_decoding
- async
- usage metrics tracing observability

大概删除568个文件，修改47个文件后，终于又可以跑起来最简单的llm推理

benchmarks

python -m python -m benchmarks.baseline

简化前 0.5.4
- Throughput: 4.76 requests/s, 4878.61 tokens/s, Delay 42.62 ms
- Throughput: 4.41 requests/s, 4512.93 tokens/s, Delay 58.77 ms
- Throughput: 4.20 requests/s, 4297.53 tokens/s, Delay 32.85 ms
- Throughput: 4.91 requests/s, 5023.96 tokens/s, Delay 48.38 ms

简化后
- Throughput: 4.85 requests/s, 4967.57 tokens/s, Delay 41.71 ms
- Throughput: 4.99 requests/s, 5113.93 tokens/s, Delay 50.99 ms
- Throughput: 4.25 requests/s, 4351.67 tokens/s, Delay 32.44 ms
- Throughput: 5.35 requests/s, 5474.75 tokens/s, Delay 43.90 ms

对贡献这些代码的提交者表示深深的歉意


# Warning
Not rigorously tested.
For research and experimentation only.

Use [vllm](https://github.com/vllm-project/vllm) for production environment


# LICENSE
vllm is licensed under Apache-2.0.
