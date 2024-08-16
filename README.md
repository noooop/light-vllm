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

# step 2 Refactor

## Cuda Graph
把Cuda Graph相关的，散落在各处的Capture Graph，检测是否能使用Cuda Graph，输入做pad，运行Graph等逻辑重构到[一个文件](https://github.com/noooop/light-vllm/blob/main/vllm/worker/cuda_graph_util.py)

使用Cuda Graph捕捉的是静态图，所有tenser尺寸和显存位置都是写死的，要一个batch都是decode阶段才行，不能在prefill阶段包括chunked_prefill使用，[条件还是挺苛刻的](https://github.com/noooop/light-vllm/blob/main/vllm/worker/cuda_graph_util.py#L248)。

python -m benchmarks.benchmark_cuda_graph

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/light-vllm/cuda_graph.png?raw=true" width="400">

实测多个版本的vllm，Qwen2-7B-Instruct使用Cuda Graph，decode延迟降低1%~5%，行吧。[LLaMA-7B 2.3x speedup](https://fireworks.ai/blog/speed-python-pick-two-how-cuda-graphs-enable-fast-python-code-for-deep-learning#llama27b--cuda-graph-inference-performance-results)不知道怎么做到的，可能是30 tokens/sec真的很慢，baseline比较低。

[This (CUDA graph) is particularly effective for small models and when using tensor parallelism.](https://github.com/vllm-project/vllm/pull/1926) 有时间再测一下

Capture Graph需要额外的时间，Graph占额外的显存空间。如果真有巨大的速度提升，调度器也会围绕这个特性，尽量一个batch都是decode阶段，这时chunked_prefill就很尴尬。实测几乎没有提升，Cuda Graph默认关了吧。

# Warning
Not rigorously tested.
For research and experimentation only.

Use [vllm](https://github.com/vllm-project/vllm) for production environment


# LICENSE
vllm is licensed under Apache-2.0.
