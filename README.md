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

python -m benchmarks.baseline

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

# step 3 Modularization + Workflow

将工程拆分成可以即插即用的模型，并提过Workflow配置

```
抽象 Workflow::

Input(request_id, prompt, params, arrival_time) -> InputProcessor -> Request
scheduler.add_request(request: Request)

engine.step
    Request -> RequestProcessor -> SequenceGroup (lazy RequestProcessor)
    seq_group_metadata_list, scheduler_outputs = scheduler.schedule()

    List[SequenceGroupMetadata], SchedulerOutputs -> ModelPreProcessor -> ExecuteInput

    ExecuteInput -> Executor -> List[ExecuteOutput]

    List[ExecuteOutput] -> OutputProcessor -> RequestOutput
    RequestOutput -> return to downstream
```

定义chat模型的ChatWorkflow

```
class ChatWorkflow(Workflow):
    InputProcessor: str = "light_vllm.task.chat.processor.input_processor:ChatModelInputProcessor"
    RequestProcessor: str = "light_vllm.task.chat.processor.input_processor:ChatModelRequestProcessor"
    OutputProcessor: str = "light_vllm.task.chat.processor.output_processor:ChatModelOutputProcessor"
    ModelPreProcessor: str = "light_vllm.task.chat.processor.model_pre_processor:ChatModelPreProcessor"
    Worker: str = "light_vllm.task.chat.worker.gpu_worker:Worker"
    
    Executor: str = "light_vllm.task.base.executor.gpu_executor:GPUExecutor"
    Scheduler: str = "light_vllm.core.scheduler:Scheduler"
    Tokenizer: str = "light_vllm.inputs.tokenizer:Tokenizer"
```

# step 4 Workflow Defined Engine
为不同架构的模型实现不同的模块，并按需加载所需的模块。
我将这种架构称为“工作流定义引擎” Workflow Defined Engine，简称为“WDE”。

# step 5 支持 prefill only models
请移步 [[RFC]: Support encode only models by Workflow Defined Engine](https://github.com/vllm-project/vllm/issues/8453)


# Warning
Not rigorously tested.
For research and experimentation only.

Use [vllm](https://github.com/vllm-project/vllm) for production environment


# LICENSE
vllm is licensed under Apache-2.0.
