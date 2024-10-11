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

# step 2 Refactor
...

# step 3 Modularization + Workflow
将工程拆分成可以即插即用的模型，并通过Workflow配置

# step 4 Workflow Defined Engine
为不同架构的模型实现不同的模块，并按需加载所需的模块。
我将这种架构称为“工作流定义引擎” Workflow Defined Engine，简称为“WDE”。

# step 5 支持 prefill only models
请移步 [[RFC]: Support encode only models by Workflow Defined Engine](https://github.com/vllm-project/vllm/issues/8453)

# step 6 全部接入Workflow Defined Engine
将所有东西移入wde文件夹，将所有东西移出wde文件夹，向wde致敬，删除wde文件夹

暂时看起来不错

# 警告
这只是我个人实验（写着玩的）项目，快速测试各种想法

未经严格测试

我会把成熟功能提交到vllm仓库

生产环境请使用[vllm](https://github.com/vllm-project/vllm)

# Warning
This is just my personal experiment project to quickly test various ideas

Not rigorously tested

I will submit tested features to the vllm

Use [vllm](https://github.com/vllm-project/vllm) for production environment


# LICENSE
vllm is licensed under Apache-2.0.
