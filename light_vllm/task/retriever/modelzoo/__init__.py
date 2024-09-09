
TASK = "retriever"
WORKFLOW = "light_vllm.task.encode_only.workflow:EncodeOnlyWorkflow"

# Architecture -> (task, module, class, workflow).
RETRIEVER_MODELS = {
    "XLMRobertaModel": (TASK, "bgem3", "BGEM3Model", WORKFLOW),
}


