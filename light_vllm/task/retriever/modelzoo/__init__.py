
TASK = "retriever"
WORKFLOW = "light_vllm.task.retriever.workflow:RetrieverWorkflow"

# Architecture -> (task, module, class, workflow).
RETRIEVER_MODELS = {
    "XLMRobertaModel": (TASK, "bgem3", "BGEM3Model", WORKFLOW),
}


