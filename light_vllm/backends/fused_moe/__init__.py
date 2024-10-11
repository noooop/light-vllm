from light_vllm.backends.fused_moe.layer import FusedMoE, FusedMoEMethodBase
from light_vllm.backends.triton_utils import HAS_TRITON

__all__ = [
    "FusedMoE",
    "FusedMoEMethodBase",
]

if HAS_TRITON:
    from light_vllm.backends.fused_moe.fused_moe import (fused_experts,
                                                         fused_moe, fused_topk,
                                                         get_config_file_name,
                                                         grouped_topk)

    __all__ += [
        "fused_moe",
        "fused_topk",
        "fused_experts",
        "get_config_file_name",
        "grouped_topk",
    ]
