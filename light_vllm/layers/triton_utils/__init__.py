from light_vllm.layers.triton_utils.importing import HAS_TRITON

__all__ = ["HAS_TRITON"]

if HAS_TRITON:

    from light_vllm.layers.triton_utils.libentry import libentry

    __all__ += ["maybe_set_triton_cache_manager", "libentry"]
