import dataclasses
import importlib
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import torch

from light_vllm.logger import init_logger
from light_vllm.wde.core.schema.execute_io import ExecuteOutput
from light_vllm.wde.core.schema.execute_io import ExecuteInput, WorkerInput, ModelInput
from light_vllm.wde.core.runner.model_runner_base import ModelRunnerBase
from light_vllm.utils import (enable_trace_function_call_for_thread,
                              update_environment_variables)

logger = init_logger(__name__)


class WorkerBase(ABC):
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.
    """

    @abstractmethod
    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        raise NotImplementedError

    @abstractmethod
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available blocks for the GPU KV cache and
        swappable CPU KV cache.

        The implementation may run profiling or other heuristics to determine
        the size of caches.

        Returns a Tuple[num_gpu_blocks, num_cpu_blocks], where num_gpu_blocks
        are blocks that are "active" on the device and can be appended to.
        num_cpu_blocks refers to "swapped" blocks in CPU memory and cannot be
        appended to.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache with the given size in blocks.
        """
        raise NotImplementedError

    @abstractmethod
    @torch.inference_mode
    def __call__(
        self,
        execute_input: Optional[ExecuteInput] = None
    ) -> Optional[List[ExecuteOutput]]:
        raise NotImplementedError

    @abstractmethod
    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.
        """
        raise NotImplementedError


class LocalOrDistributedWorkerBase(WorkerBase):
    """
    Partial implementation of WorkerBase that has a default `execute_model`
    definition to perform metadata transfer between workers when in distributed
    mode. Subclasses of this interface should use model runners that inherit
    from ModelRunnerBase, and should only need to implement worker-local logic.
    If custom control plane logic is needed to transfer metadata, or if the
    model runner cannot inherit from ModelRunnerBase, use WorkerBase instead.
    """
    is_driver_worker: bool
    model_runner: ModelRunnerBase

    @property
    @abstractmethod
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        """
        Gets the list of kv caches to pass to the worker's model runner. Each
        element in the list is a kv cache corresponding to a particular virtual
        engine (PP stream). Used by the default `execute_model`. If the worker's
        model runner does not follow the ModelRunnerBase interface, then inherit
        from WorkerBase instead.
        """
        raise NotImplementedError

    @abstractmethod
    def execute_worker(self, worker_input: WorkerInput) -> None:
        """
        Process an execution request.
        """
        raise NotImplementedError

    @torch.inference_mode
    def __call__(
        self,
        execute_input: Optional[ExecuteInput] = None
    ) -> Optional[List[ExecuteOutput]]:
        """Executes at least one model step on the given sequences, unless no
        sequences are provided."""

        self.execute_worker(execute_input.worker_input)

        # If there is no input, we don't need to execute the model.
        if execute_input.worker_input.num_seq_groups == 0:
            return []

        output = self.model_runner.execute_model(
            execute_input.model_input,
            self.kv_cache if self.kv_cache is not None else None)

        return output


class WorkerWrapperBase:
    """
    The whole point of this class is to lazily initialize the worker.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.

    If worker_class_fn is specified, it will be executed to get the worker
    class.
    Otherwise, the worker class will be obtained by dynamically importing it
    using worker_module_name and worker_class_name.
    """

    def __init__(
        self,
        worker_module_name: str,
        worker_class_name: str,
        trust_remote_code: bool = False,
        worker_class_fn: Optional[Callable[[],
                                           Type[WorkerBase]]] = None) -> None:
        self.worker_module_name = worker_module_name
        self.worker_class_name = worker_class_name
        self.worker_class_fn = worker_class_fn
        self.worker: Optional[WorkerBase] = None
        if trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from light_vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

    @staticmethod
    def update_environment_variables(envs: Dict[str, str]) -> None:
        key = 'CUDA_VISIBLE_DEVICES'
        if key in envs and key in os.environ:
            # overwriting CUDA_VISIBLE_DEVICES is desired behavior
            # suppress the warning in `update_environment_variables`
            del os.environ[key]
        update_environment_variables(envs)

    def init_worker(self, *args, **kwargs):
        """
        Here we inject some common logic before initializing the worker.
        Arguments are passed to the worker class constructor.
        """
        enable_trace_function_call_for_thread()

        # see https://github.com/NVIDIA/nccl/issues/1234
        os.environ['NCCL_CUMEM_ENABLE'] = '0'

        if self.worker_class_fn:
            worker_class = self.worker_class_fn()
        else:
            mod = importlib.import_module(self.worker_module_name)
            worker_class = getattr(mod, self.worker_class_name)

        self.worker = worker_class(*args, **kwargs)
        assert self.worker is not None

    def execute_method(self, method, *args, **kwargs):
        try:
            target = self if self.worker is None else self.worker
            executor = getattr(target, method)
            return executor(*args, **kwargs)
        except Exception as e:
            # if the driver worker also execute methods,
            # exceptions in the rest worker may cause deadlock in rpc like ray
            # see https://github.com/vllm-project/vllm/issues/3455
            # print the error and inform the user to solve the error
            msg = (f"Error executing method {method}. "
                   "This might cause deadlock in distributed execution.")
            logger.exception(msg)
            raise e
