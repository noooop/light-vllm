import pytest

from tests.wde.utils import HfRunner, VllmRunner


@pytest.fixture(scope="session")
def hf_runner():
    return HfRunner


@pytest.fixture(scope="session")
def vllm_runner():
    return VllmRunner
