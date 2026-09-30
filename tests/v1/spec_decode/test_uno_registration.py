# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from typing import get_args

import pytest
import torch

from vllm.config.speculative import SpeculativeMethod


def test_uno_is_a_speculative_method():
    assert "uno" in get_args(SpeculativeMethod)


def test_dispatch_builds_the_uno_speculator(monkeypatch):
    import vllm.v1.worker.gpu.spec_decode as spec_decode
    import vllm.v1.worker.gpu.spec_decode.uno.speculator as uno

    built = []
    monkeypatch.setattr(uno, "UnoSpeculator", lambda config, device: built.append(config) or "speculator")
    config = SimpleNamespace(speculative_config=SimpleNamespace(method="uno"))
    assert spec_decode.init_speculator(config, torch.device("cpu")) == "speculator"
    assert built == [config]


def _config(**overrides):
    parallel = dict(tensor_parallel_size=1, data_parallel_size=1, pipeline_parallel_size=1,
                    prefill_context_parallel_size=1)
    parallel.update({k: v for k, v in overrides.items() if k in parallel})
    return SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode=overrides.get("mamba_cache_mode", "none")),
        parallel_config=SimpleNamespace(**parallel),
        lora_config=overrides.get("lora_config"),
    )


def test_the_supported_configuration_is_accepted():
    from vllm.v1.worker.gpu.spec_decode.uno.speculator import unsupported_reason

    assert unsupported_reason(_config()) is None


@pytest.mark.parametrize("override,word", [
    (dict(mamba_cache_mode="align"), "prefix caching"),
    (dict(tensor_parallel_size=2), "tensor_parallel_size"),
    (dict(data_parallel_size=2), "data_parallel_size"),
    (dict(pipeline_parallel_size=2), "pipeline_parallel_size"),
    (dict(prefill_context_parallel_size=2), "prefill_context_parallel_size"),
    (dict(lora_config=object()), "LoRA"),
])
def test_unsupported_configurations_are_refused_by_name(override, word):
    from vllm.v1.worker.gpu.spec_decode.uno.speculator import unsupported_reason

    assert word in unsupported_reason(_config(**override))
