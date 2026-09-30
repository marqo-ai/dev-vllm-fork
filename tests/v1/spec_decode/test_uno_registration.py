# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from typing import get_args

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
