# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn as nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.v1.worker.gpu.spec_decode.uno.draft_norms import DraftNorms

HIDDEN = 32


def _norms(count=3):
    torch.manual_seed(0)
    with set_current_vllm_config(VllmConfig()):
        norms = [GemmaRMSNorm(HIDDEN, eps=1e-6) for _ in range(count)]
    for norm in norms:
        norm.weight.data = torch.randn(HIDDEN) * 0.1
    return norms


def _same(fn):
    return fn


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_draft_norm_is_the_modules_own_norm(dtype):
    norms = _norms()
    for norm in norms:
        norm.weight.data = norm.weight.data.to(dtype)
    x = torch.randn(4, HIDDEN).to(dtype)
    residual = torch.randn(4, HIDDEN).to(dtype)
    want_plain = norms[1](x)
    want_out, want_residual = norms[2](x, residual)
    with DraftNorms(norms, compile_fn=_same).active():
        got_plain = norms[1](x)
        got_out, got_residual = norms[2](x, residual)
    assert got_plain.dtype == dtype and torch.equal(got_plain, want_plain)
    assert torch.equal(got_out, want_out) and torch.equal(got_residual, want_residual)


def test_the_modules_forward_is_back_after_the_context():
    norms = _norms(1)
    calls = []

    def counting(fn):
        def wrapped(*args):
            calls.append(fn.__name__)
            return fn(*args)
        return wrapped

    x = torch.randn(4, HIDDEN)
    draft = DraftNorms(norms, compile_fn=counting)
    with draft.active():
        norms[0](x)
    norms[0](x)
    assert len(calls) == 1


def test_every_norm_shares_two_compiled_functions():
    compiled = []

    def compile_fn(fn):
        compiled.append(fn.__name__)
        return fn

    DraftNorms(_norms(5), compile_fn=compile_fn)
    assert len(compiled) == 2


def test_only_the_gemma_norm_is_taken():
    with pytest.raises(ValueError, match="LayerNorm"):
        DraftNorms([nn.LayerNorm(HIDDEN)], compile_fn=_same)


@pytest.mark.parametrize(
    "mode,backend,want",
    [("VLLM_COMPILE", "inductor", True), ("NONE", "inductor", False), ("VLLM_COMPILE", "eager", False)],
)
def test_norms_are_only_compiled_when_vllm_compiles(mode, backend, want):
    from types import SimpleNamespace

    from vllm.config import CompilationMode
    from vllm.v1.worker.gpu.spec_decode.uno.draft_norms import compilation_enabled

    config = SimpleNamespace(mode=getattr(CompilationMode, mode), backend=backend)
    assert compilation_enabled(config) is want
