# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused RMSNorm for the Uno draft forward.

The decoder's RMSNorm scales by (1 + w) in fp32, which vLLM's C kernel does not
take, so it runs the native fp32 formula. Inside the model's compiled forward
that formula is fused; the draft forward calls the layers directly, where it is
a dozen small kernels per norm, twice per layer. Here the same formula is
compiled on its own, once, and swapped in for the draft forward only.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn

from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.utils import maybe_disable_graph_partition
from vllm.platforms import current_platform


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # ir.ops.rms_norm, with weight = 1 + w already in fp32.
    out = x.to(torch.float32)
    variance = out.pow(2).mean(dim=-1, keepdim=True)
    out = out * torch.rsqrt(variance + eps)
    return (out * weight).to(x.dtype)


def _add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    # ir.ops.fused_add_rms_norm, likewise.
    out = x.to(torch.float32) + residual.to(torch.float32)
    residual = out.to(x.dtype)
    variance = out.pow(2).mean(dim=-1, keepdim=True)
    out = out * torch.rsqrt(variance + eps)
    return (out * weight).to(x.dtype), residual


def _compile(fn: Callable) -> Callable:
    backend = current_platform.simple_compile_backend
    return torch.compile(
        fn, dynamic=True, backend=backend, options=maybe_disable_graph_partition(backend)
    )


class DraftNorms:
    def __init__(
        self, norms: list[nn.Module], compile_fn: Callable[[Callable], Callable] = _compile
    ):
        rms_norm, add_rms_norm = compile_fn(_rms_norm), compile_fn(_add_rms_norm)
        self._forwards: list[tuple[nn.Module, Callable]] = []
        for norm in norms:
            if type(norm) is not GemmaRMSNorm:
                raise ValueError(
                    f"Uno draft norms replace GemmaRMSNorm only; got {type(norm).__name__}."
                )
            weight = norm.weight.detach().float() + 1.0
            self._forwards.append(
                (norm, self._make_forward(rms_norm, add_rms_norm, weight, norm.variance_epsilon))
            )

    @staticmethod
    def _make_forward(rms_norm, add_rms_norm, weight: torch.Tensor, eps: float):
        def forward(x, residual=None):
            if residual is None:
                return rms_norm(x, weight, eps)
            return add_rms_norm(x, residual, weight, eps)

        return forward

    @contextmanager
    def active(self) -> Iterator[None]:
        for norm, forward in self._forwards:
            norm.forward = forward
        try:
            yield
        finally:
            for norm, _ in self._forwards:
                del norm.forward
