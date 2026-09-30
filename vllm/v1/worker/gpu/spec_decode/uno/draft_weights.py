# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-bit copies of the target's linear weights for the Uno draft forward.

The draft forward is a second full-depth pass over a handful of rows, so its
cost is the bytes of weight it reads. These copies are read by that pass only,
through vLLM's Marlin kernels; the target keeps its own weights for prefill
and verification, so the output stays the target's.

Quantisation is round-to-nearest from the loaded weights, with no calibration:
    fp8      float8_e4m3fn, one scale per output channel, half the bytes
    int4     signed 4-bit, one scale per 128 inputs per output channel, a quarter
    int4g32  the same with one scale per 32 inputs (4.5 bits per weight)
"""

from collections.abc import Callable

import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear,
    marlin_make_workspace_new,
    marlin_permute_scales,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    marlin_quant_fp8_torch,
)
from vllm.scalar_type import scalar_types

KINDS = ("fp8", "int4", "int4g32")
INT4_GROUPS = {"int4": 128, "int4g32": 32}


def _marlin_int4(weight: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack an [N, K] weight as Marlin uint4b8 with one scale per `group` inputs."""
    size_n, size_k = weight.shape
    grouped = weight.float().view(size_n, size_k // group, group)
    scale = torch.maximum(grouped.amax(-1) / 7, -grouped.amin(-1) / 8).clamp_(min=1e-8)
    codes = torch.round(grouped / scale.unsqueeze(-1)).clamp_(-8, 7).to(torch.int64) + 8
    # GPTQ layout: [K / 8, N] int32, input k = 8 i + j in nibble j of row i.
    codes = codes.view(size_n, size_k).t().reshape(size_k // 8, 8, size_n)
    shifts = (4 * torch.arange(8, device=weight.device)).view(1, 8, 1)
    packed = (codes << shifts).sum(1)
    packed = torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)
    qweight = ops.gptq_marlin_repack(packed.contiguous(), size_k, size_n, 4)
    scales = marlin_permute_scales(scale.t().contiguous().to(weight.dtype), size_k, size_n, group)
    return qweight, scales


def draft_linear(
    weight: torch.Tensor, kind: str, fp32_reduce: bool = True
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return x -> x @ W^T computed from a `kind` copy of the [out, in] weight W.

    The output has `out` rounded up to a multiple of 64 columns. `fp32_reduce`
    is the kernels' own switch: sum the partial products of a long input in
    fp32 (vLLM's default) or in the model dtype.
    """
    if kind not in KINDS:
        raise ValueError(f"Uno draft weights must be one of {KINDS}; got {kind!r}.")
    weight = weight.detach()
    if weight.shape[1] % 128:
        raise ValueError(
            f"Uno draft weights need in % 128 == 0; got {weight.shape[0]} x {weight.shape[1]}."
        )
    # The kernels work on 64 output channels at a time. Pad by repeating the
    # last row (an all-zero row has no fp8 scale); the caller gets the padded
    # output and takes the columns it knows.
    pad = -weight.shape[0] % 64
    if pad:
        weight = torch.cat([weight, weight[-1:].expand(pad, -1)], dim=0)
    size_n, size_k = weight.shape
    workspace = marlin_make_workspace_new(weight.device)

    if kind == "fp8":
        _, qweight, scales = marlin_quant_fp8_torch(weight, -1)

        def apply(x: torch.Tensor) -> torch.Tensor:
            return apply_fp8_marlin_linear(
                x, qweight, scales, workspace, size_n, size_k, None, use_fp32_reduce=fp32_reduce
            )

        return apply

    qweight, scales = _marlin_int4(weight, INT4_GROUPS[kind])
    no_zero_points = torch.empty(0, dtype=torch.int, device=weight.device)

    def apply(x: torch.Tensor) -> torch.Tensor:
        return apply_gptq_marlin_linear(
            x, qweight, scales, no_zero_points, workspace, scalar_types.uint4b8, size_n, size_k,
            use_fp32_reduce=fp32_reduce,
        )

    return apply
