# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-for-bit checks that an Uno draft forward left committed state untouched.

Used only under VLLM_UNO_DEBUG=check. Comparing two runs token by token cannot
test this: on a bf16 target the verify path is not reproducible across
processes at near-ties, so two runs differ even when neither drafts.
"""

import torch

_INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """Reinterpret as integers so NaNs compare equal to themselves."""
    return tensor.contiguous().view(_INT_VIEW[tensor.element_size()])


def changed_blocks(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """`[m, ...]` snapshots of m state blocks -> bool `[m]`, True where a block differs."""
    return (_bits(before) != _bits(after)).flatten(1).any(1)


def kv_block_view(
    kv: torch.Tensor, num_blocks: int, block_size: int
) -> torch.Tensor:
    """View a K/V cache tensor as `[num_blocks, block_size, ...]`."""
    block_dims = [i for i, size in enumerate(kv.shape) if size == num_blocks]
    token_dims = [i for i, size in enumerate(kv.shape) if size == block_size]
    if len(block_dims) != 1 or len(token_dims) != 1:
        raise ValueError(
            f"Cannot tell the K/V cache layout: shape {tuple(kv.shape)} with "
            f"{num_blocks} blocks of {block_size} tokens."
        )
    return kv.movedim((block_dims[0], token_dims[0]), (0, 1))


def changed_slots(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """`[m, block_size, ...]` K/V snapshots -> bool `[m, block_size]`, True where a token slot differs."""
    return (_bits(before) != _bits(after)).flatten(2).any(2)
