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


def state_allowed(real: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor:
    """`[n, width]` state blocks -> flat bool: which blocks the draft may change.

    Only each request's scratch block, and the null block.
    """
    return ((real == scratch[:, None]) | (real == 0)).reshape(-1)


def kv_blocks_and_allowed(
    table: torch.Tensor,
    seed_pos: torch.Tensor,
    block: int,
    block_size: int,
    max_model_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Which K/V blocks to watch, and which of their token slots the draft may change.

    Returns the flat block ids `[n * cols]` of every block that holds a
    request's prefix or draft rows, and a bool `[n * cols, block_size]` that is
    True only at the draft's own positions below the context limit. Columns
    past a request's last draft row can hold stale ids and are replaced by the
    null block, which may change freely.
    """
    num_reqs = table.shape[0]
    device = table.device
    positions = seed_pos[:, None] + torch.arange(block, device=device)[None, :]
    writable = positions < max_model_len
    last_col = torch.clamp(seed_pos + block - 1, max=max_model_len - 1) // block_size
    num_cols = min(int(last_col.max().item()) + 1, table.shape[1])
    cols = torch.arange(num_cols, device=device)
    owned = cols[None, :] <= last_col[:, None]
    blocks = torch.where(owned, table[:, :num_cols].to(torch.int64), 0).reshape(-1)
    rows = torch.arange(num_reqs, device=device)[:, None].expand_as(positions)
    inside = writable & (positions // block_size < num_cols)
    allowed = torch.zeros(num_reqs, num_cols, block_size, dtype=torch.bool, device=device)
    allowed[rows[inside], (positions // block_size)[inside], (positions % block_size)[inside]] = True
    return blocks, allowed.reshape(-1, block_size) | (blocks == 0)[:, None]
