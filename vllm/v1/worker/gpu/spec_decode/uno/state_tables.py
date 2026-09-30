# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""State-slot tables for the Uno draft forward on a hybrid (GDN) target.

A request owns `width = 1 + num_speculative_tokens` state blocks per GDN group.
After a verify step that accepted `a` rows, the committed recurrent state is in
block column `a - 1`, and the committed convolution window is in block column 0
at token offset `a - 1`.

The draft forward runs the same kernels as verify, with a table that makes the
committed state read-only:

    column 0          a scratch block: the convolution kernel reads and rewrites it
    columns 1..K-1    null: the recurrent kernel skips writes to block 0
    column K          the committed block

and with `num_accepted_tokens = width` for every request, so the recurrent
kernel reads its starting state from column K. The draft has K rows, so it
writes columns 0..K-1 only and never the committed block.
"""

import torch


def build_draft_state_table(
    real_table: torch.Tensor,
    num_accepted: torch.Tensor,
    active: torch.Tensor,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(table, conv_src_block, conv_dst_block, conv_src_offset)`.

    Before the draft forward, the caller copies the convolution history
    `conv_src_block[conv_src_offset : conv_src_offset + kernel - 1]` into
    `conv_dst_block[width - 1 : width - 1 + kernel - 1]`.
    Inactive rows get null blocks everywhere.
    """
    if width < 3:
        raise ValueError(
            "Uno needs a scratch state block besides block 0 and the committed "
            "block: num_speculative_tokens >= 2."
        )
    num_reqs = real_table.shape[0]
    rows = torch.arange(num_reqs, device=real_table.device)
    blocks = real_table[:, :width].to(torch.int64)
    committed_col = (num_accepted.to(torch.int64) - 1).clamp(0, width - 1)
    scratch_col = torch.where(committed_col == 1, 2, 1)
    committed = blocks[rows, committed_col]
    scratch = blocks[rows, scratch_col]

    table = torch.zeros(num_reqs, width, dtype=torch.int32, device=real_table.device)
    table[:, 0] = scratch.to(torch.int32)
    table[:, width - 1] = committed.to(torch.int32)
    table[~active] = 0

    zero = torch.zeros_like(committed)
    return (
        table,
        torch.where(active, blocks[:, 0], zero),
        torch.where(active, scratch, zero),
        torch.where(active, committed_col, zero),
    )


def stage_conv_windows(
    conv_states: list[torch.Tensor],
    src_block: torch.Tensor,
    dst_block: torch.Tensor,
    src_offset: torch.Tensor,
    dst_offset: int,
    history: int,
    dim_first: bool,
) -> None:
    """Copy each request's committed convolution history into its scratch block.

    `conv_states` holds one tensor per GDN layer, `[blocks, dim, state_len]` when
    `dim_first`, else `[blocks, state_len, dim]`. Entries
    `[src_offset, src_offset + history)` of `src_block` go to entries
    `[dst_offset, dst_offset + history)` of `dst_block`, which is where the
    convolution kernel reads when `num_accepted_tokens = dst_offset + 1`.
    """
    steps = torch.arange(history, device=src_block.device)
    src_tokens = src_offset[:, None] + steps[None, :]
    dst_tokens = (dst_offset + steps)[None, :]
    for conv in conv_states:
        if dim_first:
            conv[dst_block[:, None], :, dst_tokens] = conv[src_block[:, None], :, src_tokens]
        else:
            conv[dst_block[:, None], dst_tokens] = conv[src_block[:, None], src_tokens]
