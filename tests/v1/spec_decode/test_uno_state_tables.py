# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.uno.state_tables import build_draft_state_table

WIDTH = 9  # K = 8


def _real(n):
    return torch.arange(1, 1 + n * WIDTH, dtype=torch.int32).view(n, WIDTH)


@pytest.mark.parametrize("accepted", list(range(1, WIDTH + 1)))
def test_committed_block_sits_in_the_last_column(accepted):
    real = _real(1)
    table, _, _, _ = build_draft_state_table(real, torch.tensor([accepted]), torch.tensor([True]), WIDTH)
    assert table[0, WIDTH - 1].item() == real[0, accepted - 1].item()


@pytest.mark.parametrize("accepted", list(range(1, WIDTH + 1)))
def test_scratch_is_neither_block_zero_nor_the_committed_block(accepted):
    real = _real(1)
    table, _, dst, _ = build_draft_state_table(real, torch.tensor([accepted]), torch.tensor([True]), WIDTH)
    scratch = table[0, 0].item()
    assert scratch != real[0, 0].item()
    assert scratch != real[0, accepted - 1].item()
    assert scratch in real[0].tolist()
    assert dst[0].item() == scratch


def test_only_the_first_and_last_columns_are_written_or_read():
    table, _, _, _ = build_draft_state_table(_real(2), torch.tensor([1, 5]), torch.ones(2, dtype=torch.bool), WIDTH)
    assert torch.count_nonzero(table[:, 1:WIDTH - 1]).item() == 0


def test_conv_staging_reads_block_zero_at_the_accepted_offset():
    real = _real(2)
    _, src, _, off = build_draft_state_table(real, torch.tensor([1, 5]), torch.ones(2, dtype=torch.bool), WIDTH)
    assert src.tolist() == real[:, 0].tolist()
    assert off.tolist() == [0, 4]


def test_inactive_rows_are_null_everywhere():
    table, src, dst, off = build_draft_state_table(_real(2), torch.tensor([3, 3]), torch.tensor([True, False]), WIDTH)
    assert torch.count_nonzero(table[1]).item() == 0
    assert (src[1].item(), dst[1].item(), off[1].item()) == (0, 0, 0)


def test_blocks_wider_than_the_table_are_ignored():
    real = torch.arange(1, 13, dtype=torch.int32).view(1, 12)
    table, _, _, _ = build_draft_state_table(real, torch.tensor([9]), torch.tensor([True]), WIDTH)
    assert table[0, WIDTH - 1].item() == real[0, 8].item()


def test_a_table_too_narrow_for_a_scratch_block_is_rejected():
    with pytest.raises(ValueError, match="num_speculative_tokens >= 2"):
        build_draft_state_table(torch.ones(1, 2, dtype=torch.int32), torch.tensor([1]), torch.tensor([True]), 2)
