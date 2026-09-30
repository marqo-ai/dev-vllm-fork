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


@pytest.mark.parametrize("dim_first", [True, False])
def test_conv_history_is_copied_to_the_scratch_block_at_the_read_offset(dim_first):
    from vllm.v1.worker.gpu.spec_decode.uno.state_tables import stage_conv_windows

    torch.manual_seed(0)
    blocks, dim, history = 8, 5, 3
    state_len = history + WIDTH - 1
    shape = (blocks, dim, state_len) if dim_first else (blocks, state_len, dim)
    conv = torch.randn(shape)
    before = conv.clone()
    src, dst, off = torch.tensor([1, 4]), torch.tensor([2, 6]), torch.tensor([0, 5])

    stage_conv_windows([conv], src, dst, off, WIDTH - 1, history, dim_first)

    def window(tensor, block, start):
        return tensor[block, :, start:start + history] if dim_first else tensor[block, start:start + history]

    assert torch.equal(window(conv, 2, WIDTH - 1), window(before, 1, 0))
    assert torch.equal(window(conv, 6, WIDTH - 1), window(before, 4, 5))
    untouched = [0, 1, 3, 4, 5, 7]
    assert torch.equal(conv[untouched], before[untouched])
    head = (slice(None), slice(0, WIDTH - 1)) if dim_first else (slice(0, WIDTH - 1),)
    assert torch.equal(conv[2][head], before[2][head])


def test_inactive_rows_stage_inside_the_null_block_only():
    from vllm.v1.worker.gpu.spec_decode.uno.state_tables import stage_conv_windows

    conv = torch.randn(4, 5, 11)
    before = conv.clone()
    zero = torch.zeros(2, dtype=torch.int64)
    stage_conv_windows([conv], zero, zero, zero, WIDTH - 1, 3, True)
    assert torch.equal(conv[1:], before[1:])


def test_seed_position_follows_the_last_accepted_row():
    from vllm.v1.worker.gpu.spec_decode.uno.state_tables import draft_seed_positions

    # Two requests verified 9 rows each, at positions 100..108 and 40..48.
    positions = torch.cat([torch.arange(100, 109), torch.arange(40, 49)])
    query_start_loc = torch.tensor([0, 9, 18], dtype=torch.int32)
    # The first rejected 8 rows (one token emitted), the second rejected none.
    num_rejected = torch.tensor([8, 0], dtype=torch.int32)
    assert draft_seed_positions(positions, query_start_loc, num_rejected).tolist() == [101, 49]


def test_draft_rows_past_the_context_limit_write_no_kv():
    from vllm.v1.worker.gpu.spec_decode.uno.state_tables import pad_slots_past_limit

    block, limit, pad = 8, 1664, -1
    slots = torch.arange(100, 100 + 3 * block)
    before = slots.clone()
    # Request 0 ends well short of the limit, request 1 crosses it after 2 rows, request 2 starts on it.
    pad_slots_past_limit(slots, torch.tensor([1000, 1662, 1664]), block, limit, pad)
    assert torch.equal(slots[:block], before[:block])
    assert torch.equal(slots[block:block + 2], before[block:block + 2])
    assert (slots[block + 2:] == pad).all()


def test_mrope_positions_add_each_requests_offset_on_every_axis():
    from vllm.v1.worker.gpu.spec_decode.uno.state_tables import write_mrope_positions

    block = 4
    positions = torch.tensor([10, 11, 12, 13, 50, 51, 52, 53, 0, 0, 0, 0])   # two requests and one padded block
    out = torch.full((3, 16), -7, dtype=torch.int64)
    write_mrope_positions(out, positions, torch.tensor([0, 25], dtype=torch.int32), block)
    for axis in range(3):
        assert out[axis, :12].tolist() == [10, 11, 12, 13, 75, 76, 77, 78, 0, 0, 0, 0]
    assert (out[:, 12:] == -7).all()
