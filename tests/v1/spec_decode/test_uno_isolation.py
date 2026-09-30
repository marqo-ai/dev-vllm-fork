# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.uno.isolation import (
    changed_blocks,
    changed_slots,
    kv_block_view,
)

NUM_BLOCKS, BLOCK_SIZE = 6, 8


def test_untouched_state_blocks_report_no_change():
    state = torch.randn(NUM_BLOCKS, 3, 4)
    before = state.index_select(0, torch.tensor([1, 2, 5])).clone()
    assert changed_blocks(before, state.index_select(0, torch.tensor([1, 2, 5]))).tolist() == [False] * 3


def test_a_written_state_block_is_reported():
    state = torch.randn(NUM_BLOCKS, 3, 4)
    blocks = torch.tensor([1, 2, 5])
    before = state.index_select(0, blocks).clone()
    state[2, 0, 0] += 1
    assert changed_blocks(before, state.index_select(0, blocks)).tolist() == [False, True, False]


def test_nan_state_is_not_reported_as_a_change():
    state = torch.full((NUM_BLOCKS, 3, 4), float("nan"))
    blocks = torch.tensor([0, 1])
    before = state.index_select(0, blocks).clone()
    assert changed_blocks(before, state.index_select(0, blocks)).tolist() == [False, False]


def test_bfloat16_state_is_compared_bit_for_bit():
    state = torch.randn(NUM_BLOCKS, 3, 4).to(torch.bfloat16)
    blocks = torch.tensor([0, 3])
    before = state.index_select(0, blocks).clone()
    state[3, 1, 1] = -state[3, 1, 1]
    assert changed_blocks(before, state.index_select(0, blocks)).tolist() == [False, True]


@pytest.mark.parametrize("shape,block_dim", [((2, NUM_BLOCKS, BLOCK_SIZE, 3, 5), 1), ((NUM_BLOCKS, 2, BLOCK_SIZE, 3, 5), 0)])
def test_kv_view_puts_blocks_then_token_slots_first(shape, block_dim):
    kv = torch.randn(shape)
    view = kv_block_view(kv, NUM_BLOCKS, BLOCK_SIZE)
    assert view.shape[:2] == (NUM_BLOCKS, BLOCK_SIZE)
    index = [0] * kv.ndim
    index[block_dim], index[2] = 4, 6
    assert view[4, 6].flatten()[0] == kv[tuple(index)]


def test_kv_view_rejects_an_ambiguous_layout():
    with pytest.raises(ValueError, match="layout"):
        kv_block_view(torch.randn(2, 8, 8, 3, 5), 8, 8)


def test_changed_slots_names_the_block_and_token_slot():
    kv = torch.randn(2, NUM_BLOCKS, BLOCK_SIZE, 3, 5)
    view = kv_block_view(kv, NUM_BLOCKS, BLOCK_SIZE)
    blocks = torch.tensor([1, 4])
    before = view.index_select(0, blocks).clone()
    kv[1, 4, 6, 2, 0] += 1
    changed = changed_slots(before, view.index_select(0, blocks))
    assert changed.shape == (2, BLOCK_SIZE)
    assert changed.nonzero().tolist() == [[1, 6]]


def test_only_the_scratch_block_and_null_blocks_may_change_state():
    from vllm.v1.worker.gpu.spec_decode.uno.isolation import state_allowed

    real = torch.tensor([[11, 12, 13], [21, 22, 23], [0, 0, 0]])
    allowed = state_allowed(real, torch.tensor([12, 23, 0]))
    assert allowed.tolist() == [False, True, False, False, False, True, True, True, True]


def test_only_the_draft_positions_may_change_kv():
    from vllm.v1.worker.gpu.spec_decode.uno.isolation import kv_blocks_and_allowed

    block, block_size = 4, 8
    table = torch.tensor([[5, 6, 99], [7, 98, 97]])     # columns past a request's last draft row hold stale ids
    # Request 0 drafts positions 6..9 (crossing into its second block); request 1 drafts 2..5.
    blocks, allowed = kv_blocks_and_allowed(table, torch.tensor([6, 2]), block, block_size, 100)
    assert blocks.tolist() == [5, 6, 7, 0]
    assert allowed[0].nonzero().flatten().tolist() == [6, 7]
    assert allowed[1].nonzero().flatten().tolist() == [0, 1]
    assert allowed[2].nonzero().flatten().tolist() == [2, 3, 4, 5]
    assert allowed[3].all()                             # a null block is a free sink


def test_positions_past_the_context_limit_are_not_allowed_to_change_kv():
    from vllm.v1.worker.gpu.spec_decode.uno.isolation import kv_blocks_and_allowed

    block, block_size, limit = 4, 8, 16
    table = torch.tensor([[5, 6]])
    # Drafting 14..17: positions 16 and 17 are past the limit and would wrap into block 6's first slots.
    blocks, allowed = kv_blocks_and_allowed(table, torch.tensor([14]), block, block_size, limit)
    assert blocks.tolist() == [5, 6]
    assert allowed[0].nonzero().numel() == 0
    assert allowed[1].nonzero().flatten().tolist() == [6, 7]
