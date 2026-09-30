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
