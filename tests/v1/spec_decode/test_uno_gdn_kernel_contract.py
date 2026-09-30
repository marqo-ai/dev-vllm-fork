# SPDX-License-Identifier: Apache-2.0
"""The two kernel behaviours the Uno draft forward relies on (GPU only)."""
import pytest
import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.v1.worker.gpu.spec_decode.uno.state_tables import build_draft_state_table

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

N, W = 3, 8
WIDTH = W + 1
DEV = "cuda"


def _tables(accepted):
    real = torch.arange(1, 1 + N * WIDTH, dtype=torch.int32, device=DEV).view(N, WIDTH)
    accepted = torch.tensor(accepted, dtype=torch.int32, device=DEV)
    active = torch.ones(N, dtype=torch.bool, device=DEV)
    return real, accepted, build_draft_state_table(real, accepted, active, WIDTH)


@pytest.mark.parametrize("accepted", [[1, 1, 1], [1, 4, WIDTH], [2, WIDTH, 3]])
def test_recurrent_kernel_reads_the_committed_block_and_never_writes_it(accepted):
    torch.manual_seed(0)
    H, HV, K, V = 4, 8, 16, 16
    real, acc, (table, _, dst, _) = _tables(accepted)
    slots, tokens = 1 + N * WIDTH, N * W
    state = torch.randn(slots, HV, V, K, device=DEV)
    inputs = dict(
        A_log=torch.randn(HV, device=DEV), dt_bias=torch.randn(HV, device=DEV),
        a=torch.randn(tokens, HV, device=DEV), b=torch.randn(tokens, HV, device=DEV),
        q=torch.randn(1, tokens, H, K, device=DEV), k=torch.randn(1, tokens, H, K, device=DEV),
        v=torch.randn(1, tokens, HV, V, device=DEV),
        cu_seqlens=(torch.arange(N + 1, device=DEV) * W).to(torch.int32),
        inplace_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    stock_state = state.clone()
    stock_out, _ = fused_sigmoid_gating_delta_rule_update(
        **inputs, initial_state=stock_state, ssm_state_indices=real.contiguous(), num_accepted_tokens=acc)

    draft_state = state.clone()
    draft_out, _ = fused_sigmoid_gating_delta_rule_update(
        **inputs, initial_state=draft_state, ssm_state_indices=table.contiguous(),
        num_accepted_tokens=torch.full((N,), WIDTH, dtype=torch.int32, device=DEV))

    assert torch.equal(draft_out, stock_out)
    changed = (draft_state != state).flatten(1).any(1)
    expected = torch.zeros(slots, dtype=torch.bool, device=DEV)
    expected[dst] = True
    assert torch.equal(changed, expected)


@pytest.mark.parametrize("accepted", [[1, 1, 1], [1, 4, WIDTH], [2, WIDTH, 3]])
def test_conv_kernel_reads_the_staged_window_and_never_writes_block_zero(accepted):
    torch.manual_seed(0)
    dim, kernel = 64, 4
    hist, state_len = kernel - 1, kernel - 1 + W
    real, acc, (table, src, dst, off) = _tables(accepted)
    slots, tokens = 1 + N * WIDTH, N * W
    conv = torch.randn(slots, dim, state_len, device=DEV)
    weight, bias = torch.randn(dim, kernel, device=DEV), torch.randn(dim, device=DEV)
    x = torch.randn(tokens, dim, device=DEV)
    qsl = (torch.arange(N + 1, device=DEV) * W).to(torch.int32)

    stock_conv = conv.clone()
    stock_out = causal_conv1d_update(
        x.clone(), stock_conv, weight, bias, "silu", conv_state_indices=real[:, 0].contiguous(),
        num_accepted_tokens=acc, query_start_loc=qsl, max_query_len=WIDTH)

    draft_conv = conv.clone()
    steps = torch.arange(hist, device=DEV)
    window = draft_conv[src[:, None], :, off[:, None] + steps[None, :]]
    draft_conv[dst[:, None], :, (WIDTH - 1) + steps[None, :]] = window
    draft_out = causal_conv1d_update(
        x.clone(), draft_conv, weight, bias, "silu", conv_state_indices=table[:, 0].contiguous(),
        num_accepted_tokens=torch.full((N,), WIDTH, dtype=torch.int32, device=DEV),
        query_start_loc=qsl, max_query_len=WIDTH)

    assert torch.equal(draft_out, stock_out)
    changed = (draft_conv != conv).flatten(1).any(1)
    expected = torch.zeros(slots, dtype=torch.bool, device=DEV)
    expected[dst] = True
    assert torch.equal(changed, expected)
