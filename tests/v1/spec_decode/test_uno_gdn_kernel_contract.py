# SPDX-License-Identifier: Apache-2.0
"""The two kernel behaviours the Uno draft forward relies on (GPU only)."""
import pytest
import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.v1.worker.gpu.spec_decode.uno.state_tables import (
    build_draft_state_table,
    stage_conv_windows,
)

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
    stage_conv_windows([draft_conv], src, dst, off, WIDTH - 1, hist, True)
    draft_out = causal_conv1d_update(
        x.clone(), draft_conv, weight, bias, "silu", conv_state_indices=table[:, 0].contiguous(),
        num_accepted_tokens=torch.full((N,), WIDTH, dtype=torch.int32, device=DEV),
        query_start_loc=qsl, max_query_len=WIDTH)

    assert torch.equal(draft_out, stock_out)
    changed = (draft_conv != conv).flatten(1).any(1)
    expected = torch.zeros(slots, dtype=torch.bool, device=DEV)
    expected[dst] = True
    assert torch.equal(changed, expected)


def test_an_inactive_request_touches_no_state_and_leaves_the_others_exact():
    torch.manual_seed(0)
    H, HV, K, V = 4, 8, 16, 16
    real = torch.arange(1, 1 + N * WIDTH, dtype=torch.int32, device=DEV).view(N, WIDTH)
    acc = torch.tensor([1, 4, WIDTH], dtype=torch.int32, device=DEV)
    active = torch.tensor([True, False, True], device=DEV)
    table, _, dst, _ = build_draft_state_table(real, acc, active, WIDTH)
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
    stock_out, _ = fused_sigmoid_gating_delta_rule_update(
        **inputs, initial_state=state.clone(), ssm_state_indices=real.contiguous(), num_accepted_tokens=acc)
    draft_state = state.clone()
    draft_out, _ = fused_sigmoid_gating_delta_rule_update(
        **inputs, initial_state=draft_state, ssm_state_indices=table.contiguous(),
        num_accepted_tokens=torch.full((N,), WIDTH, dtype=torch.int32, device=DEV))

    rows = torch.arange(tokens, device=DEV).view(N, W)[active].reshape(-1)
    assert torch.equal(draft_out[0, rows], stock_out[0, rows])
    changed = (draft_state != state).flatten(1).any(1)
    expected = torch.zeros(slots, dtype=torch.bool, device=DEV)
    expected[dst[active]] = True
    assert torch.equal(changed, expected)


FUSED_W = 7                     # a table of at most 8 columns selects the fused CUDA kernel
FUSED_WIDTH = FUSED_W + 1


@pytest.mark.skipif(not hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp"), reason="no fused GDN kernel in this build")
@pytest.mark.parametrize("accepted", [[1, 1, 1], [1, 4, FUSED_WIDTH], [2, FUSED_WIDTH, 3]])
def test_fused_cuda_kernel_reads_the_committed_block_and_never_writes_it(accepted):
    import vllm._custom_ops as ops

    torch.manual_seed(0)
    H, HV, D = 2, 4, 128
    real = torch.arange(1, 1 + N * FUSED_WIDTH, dtype=torch.int32, device=DEV).view(N, FUSED_WIDTH)
    acc = torch.tensor(accepted, dtype=torch.int32, device=DEV)
    table, _, dst, _ = build_draft_state_table(real, acc, torch.ones(N, dtype=torch.bool, device=DEV), FUSED_WIDTH)
    slots, tokens = 1 + N * FUSED_WIDTH, N * FUSED_W
    state = torch.randn(slots, HV, D, D, device=DEV)
    bf16 = dict(device=DEV, dtype=torch.bfloat16)
    inputs = dict(
        mixed_qkv=torch.randn(tokens, 2 * H * D + HV * D, **bf16),
        a=torch.randn(tokens, HV, **bf16), b=torch.randn(tokens, HV, **bf16),
        A_log=torch.randn(HV, device=DEV), dt_bias=torch.randn(HV, device=DEV),
        cu_seqlens=(torch.arange(N + 1, device=DEV) * FUSED_W).to(torch.int32),
        output_gate=torch.randn(tokens, HV, D, **bf16), norm_weight=torch.ones(D, device=DEV),
    )

    def run(indices, num_accepted, state_tensor):
        out = torch.zeros(tokens, HV, D, **bf16)
        ops.fused_gdn_decode_post_conv_mtp(
            **inputs, state_indices=indices.contiguous(), num_accepted_tokens=num_accepted, state=state_tensor, out=out)
        return out

    stock_out = run(real, acc, state.clone())
    draft_state = state.clone()
    draft_out = run(table, torch.full((N,), FUSED_WIDTH, dtype=torch.int32, device=DEV), draft_state)

    assert torch.equal(draft_out, stock_out)
    changed = (draft_state != state).flatten(1).any(1)
    expected = torch.zeros(slots, dtype=torch.bool, device=DEV)
    expected[dst] = True
    assert torch.equal(changed, expected)
