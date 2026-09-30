# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch
import torch.nn as nn

from vllm.v1.worker.gpu.spec_decode.uno.gated_lora import GatedLoRA, load_uno_adapter

HIDDEN, INTER, RANK, SCALE = 16, 24, 4, 64.0
QKV, Z = 40, 12            # GDN in_proj_qkvz = [qkv | z]
Q, KV = 20, 6              # attention qkv_proj = [q | k | v]


class _Linear(nn.Module):
    """vLLM's linear layers return (output, bias)."""

    def __init__(self, inp, out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out, inp))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight), None


def _holder(**modules):
    holder = nn.Module()
    for name, module in modules.items():
        setattr(holder, name, module)
    return holder


def _layers():
    gdn = _holder(
        linear_attn=_holder(in_proj_qkvz=_Linear(HIDDEN, QKV + Z), out_proj=_Linear(HIDDEN, HIDDEN)),
        mlp=_holder(gate_up_proj=_Linear(HIDDEN, 2 * INTER), down_proj=_Linear(INTER, HIDDEN)))
    attn = _holder(
        self_attn=_holder(qkv_proj=_Linear(HIDDEN, Q + 2 * KV), o_proj=_Linear(HIDDEN, HIDDEN)),
        mlp=_holder(gate_up_proj=_Linear(HIDDEN, 2 * INTER), down_proj=_Linear(INTER, HIDDEN)))
    return nn.ModuleList([gdn, attn])


def _pair(out, inp):
    return torch.randn(RANK, inp), torch.randn(out, RANK)


def _state():
    shapes = {
        "model.layers.0.linear_attn.in_proj_qkv": (QKV, HIDDEN),
        "model.layers.0.linear_attn.out_proj": (HIDDEN, HIDDEN),
        "model.layers.0.mlp.gate_proj": (INTER, HIDDEN),
        "model.layers.0.mlp.up_proj": (INTER, HIDDEN),
        "model.layers.0.mlp.down_proj": (HIDDEN, INTER),
        "model.layers.1.self_attn.q_proj": (Q, HIDDEN),
        "model.layers.1.self_attn.k_proj": (KV, HIDDEN),
        "model.layers.1.self_attn.v_proj": (KV, HIDDEN),
        "model.layers.1.self_attn.o_proj": (HIDDEN, HIDDEN),
        "model.layers.1.mlp.gate_proj": (INTER, HIDDEN),
        "model.layers.1.mlp.up_proj": (INTER, HIDDEN),
        "model.layers.1.mlp.down_proj": (HIDDEN, INTER),
    }
    state = {}
    for name, (out, inp) in shapes.items():
        state[name + ".A"], state[name + ".B"] = _pair(out, inp)
    return state


def _delta(state, name, x):
    return (x @ state[name + ".A"].T) @ state[name + ".B"].T * SCALE


def _mask(rows):
    mask = torch.ones(rows, 1)
    mask[0] = 0            # row 0 is the seed row
    return mask


def _call_all(layers, x, inter):
    layers[0].linear_attn.in_proj_qkvz(x); layers[0].linear_attn.out_proj(x)
    layers[0].mlp.gate_up_proj(x); layers[0].mlp.down_proj(inter)
    layers[1].self_attn.qkv_proj(x); layers[1].self_attn.o_proj(x)
    layers[1].mlp.gate_up_proj(x); layers[1].mlp.down_proj(inter)


def test_delta_lands_on_noise_rows_only_and_on_the_right_columns():
    torch.manual_seed(0)
    layers, state, x = _layers(), _state(), torch.randn(4, HIDDEN)
    lora = GatedLoRA(layers, state, SCALE, _mask(4), torch.device("cpu"))
    base_qkvz, _ = layers[0].linear_attn.in_proj_qkvz(x)
    base_gate_up, _ = layers[1].mlp.gate_up_proj(x)
    base_qkv, _ = layers[1].self_attn.qkv_proj(x)
    with lora.active():
        out_qkvz, _ = layers[0].linear_attn.in_proj_qkvz(x)
        out_gate_up, _ = layers[1].mlp.gate_up_proj(x)
        out_qkv, _ = layers[1].self_attn.qkv_proj(x)

    assert torch.equal(out_qkvz[0], base_qkvz[0])                                    # seed row untouched
    assert torch.equal(out_qkvz[:, QKV:], base_qkvz[:, QKV:])                        # z columns untouched
    want = base_qkvz[1:, :QKV] + _delta(state, "model.layers.0.linear_attn.in_proj_qkv", x[1:])
    assert torch.allclose(out_qkvz[1:, :QKV], want, atol=1e-2)

    want_gate = base_gate_up[1:, :INTER] + _delta(state, "model.layers.1.mlp.gate_proj", x[1:])
    want_up = base_gate_up[1:, INTER:] + _delta(state, "model.layers.1.mlp.up_proj", x[1:])
    assert torch.allclose(out_gate_up[1:, :INTER], want_gate, atol=1e-2)
    assert torch.allclose(out_gate_up[1:, INTER:], want_up, atol=1e-2)

    want_k = base_qkv[1:, Q:Q + KV] + _delta(state, "model.layers.1.self_attn.k_proj", x[1:])
    assert torch.allclose(out_qkv[1:, Q:Q + KV], want_k, atol=1e-2)


def test_nothing_changes_outside_the_context_manager():
    torch.manual_seed(0)
    layers, x = _layers(), torch.randn(4, HIDDEN)
    lora = GatedLoRA(layers, _state(), SCALE, _mask(4), torch.device("cpu"))
    before, _ = layers[0].mlp.gate_up_proj(x)
    with lora.active():
        layers[0].mlp.gate_up_proj(x)
    after, _ = layers[0].mlp.gate_up_proj(x)
    assert torch.equal(before, after)


def test_every_adapted_module_must_be_called():
    layers, x = _layers(), torch.randn(4, HIDDEN)
    lora = GatedLoRA(layers, _state(), SCALE, _mask(4), torch.device("cpu"))
    assert lora.num_modules == 8
    with lora.active():
        _call_all(layers, x, torch.randn(4, INTER))
    lora.check_all_fired()
    with lora.active():
        layers[0].mlp.gate_up_proj(x)
    with pytest.raises(RuntimeError, match="1 of 8"):
        lora.check_all_fired()


def test_unknown_projection_is_rejected():
    state = _state()
    state["model.layers.0.linear_attn.in_proj_z.A"] = torch.randn(RANK, HIDDEN)
    state["model.layers.0.linear_attn.in_proj_z.B"] = torch.randn(Z, RANK)
    with pytest.raises(ValueError, match="in_proj_z"):
        GatedLoRA(_layers(), state, SCALE, _mask(4), torch.device("cpu"))


def test_a_packed_module_must_be_covered_from_its_first_slice():
    state = _state()
    del state["model.layers.1.self_attn.q_proj.A"], state["model.layers.1.self_attn.q_proj.B"]
    with pytest.raises(ValueError, match="self_attn.qkv_proj"):
        GatedLoRA(_layers(), state, SCALE, _mask(4), torch.device("cpu"))


def test_scale_comes_from_the_sidecar(tmp_path):
    path = tmp_path / "adapter-000599.pt"
    torch.save(_state(), path)
    path.with_suffix(".json").write_text(json.dumps({"rank": 4, "alpha": 256, "scale": 64.0}))
    state, scale = load_uno_adapter(str(path))
    assert scale == 64.0 and len(state) == 24


def test_a_missing_sidecar_is_an_error(tmp_path):
    path = tmp_path / "adapter.pt"
    torch.save(_state(), path)
    with pytest.raises(FileNotFoundError):
        load_uno_adapter(str(path))


def test_model_dtype_path_matches_the_fp32_path():
    torch.manual_seed(0)
    layers, state, x = _layers(), _state(), torch.randn(4, HIDDEN)
    exact = GatedLoRA(layers, state, SCALE, _mask(4), torch.device("cpu"))
    fused = GatedLoRA(layers, state, SCALE, _mask(4), torch.device("cpu"), dtype=torch.float32)
    base, _ = layers[1].self_attn.qkv_proj(x)
    with exact.active():
        want, _ = layers[1].self_attn.qkv_proj(x)
    with fused.active():
        got, _ = layers[1].self_attn.qkv_proj(x)
        layers[0].linear_attn.in_proj_qkvz(x)
    assert torch.equal(got[0], base[0])                       # seed row untouched
    assert torch.allclose(got, want, atol=1e-2)
    assert not torch.allclose(got[1:], base[1:], atol=1e-2)   # and the delta is really there
    assert fused.fired == 2


def test_bfloat16_delta_lands_on_the_leading_columns_of_a_wider_module():
    torch.manual_seed(0)
    layers, state = _layers(), _state()
    for module in layers.modules():
        if isinstance(module, _Linear):
            module.weight.data = module.weight.data.to(torch.bfloat16)
    x = torch.randn(4, HIDDEN).to(torch.bfloat16)
    mask = _mask(4).to(torch.bfloat16)
    lora = GatedLoRA(layers, state, SCALE, mask, torch.device("cpu"), dtype=torch.bfloat16)
    base, _ = layers[0].linear_attn.in_proj_qkvz(x)
    with lora.active():
        out, _ = layers[0].linear_attn.in_proj_qkvz(x)
    assert torch.equal(out[0], base[0])                       # seed row untouched
    assert torch.equal(out[:, QKV:], base[:, QKV:])           # z columns untouched
    want = base[1:, :QKV].float() + _delta(state, "model.layers.0.linear_attn.in_proj_qkv", x[1:].float())
    error = (out[1:, :QKV].float() - want).abs().max() / want.abs().max()
    assert error < 0.02


def _exact_base(weight):
    """A draft base that is the plain linear of whatever weight it is handed."""
    return lambda inputs: torch.nn.functional.linear(inputs, weight)


def test_a_draft_only_base_replaces_the_module_forward_inside_the_context_only():
    torch.manual_seed(0)
    layers, state, x = _layers(), _state(), torch.randn(4, HIDDEN)
    module = layers[1].mlp.gate_up_proj
    lora = GatedLoRA(layers, state, SCALE, _mask(4), torch.device("cpu"), dtype=torch.float32,
                     draft_base=lambda weight: (lambda inputs: 2 * torch.nn.functional.linear(inputs, weight)))
    original, _ = module(x)
    with lora.active():
        drafted, bias = module(x)
    after, _ = module(x)

    assert bias is None
    assert torch.equal(after, original)                                  # forward restored
    assert torch.allclose(drafted[0], 2 * original[0], atol=1e-4)        # seed row: draft base, no LoRA
    want = 2 * original[1:, :INTER] + _delta(state, "model.layers.1.mlp.gate_proj", x[1:])
    assert torch.allclose(drafted[1:, :INTER], want, atol=1e-2)
    assert lora.fired == 1


def test_a_draft_only_base_needs_the_model_dtype_lora_path():
    with pytest.raises(ValueError, match="dtype"):
        GatedLoRA(_layers(), _state(), SCALE, _mask(4), torch.device("cpu"), draft_base=_exact_base)


def test_folding_hands_the_base_a_weight_with_the_lora_inputs_appended():
    torch.manual_seed(0)
    layers, state = _layers(), _state()
    shapes = []

    def draft_base(weight):
        shapes.append(tuple(weight.shape))
        return _exact_base(weight)

    GatedLoRA(layers, state, SCALE, _mask(4), torch.device("cpu"), dtype=torch.float32,
              draft_base=draft_base, fold=True)
    # GDN layer: qkvz + 1 projection, out_proj + 1, gate_up + 2, down + 1; attention layer: qkv + 3, ...
    assert shapes == [
        (QKV + Z + RANK, HIDDEN), (HIDDEN + RANK, HIDDEN), (HIDDEN + RANK, INTER), (2 * INTER + 2 * RANK, HIDDEN),
        (HIDDEN + RANK, INTER), (2 * INTER + 2 * RANK, HIDDEN), (HIDDEN + RANK, HIDDEN), (Q + 2 * KV + 3 * RANK, HIDDEN),
    ]


@pytest.mark.parametrize("requests", [1, 2])
def test_a_folded_base_gives_the_same_output_as_the_separate_lora(requests):
    torch.manual_seed(0)
    layers, state = _layers(), _state()
    rows = 4 * requests
    x = torch.randn(rows, HIDDEN)
    mask = torch.ones(rows, 1)
    mask[0::4] = 0
    separate = GatedLoRA(layers, state, SCALE, mask, torch.device("cpu"), dtype=torch.float32)
    folded = GatedLoRA(layers, state, SCALE, mask, torch.device("cpu"), dtype=torch.float32,
                       draft_base=_exact_base, fold=True, block=4)
    for module in (layers[0].linear_attn.in_proj_qkvz, layers[1].self_attn.qkv_proj, layers[1].mlp.down_proj):
        inputs = x if module.weight.shape[1] == HIDDEN else torch.randn(rows, INTER)
        base, _ = module(inputs)
        with separate.active():
            want, _ = module(inputs)
        with folded.active():
            got, _ = module(inputs)
        assert got.shape == base.shape
        assert torch.equal(got[0::4], base[0::4])                 # seed rows untouched
        assert torch.allclose(got, want, atol=1e-2)
        assert not torch.allclose(got[1], base[1], atol=1e-2)


def test_a_single_request_padded_past_the_block_still_leaves_the_seed_row_alone():
    torch.manual_seed(0)
    layers, state, x = _layers(), _state(), torch.randn(5, HIDDEN)     # block 4 + one padding row
    mask = torch.ones(8, 1)
    mask[0::4] = 0
    lora = GatedLoRA(layers, state, SCALE, mask, torch.device("cpu"), dtype=torch.float32,
                     draft_base=_exact_base, block=4)
    module = layers[1].self_attn.o_proj
    base, _ = module(x)
    with lora.active():
        got, _ = module(x)
    assert torch.equal(got[0], base[0])
    want = base[1:4] + _delta(state, "model.layers.1.self_attn.o_proj", x[1:4])
    assert torch.allclose(got[1:4], want, atol=1e-2)


def test_a_module_with_a_bias_cannot_take_a_draft_base():
    layers = _layers()
    layers[0].mlp.down_proj.bias = nn.Parameter(torch.zeros(HIDDEN))
    with pytest.raises(ValueError, match="bias"):
        GatedLoRA(layers, _state(), SCALE, _mask(4), torch.device("cpu"), dtype=torch.float32,
                  draft_base=_exact_base)


@pytest.mark.parametrize("fold", [False, True])
def test_batches_above_max_rows_read_the_modules_own_weight(fold):
    torch.manual_seed(0)
    layers, state = _layers(), _state()
    mask = torch.ones(8, 1)
    mask[0::4] = 0
    doubled = lambda weight: (lambda inputs: 2 * torch.nn.functional.linear(inputs, weight))  # noqa: E731
    lora = GatedLoRA(layers, state, SCALE, mask, torch.device("cpu"), dtype=torch.float32,
                     draft_base=doubled, fold=fold, block=4, max_rows=4)
    module = layers[1].self_attn.o_proj
    small, large = torch.randn(4, HIDDEN), torch.randn(8, HIDDEN)
    base_small, _ = module(small)
    base_large, _ = module(large)
    with lora.active():
        got_small, _ = module(small)
        got_large, bias = module(large)

    assert bias is None
    assert torch.allclose(got_small[0], 2 * base_small[0], atol=1e-4)      # the draft base
    assert torch.equal(got_large[0::4], base_large[0::4])                   # the module's own weight, seed rows
    noise = [1, 2, 3, 5, 6, 7]
    want = base_large[noise] + _delta(state, "model.layers.1.self_attn.o_proj", large[noise])
    assert torch.allclose(got_large[noise], want, atol=1e-2)
    assert lora.fired == 2


def test_a_folded_base_may_return_padding_columns_after_the_lora_ones():
    torch.manual_seed(0)
    layers, state, x = _layers(), _state(), torch.randn(4, HIDDEN)

    def padded(weight):
        return lambda inputs: torch.nn.functional.pad(torch.nn.functional.linear(inputs, weight), (0, 5), value=7.0)

    separate = GatedLoRA(layers, state, SCALE, _mask(4), torch.device("cpu"), dtype=torch.float32)
    folded = GatedLoRA(layers, state, SCALE, _mask(4), torch.device("cpu"), dtype=torch.float32,
                       draft_base=padded, fold=True, block=4)
    module = layers[1].self_attn.qkv_proj
    with separate.active():
        want, _ = module(x)
    with folded.active():
        got, _ = module(x)
    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=1e-2)
