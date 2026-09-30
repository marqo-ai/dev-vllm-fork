# SPDX-License-Identifier: Apache-2.0
"""Low-bit draft copies of the target's linear weights (GPU only: Marlin kernels)."""
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.uno.draft_weights import draft_linear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


# (out, in) of the 27B's adapted modules: in_proj_qkvz, gate_up_proj, down_proj, attention qkv_proj,
# out_proj / o_proj
SHAPES = [(16384, 5120), (34816, 5120), (5120, 17408), (14336, 5120), (5120, 6144)]


@pytest.mark.parametrize("kind,tolerance", [("fp8", 0.04), ("int4", 0.2)])
@pytest.mark.parametrize("shape", SHAPES)
def test_draft_linear_approximates_the_bf16_linear(kind, tolerance, shape):
    torch.manual_seed(0)
    weight = (torch.randn(shape, device="cuda") * 0.02).to(torch.bfloat16)
    x = torch.randn(8, shape[1], device="cuda").to(torch.bfloat16)
    apply = draft_linear(weight, kind)
    out = apply(x)
    want = torch.nn.functional.linear(x, weight)
    assert out.shape == want.shape and out.dtype == want.dtype
    error = (out.float() - want.float()).norm() / want.float().norm()
    assert error < tolerance, float(error)


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="fp8"):
        draft_linear(torch.zeros(64, 128, device="cuda", dtype=torch.bfloat16), "nf3")


def test_an_input_width_the_kernel_cannot_take_is_rejected():
    with pytest.raises(ValueError, match="64 x 100"):
        draft_linear(torch.zeros(64, 100, device="cuda", dtype=torch.bfloat16), "fp8")


@pytest.mark.parametrize("kind,tolerance", [("fp8", 0.04), ("int4", 0.2)])
def test_an_output_width_off_the_kernels_grid_is_padded(kind, tolerance):
    torch.manual_seed(0)
    weight = (torch.randn(8200, 1024, device="cuda") * 0.02).to(torch.bfloat16)   # 8192 + a rank-8 adapter
    x = torch.randn(4, 1024, device="cuda").to(torch.bfloat16)
    out = draft_linear(weight, kind)(x)
    want = torch.nn.functional.linear(x, weight)
    assert out.shape[1] >= 8200
    error = (out[:, :8200].float() - want.float()).norm() / want.float().norm()
    assert error < tolerance, float(error)


@pytest.mark.parametrize("kind,tolerance", [("fp8", 0.04), ("int4", 0.2)])
def test_reducing_in_the_model_dtype_stays_within_the_same_tolerance(kind, tolerance):
    torch.manual_seed(0)
    weight = (torch.randn(5120, 17408, device="cuda") * 0.02).to(torch.bfloat16)   # down_proj: the longest reduction
    x = torch.randn(8, 17408, device="cuda").to(torch.bfloat16)
    out = draft_linear(weight, kind, fp32_reduce=False)(x)
    want = torch.nn.functional.linear(x, weight)
    error = (out.float() - want.float()).norm() / want.float().norm()
    assert error < tolerance, float(error)


def test_int4_with_a_scale_per_32_inputs_is_closer_than_per_128():
    torch.manual_seed(0)
    weight = (torch.randn(5120, 6144, device="cuda") * 0.02).to(torch.bfloat16)
    x = torch.randn(8, 6144, device="cuda").to(torch.bfloat16)
    want = torch.nn.functional.linear(x, weight).float()
    error = {
        kind: float((draft_linear(weight, kind)(x).float() - want).norm() / want.norm())
        for kind in ("int4", "int4g32")
    }
    assert error["int4g32"] < 0.9 * error["int4"], error
