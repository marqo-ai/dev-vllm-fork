# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-gated dense LoRA for the Uno draft forward.

Uno's draft runs the target model with a LoRA on the noise rows and the
original weights on the seed row. The delta is added by forward hooks that are
installed only while the draft forward is traced into its CUDA graph, so the
target's modules, compiled forward and graphs never see it.

Math, per adapted projection, as in training:
    y += row_mask * scale * (x.float() @ A^T) @ B^T
Projections that vLLM packs into one module (q/k/v, gate/up, qkv/z) are
concatenated along the rank axis so each module costs two GEMMs.

Two ways to run it. With `dtype=None` the delta is computed in fp32 exactly as
in training and cast back, about six small kernels per module. With a dtype
(the model's), the matrices live in that dtype and the delta is three kernels:
a GEMM, the row mask, and an in-place addmm.

`draft_base` swaps the base projection as well, for the draft forward only: it
maps a weight to a function computing the linear output from a cheaper copy of
it (see draft_weights.py). Every draft row then reads the cheaper copy, the
seed row included, so this trades acceptance for the bytes the draft forward
has to read; the verify forward is never affected. With `fold`, A is appended
to that weight as extra output channels, so the base GEMM also produces the
LoRA's hidden and the delta costs one addmm. With `block` (rows per request),
a batch of one request adds the delta to its noise rows directly instead of
masking the seed row out. A batch of more than `max_rows` rows reads the
module's own weight: a low-bit kernel decodes per row and loses to the dense
GEMM once the batch is large.
"""

import json
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn

# adapter projection -> (vLLM module path inside a decoder layer, slice index in that module)
_MODULES = {
    "linear_attn.in_proj_qkv": ("linear_attn.in_proj_qkvz", 0),
    "linear_attn.out_proj": ("linear_attn.out_proj", 0),
    "self_attn.q_proj": ("self_attn.qkv_proj", 0),
    "self_attn.k_proj": ("self_attn.qkv_proj", 1),
    "self_attn.v_proj": ("self_attn.qkv_proj", 2),
    "self_attn.o_proj": ("self_attn.o_proj", 0),
    "mlp.gate_proj": ("mlp.gate_up_proj", 0),
    "mlp.up_proj": ("mlp.gate_up_proj", 1),
    "mlp.down_proj": ("mlp.down_proj", 0),
}
_KEY = re.compile(r"^model\.layers\.(\d+)\.(.+)\.([AB])$")


def load_uno_adapter(path: str) -> tuple[dict[str, torch.Tensor], float]:
    """Return the adapter state dict and its alpha / rank scale.

    The scale is read from the sidecar `<path stem>.json`: loading an adapter
    at the wrong scale degrades acceptance silently.
    """
    adapter = Path(path)
    scale = float(json.loads(adapter.with_suffix(".json").read_text())["scale"])
    if adapter.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(adapter))
    else:
        state = torch.load(adapter, map_location="cpu", weights_only=True)
    return state, scale


class GatedLoRA:
    def __init__(
        self,
        layers: nn.ModuleList,
        state: dict[str, torch.Tensor],
        scale: float,
        row_mask: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype | None = None,
        draft_base: Callable[[torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]
        | None = None,
        fold: bool = False,
        block: int | None = None,
        max_rows: int | None = None,
    ):
        if draft_base is not None and dtype is None:
            raise ValueError(
                "A draft-only base needs the LoRA delta in the model dtype; pass dtype."
            )
        if fold and draft_base is None:
            raise ValueError("Folding A into the base weight needs a draft-only base.")
        grouped: dict[tuple[int, str], dict[int, dict[str, torch.Tensor]]] = {}
        for key, tensor in state.items():
            match = _KEY.match(key)
            if match is None:
                raise ValueError(f"Unexpected Uno adapter key: {key}")
            layer, name, side = int(match.group(1)), match.group(2), match.group(3)
            if name not in _MODULES:
                raise ValueError(f"Uno adapter targets an unsupported projection: {name}")
            path, slot = _MODULES[name]
            grouped.setdefault((layer, path), {}).setdefault(slot, {})[side] = tensor

        self.row_mask = row_mask
        self.block = block
        self.max_rows = max_rows
        self.fired = 0
        self._hooks: list[tuple[nn.Module, object]] = []
        self._forwards: list[tuple[nn.Module, object]] = []
        for (layer, path), slices in sorted(grouped.items()):
            if sorted(slices) != list(range(len(slices))):
                raise ValueError(
                    f"Uno adapter must cover layer {layer} {path} from its first "
                    f"slice; got slices {sorted(slices)}"
                )
            a_cat = torch.cat([slices[i]["A"].float() for i in range(len(slices))], dim=0)
            b_cat = torch.block_diag(*[slices[i]["B"].float() * scale for i in range(len(slices))])
            module = layers[layer].get_submodule(path)
            if dtype is None:
                self._hooks.append((module, self._make_hook(a_cat.to(device), b_cat.to(device))))
                continue
            a = a_cat.to(device=device, dtype=dtype)
            b_t = b_cat.to(device=device, dtype=dtype).t().contiguous()
            if draft_base is None:
                self._hooks.append((module, self._make_fused_hook(a, b_t)))
                continue
            if getattr(module, "bias", None) is not None:
                raise ValueError(
                    f"A draft-only base needs a bias-free projection; layer {layer} {path} has a bias."
                )
            weight = module.weight.detach()
            base = draft_base(torch.cat([weight, a.to(weight.dtype)], dim=0) if fold else weight)
            self._forwards.append((module, self._make_forward(module, base, fold, a, b_t)))
        self.num_modules = len(self._hooks) + len(self._forwards)

    def _make_hook(self, a: torch.Tensor, b: torch.Tensor):
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            x = args[0]
            hidden = torch.nn.functional.linear(x.float(), a) * self.row_mask[: x.shape[0]]
            out[:, : b.shape[0]] += torch.nn.functional.linear(hidden, b).to(out.dtype)
            self.fired += 1

        return hook

    def _make_fused_hook(self, a: torch.Tensor, b_t: torch.Tensor):
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            x = args[0]
            hidden = torch.nn.functional.linear(x, a) * self.row_mask[: x.shape[0]]
            out[:, : b_t.shape[1]].addmm_(hidden, b_t)
            self.fired += 1

        return hook

    def _make_forward(self, module: nn.Module, base, fold: bool, a: torch.Tensor, b_t: torch.Tensor):
        # vLLM's linear layers return (output, bias) unless return_bias is off.
        with_bias = getattr(module, "return_bias", True)
        own_forward = type(module).forward
        width = module.weight.shape[0]
        rank, cols = b_t.shape

        def forward(x):
            rows = x.shape[0]
            self.fired += 1
            if self.max_rows is not None and rows > self.max_rows:
                result = own_forward(module, x)
                out = result[0] if with_bias else result
                hidden = torch.nn.functional.linear(x, a) * self.row_mask[:rows]
                out[:, :cols].addmm_(hidden, b_t)
                return result
            out = base(x)
            # Folded: the columns past the module's own are x @ A^T, then
            # whatever the base pads its output with.
            hidden = out[:, width : width + rank] if fold else torch.nn.functional.linear(x, a)
            out = out[:, :width]
            if self.block is not None and rows < 2 * self.block:
                # One request: row 0 is its seed row, the rest noise or padding.
                out[1:, :cols].addmm_(hidden[1:], b_t)
            else:
                out[:, :cols].addmm_(hidden * self.row_mask[:rows], b_t)
            return (out, None) if with_bias else out

        return forward

    @contextmanager
    def active(self) -> Iterator[None]:
        handles = [module.register_forward_hook(hook) for module, hook in self._hooks]
        for module, forward in self._forwards:
            module.forward = forward
        self.fired = 0
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            for module, _ in self._forwards:
                del module.forward

    def check_all_fired(self) -> None:
        """Every adapted module must have run exactly once in the last forward.

        A module that the model reaches without calling it (a fused path that
        reads `.weight` directly) would drop its delta without any error.
        """
        if self.fired != self.num_modules:
            raise RuntimeError(
                f"Uno LoRA reached {self.fired} of {self.num_modules} adapted "
                "modules in the draft forward."
            )
