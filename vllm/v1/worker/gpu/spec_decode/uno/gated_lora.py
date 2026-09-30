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
"""

import json
import re
from collections.abc import Iterator
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
    ):
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
        self.fired = 0
        self._hooks: list[tuple[nn.Module, object]] = []
        for (layer, path), slices in sorted(grouped.items()):
            if sorted(slices) != list(range(len(slices))):
                raise ValueError(
                    f"Uno adapter must cover layer {layer} {path} from its first "
                    f"slice; got slices {sorted(slices)}"
                )
            a_cat = torch.cat([slices[i]["A"].float() for i in range(len(slices))], dim=0)
            b_cat = torch.block_diag(*[slices[i]["B"].float() * scale for i in range(len(slices))])
            module = layers[layer].get_submodule(path)
            self._hooks.append((module, self._make_hook(a_cat.to(device), b_cat.to(device))))
        self.num_modules = len(self._hooks)

    def _make_hook(self, a: torch.Tensor, b: torch.Tensor):
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            x = args[0]
            hidden = torch.nn.functional.linear(x.float(), a) * self.row_mask[: x.shape[0]]
            out[:, : b.shape[0]] += torch.nn.functional.linear(hidden, b).to(out.dtype)
            self.fired += 1

        return hook

    @contextmanager
    def active(self) -> Iterator[None]:
        handles = [module.register_forward_hook(hook) for module, hook in self._hooks]
        self.fired = 0
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

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
