# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Uno (arXiv 2609.04010) linear sampler on a hybrid GDN + attention target.

Per round, for a request whose committed length is L:

  draft   one forward of the TARGET's own layers over [seed, noise x (K-1)] at
          positions L-1 .. L+K-2, where the seed is the last sampled token. Row 0
          uses the original weights and yields the exact next token; rows 1..K-1
          carry the Uno LoRA and yield the draft.
  verify  vLLM's unmodified verify step over [seed, draft x K].

The draft leaves no trace. Attention K/V for its positions land in the request's
own pages and are rewritten by the verify before being read. GDN state is
handled by `state_tables.build_draft_state_table`.
"""

import os
import time
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridAttnMetadata
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.uno.draft_norms import DraftNorms, compilation_enabled
from vllm.v1.worker.gpu.spec_decode.uno.draft_weights import KINDS, draft_linear
from vllm.v1.worker.gpu.spec_decode.uno.gated_lora import GatedLoRA, load_uno_adapter
from vllm.v1.worker.gpu.spec_decode.uno.isolation import (
    changed_blocks,
    changed_slots,
    kv_block_view,
    kv_blocks_and_allowed,
    state_allowed,
)
from vllm.v1.worker.gpu.spec_decode.uno.state_tables import (
    build_draft_state_table,
    draft_seed_positions,
    pad_slots_past_limit,
    stage_conv_windows,
    write_mrope_positions,
)
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


def unsupported_reason(vllm_config: VllmConfig) -> str | None:
    """Why this configuration cannot run `uno`, or None if it can."""
    if vllm_config.cache_config.mamba_cache_mode != "none":
        return (
            "uno requires prefix caching to be disabled "
            "(enable_prefix_caching=False): its draft forward addresses a "
            "request's GDN state blocks by column, which only the 'none' mamba "
            "cache mode provides."
        )
    parallel = vllm_config.parallel_config
    for name in (
        "tensor_parallel_size",
        "data_parallel_size",
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
    ):
        if getattr(parallel, name) != 1:
            return f"uno supports {name}=1 only."
    if vllm_config.lora_config is not None:
        return (
            "uno cannot run on a LoRA-enabled engine: its draft forward would "
            "go through the engine's LoRA layers with the target's token mapping."
        )
    return None


class UnoSpeculator(DraftModelSpeculator):
    _speculator_name = "Uno"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        reason = unsupported_reason(vllm_config)
        if reason is not None:
            raise ValueError(reason)

        # K: one seed row plus K - 1 noise rows in, K proposals out.
        self.block = self.num_speculative_steps
        # State blocks a request owns in each GDN group.
        self.state_width = self.block + 1
        self.adapter_path = self.speculative_config.model
        self.debug_mode = os.environ.get("VLLM_UNO_DEBUG", "")
        self.use_graphs = os.environ.get("VLLM_UNO_GRAPHS", "1") == "1"
        # "model": the LoRA delta in the model's dtype, three kernels per
        # module. "float32": in fp32 as in training, about six.
        self.lora_dtype = os.environ.get("VLLM_UNO_LORA_DTYPE", "model")
        # "fp8" / "int4" / "int4g32": the draft forward reads a low-bit copy of every
        # adapted projection (draft_weights.py). Costs acceptance, never output.
        self.draft_weights = os.environ.get("VLLM_UNO_DRAFT_WEIGHTS", "")
        if self.draft_weights and (
            self.draft_weights not in KINDS or self.lora_dtype == "float32"
        ):
            raise ValueError(
                f"VLLM_UNO_DRAFT_WEIGHTS must be one of {KINDS} and needs "
                "VLLM_UNO_LORA_DTYPE=model."
            )
        # The Marlin kernels' reduction dtype for the draft weights (see draft_linear).
        self.draft_fp32_reduce = os.environ.get("VLLM_UNO_DRAFT_FP32_REDUCE", "1") == "1"
        # Fold the LoRA's A into the draft weight as extra output channels
        # (one addmm per module instead of a GEMM and an addmm). Measured free
        # with fp8 and within noise with 4-bit.
        self.fold_lora = os.environ.get("VLLM_UNO_FOLD_LORA", "1")
        # The draft forward's RMSNorms as one compiled kernel pair each
        # (draft_norms.py); "0" leaves them as the eager fp32 formula.
        self.fused_norms = os.environ.get("VLLM_UNO_FUSED_NORMS", "1") == "1"
        # Above this many draft rows the dense GEMM on the target's weight wins
        # (measured on an H200: even at 64 rows, behind at 128).
        self.draft_weights_max_rows = int(
            os.environ.get("VLLM_UNO_DRAFT_WEIGHTS_MAX_ROWS", "64")
        )
        # "1": per-round timers. "kernels": also one table of the draft
        # graph's GPU kernels by time, at round 100.
        self.profile_kernels = os.environ.get("VLLM_UNO_PROF", "0") == "kernels"
        self.profile = self.profile_kernels or os.environ.get("VLLM_UNO_PROF", "0") == "1"
        self._prof = [0.0, 0.0, 0.0, 0]
        self._laps: dict[str, float] = {}
        self._lap_start = 0.0

        max_sampled = self.max_num_reqs * self.block
        self.sample_indices = torch.zeros(max_sampled, dtype=torch.int64, device=device)
        self.sample_pos = torch.zeros(max_sampled, dtype=torch.int64, device=device)
        # -1 marks an inert sampling row (see DFlashSpeculator).
        self.sample_idx_mapping = torch.full(
            (max_sampled,), -1, dtype=torch.int32, device=device
        )
        self.sample_col = torch.arange(
            self.block, dtype=torch.int32, device=device
        ).repeat(self.max_num_reqs)
        # prepare_dflash_inputs also emits the target rows' context positions
        # and slots for DFlash's K/V precompute. Uno reads the target's own
        # cache, so these are write-only scratch.
        self._context_positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )
        self._context_slots = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

        # 1 on noise rows, 0 on seed rows. Rows are laid out K per request.
        self.row_mask = torch.ones(
            self.max_num_tokens,
            1,
            dtype=torch.float32 if self.lora_dtype == "float32" else self.dtype,
            device=device,
        )
        self.row_mask[0 :: self.block] = 0
        self.noise_generator = torch.Generator(device=device).manual_seed(718300)
        self.rope_state = None
        self.positions_nd: torch.Tensor | None = None

        self.query_cudagraph_manager: DFlashCudaGraphManager | None = None
        self.state_tables: dict[int, torch.Tensor] = {}
        self.stage: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._checked_rounds = 0
        self._accepted_hist = torch.zeros(self.state_width + 1, dtype=torch.int64, device=device)

    # ---- model: the draft is the target ----
    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        return target_model

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        # The draft block is text only.
        self.supports_mm_inputs = False
        causal = getattr(target_model, "language_model", target_model)
        self.decoder = causal.model
        layers = self.decoder.layers
        if not all(hasattr(layer, "layer_type") for layer in layers):
            raise ValueError(
                "uno supports the Qwen3.5-family hybrid decoder only; got "
                f"{type(target_model).__name__}."
            )
        state, scale = load_uno_adapter(self.adapter_path)
        self.lora = GatedLoRA(
            layers,
            state,
            scale,
            self.row_mask,
            self.device,
            None if self.lora_dtype == "float32" else self.dtype,
            (lambda weight: draft_linear(weight, self.draft_weights, self.draft_fp32_reduce))
            if self.draft_weights
            else None,
            fold=bool(self.draft_weights) and self.fold_lora == "1",
            block=self.block if self.draft_weights else None,
            max_rows=self.draft_weights_max_rows if self.draft_weights else None,
        )
        norms = [
            norm
            for layer in layers
            for norm in (layer.input_layernorm, layer.post_attention_layernorm)
        ] + [self.decoder.norm]
        self.draft_norms = (
            DraftNorms(norms)
            if self.fused_norms and compilation_enabled(self.vllm_config.compilation_config)
            else None
        )
        self.gdn_modules = [
            layer.linear_attn for layer in layers if layer.layer_type == "linear_attention"
        ]
        self.conv_history = self.gdn_modules[0].conv_kernel_size - 1
        logger.info(
            "%s: draft = target (%d layers, %d GDN), block %d, %d adapted modules, "
            "scale %.1f, adapter %s, draft weights %s",
            self._speculator_name, len(layers), len(self.gdn_modules), self.block,
            self.lora.num_modules, scale, self.adapter_path,
            self.draft_weights or "the target's",
        )

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        # No super().set_attn(): the draft adds no layers, so there are no
        # draft attention groups to build. It runs the target's layers through
        # the target's own metadata builders.
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.block_tables = block_tables
        self.target_input_buffers = target_input_buffers
        self.target_attn_groups = target_attn_groups
        self.attn_groups = target_attn_groups
        # M-RoPE targets take one position per axis; the runner keeps each
        # request's offset from its token index in the rope state.
        self.rope_state = getattr(model_state, "rope_state", None)
        if self.rope_state is not None:
            self.positions_nd = torch.zeros(
                self.rope_state.num_dims,
                self.max_num_tokens,
                dtype=torch.int64,
                device=self.device,
            )

        groups = kv_cache_config.kv_cache_groups
        self.mamba_gids = [
            gid for gid, group in enumerate(groups)
            if isinstance(group.kv_cache_spec, MambaSpec)
        ]
        self.attn_gids = [gid for gid in range(len(groups)) if gid not in self.mamba_gids]
        self.gid_of_layer = {
            name: gid for gid, group in enumerate(groups) for name in group.layer_names
        }
        for gid in self.mamba_gids:
            self.state_tables[gid] = torch.zeros(
                self.max_num_reqs, self.state_width, dtype=torch.int32, device=self.device
            )
            self.stage[gid] = tuple(
                torch.zeros(self.max_num_reqs, dtype=torch.int64, device=self.device)
                for _ in range(3)
            )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        wants_full = cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        mode = (
            CUDAGraphMode.FULL_DECODE_ONLY
            if wants_full and self.use_graphs
            else CUDAGraphMode.NONE
        )
        self.query_cudagraph_manager = DFlashCudaGraphManager(
            self.vllm_config, self.device, mode, decode_query_len=self.block
        )
        logger.info("%s: draft CUDA graph mode %s", self._speculator_name, mode)

    def capture(self) -> None:
        assert self.query_cudagraph_manager is not None
        if self.query_cudagraph_manager.cudagraph_mode == CUDAGraphMode.NONE:
            return
        # Capture against null blocks: padded rows must not touch live state.
        self.sample_indices.zero_()
        self.sample_pos.zero_()
        self.sample_idx_mapping.fill_(-1)
        for gid in self.mamba_gids:
            self.state_tables[gid].zero_()
            for buffer in self.stage[gid]:
                buffer.zero_()
        self.query_cudagraph_manager.capture(
            self._generate_draft,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            self.max_model_len,
            causal=True,
            progress_bar_desc="Capturing uno CUDA graphs",
        )

    # ---- GDN state ----
    def _stage_conv_windows(self, num_reqs: int) -> None:
        """Copy each request's committed convolution history into its scratch
        block, at the offset the kernel reads when num_accepted == state_width."""
        dim_first = is_conv_state_dim_first()
        for gid in self.mamba_gids:
            src_block, dst_block, src_offset = (b[:num_reqs] for b in self.stage[gid])
            stage_conv_windows(
                # Resolved on every call, never cached: vLLM builds a temporary
                # KV cache for memory profiling and replaces it afterwards.
                [
                    module.kv_cache[0]
                    for module in self.gdn_modules
                    if self.gid_of_layer[module.prefix] == gid
                ],
                src_block,
                dst_block,
                src_offset,
                self.state_width - 1,
                self.conv_history,
                dim_first,
            )

    def _prepare_state_tables(
        self, input_batch: InputBatch, active: torch.Tensor
    ) -> None:
        num_reqs = input_batch.num_reqs
        accepted = self.model_state.num_accepted_tokens_gpu[input_batch.idx_mapping]
        for gid in self.mamba_gids:
            real = self.block_tables.input_block_tables[gid][:num_reqs]
            table, src_block, dst_block, src_offset = build_draft_state_table(
                real, accepted, active, self.state_width
            )
            self.state_tables[gid].zero_()
            self.state_tables[gid][:num_reqs] = table
            for buffer, value in zip(self.stage[gid], (src_block, dst_block, src_offset)):
                buffer.zero_()
                buffer[:num_reqs] = value

    # ---- VLLM_UNO_DEBUG=check: the draft forward must leave no trace ----
    def _attention_caches(self, gid: int) -> list[torch.Tensor]:
        caches = []
        layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for name, layer in layers.items():
            kv = layer.kv_cache
            kv = kv[0] if isinstance(kv, (list, tuple)) else kv
            if self.gid_of_layer.get(name) == gid and kv.numel():
                caches.append(kv)
        return caches

    def _snapshot_committed(self, num_reqs: int, seed_pos: torch.Tensor) -> list[tuple]:
        """Copy every state block and K/V slot the draft forward must not change.

        Allowed to change: each request's scratch state block, the K/V slots at
        its draft positions below the context limit, and the null block.
        """
        snaps: list[tuple] = []
        for gid in self.mamba_gids:
            real = self.block_tables.input_block_tables[gid][:num_reqs, : self.state_width]
            real = real.to(torch.int64)
            blocks = real.reshape(-1)
            allowed = state_allowed(real, self.stage[gid][1][:num_reqs])
            for module in self.gdn_modules:
                if self.gid_of_layer[module.prefix] != gid:
                    continue
                for which, state in zip(("conv", "recurrent"), module.kv_cache):
                    before = state.index_select(0, blocks).clone()
                    snaps.append((f"{module.prefix} {which} state", state, blocks, allowed, before))
        for gid in self.attn_gids:
            block_size = int(self.block_tables.kernel_block_sizes[gid])
            blocks, allowed = kv_blocks_and_allowed(
                self.block_tables.input_block_tables[gid][:num_reqs],
                seed_pos,
                self.block,
                block_size,
                self.max_model_len,
            )
            for kv in self._attention_caches(gid):
                view = kv_block_view(kv, self.kv_cache_config.num_blocks, block_size)
                before = view.index_select(0, blocks).clone()
                snaps.append((f"attention K/V (group {gid})", view, blocks, allowed, before))
        return snaps

    def _assert_untouched(self, snaps: list[tuple], accepted: torch.Tensor) -> None:
        for name, tensor, blocks, allowed, before in snaps:
            after = tensor.index_select(0, blocks)
            compare = changed_blocks if allowed.ndim == 1 else changed_slots
            bad = compare(before, after) & ~allowed
            if bool(bad.any()):
                raise RuntimeError(
                    f"Uno draft forward changed committed state: {name}, at "
                    f"{bad.nonzero()[:8].tolist()} of blocks {blocks.tolist()}"
                )
        self._accepted_hist += torch.bincount(
            accepted.to(torch.int64).clamp(0, self.state_width), minlength=self.state_width + 1
        )
        self._checked_rounds += 1
        if self._checked_rounds % 200 == 0:
            logger.info(
                "%s: isolation check passed for %d rounds (%d tensors per round); "
                "num_accepted histogram %s",
                self._speculator_name, self._checked_rounds, len(snaps),
                self._accepted_hist.tolist(),
            )

    # ---- attention metadata ----
    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        batch_desc: BatchExecutionDescriptor,
        seq_lens_cpu_upper_bound: torch.Tensor,
    ) -> dict[str, Any]:
        block = self.block
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        num_tokens = (
            batch_desc.num_tokens
            if batch_desc.cg_mode == CUDAGraphMode.FULL
            else num_reqs * block
        )
        query_start_loc_cpu = torch.empty(num_reqs_padded + 1, dtype=torch.int32)
        query_start_loc_cpu[: num_reqs + 1] = (
            torch.arange(num_reqs + 1, dtype=torch.int32) * block
        )
        query_start_loc_cpu[num_reqs:] = num_reqs * block
        seq_lens_upper = torch.zeros(num_reqs_padded, dtype=torch.int32)
        torch.add(
            seq_lens_cpu_upper_bound[:num_reqs], block, out=seq_lens_upper[:num_reqs]
        )
        seq_lens_upper[:num_reqs].clamp_(max=self.max_model_len)

        # GDN groups get the draft tables; attention groups keep the real ones.
        block_tables = [
            (self.state_tables[gid] if gid in self.state_tables else table)[:num_reqs_padded]
            for gid, table in enumerate(self.block_tables.input_block_tables)
        ]
        # Mark every real row as a speculative decode of `block` rows that
        # starts from the last state column. Padded rows stay non-speculative
        # and zero-length, which the GDN builder ignores.
        num_draft_tokens_cpu = torch.full((num_reqs_padded,), -1, dtype=torch.int32)
        num_draft_tokens_cpu[:num_reqs] = block - 1
        hybrid_metadata = MambaHybridAttnMetadata(
            is_prefilling=torch.zeros(num_reqs_padded, dtype=torch.bool),
            num_accepted_tokens=torch.full(
                (num_reqs_padded,), self.state_width, dtype=torch.int32, device=self.device
            ),
            num_decode_draft_tokens_cpu=num_draft_tokens_cpu,
        )
        return build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs_padded,
            num_tokens=num_tokens,
            query_start_loc_gpu=self.input_buffers.query_start_loc[: num_reqs_padded + 1],
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=block,
            seq_lens=self.input_buffers.seq_lens[:num_reqs_padded],
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=self.block_tables.slot_mappings[:, :num_tokens],
            kv_cache_config=self.kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_upper,
            model_specific_attn_metadata=hybrid_metadata,
            causal=True,
        )

    # ---- the draft forward (this is what the CUDA graph captures) ----
    @torch.inference_mode()
    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        self._stage_conv_windows(num_reqs)
        input_ids = self.input_buffers.input_ids[:num_tokens_padded]
        positions = (
            self.positions_nd[:, :num_tokens_padded]
            if self.positions_nd is not None
            else self.input_buffers.positions[:num_tokens_padded]
        )
        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                slot_mapping=slot_mappings,
                batch_descriptor=BatchDescriptor(num_tokens=num_tokens_padded),
            ),
            # VLLM_UNO_DEBUG=nolora drafts with the original weights on every
            # row, to measure what the LoRA costs.
            nullcontext() if self.debug_mode == "nolora" else self.lora.active(),
            nullcontext() if self.draft_norms is None else self.draft_norms.active(),
        ):
            # The decoder's own layer loop, run directly: its compiled forward
            # is traced for the runner's argument pattern.
            hidden_states = self.decoder.embed_input_ids(input_ids)
            residual = None
            for layer in self.decoder.layers:
                hidden_states, residual = layer(
                    positions=positions, hidden_states=hidden_states, residual=residual
                )
            hidden_states, _ = self.decoder.norm(hidden_states, residual)
        if self.debug_mode != "nolora":
            self.lora.check_all_fired()

        num_sample = num_reqs * self.block
        draft_tokens = self.sample_draft(
            hidden_states[self.sample_indices[:num_sample]],
            # sample_pos is the predicted token's position; a draw is keyed by
            # the position before it.
            self.sample_pos[:num_sample] - 1,
            self.sample_idx_mapping[:num_sample],
            self.temperature,
            self.seeds,
            self.sample_col[:num_sample],
            self.draft_logits,
        )
        self.draft_tokens[:num_reqs] = draft_tokens.view(num_reqs, self.block)

    def _tick(self) -> float:
        if not self.profile:
            return 0.0
        torch.cuda.synchronize()
        return time.perf_counter()

    def _lap(self, name: str) -> None:
        """Charge the time since the last lap (or tick) to `name`."""
        if not self.profile:
            return
        torch.cuda.synchronize()
        now = time.perf_counter()
        self._laps[name] = self._laps.get(name, 0.0) + now - self._lap_start
        self._lap_start = now

    def _log_kernel_table(self, batch_desc, replays: int = 5) -> None:
        """Replay the draft graph under the profiler and log its kernels by time.

        Replaying is harmless: the draft forward leaves committed state alone
        and its inputs sit in the buffers.
        """
        from torch.profiler import ProfilerActivity, profile

        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(replays):
                self.query_cudagraph_manager.run_fullgraph(batch_desc)
            torch.cuda.synchronize()
        averages = prof.key_averages()
        try:
            table = averages.table(
                sort_by="self_cuda_time_total", row_limit=45, max_name_column_width=100
            )
        except (AttributeError, KeyError):
            table = averages.table(
                sort_by="self_device_time_total", row_limit=45, max_name_column_width=100
            )
        logger.info(
            "%s: draft graph kernels over %d replays (%d rows):\n%s",
            self._speculator_name, replays, batch_desc.num_tokens, table,
        )

    # ---- the round ----
    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        block = self.block
        if dummy_run or is_profile or self.debug_mode == "null":
            self.draft_tokens[:num_reqs].zero_()
            return self.draft_tokens[:num_reqs]
        t0 = self._lap_start = self._tick()
        if self.debug_mode == "slow_prepare":
            # 3 ms of CPU time before the draft is launched: if throughput does
            # not move, the prepare stage is hidden behind the verify forward.
            time.sleep(0.003)

        # Inputs for [seed, placeholder x (K-1)] per request: ids, positions,
        # sequence lengths, K/V slots in the request's own pages, and the
        # sampling indices. Every row predicts the next token
        # (sample_from_anchor), which is Uno's layout.
        for gid in self.attn_gids:
            prepare_dflash_inputs(
                input_buffers=self.input_buffers,
                query_slot_mapping=self.block_tables.slot_mappings[gid],
                context_positions=self._context_positions,
                context_slot_mapping=self._context_slots,
                sample_indices=self.sample_indices,
                sample_pos=self.sample_pos,
                sample_idx_mapping=self.sample_idx_mapping,
                temperature=self.temperature,
                seeds=self.seeds,
                input_batch=input_batch,
                num_sampled=num_sampled,
                num_rejected=num_rejected,
                last_sampled=last_sampled,
                next_prefill_tokens=next_prefill_tokens,
                input_temperature=temperature,
                input_seeds=seeds,
                block_table=self.block_tables.input_block_tables[gid],
                block_size=self.block_tables.kernel_block_sizes[gid],
                cp_rank=self.block_tables.cp_rank,
                cp_size=self.block_tables.cp_size,
                cp_interleave=self.block_tables.cp_interleave,
                parallel_drafting_token_id=0,
                num_query_per_req=block,
                num_speculative_steps=block,
                max_num_reqs=self.max_num_reqs,
                max_num_tokens=self.max_num_tokens,
                max_model_len=self.max_model_len,
                sample_from_anchor=True,
            )
        self._lap("inputs")
        # Rows at or past the context limit must not write K/V: the kernel
        # would wrap them onto committed slots of the request's last block.
        seed_pos = draft_seed_positions(
            input_batch.positions, input_batch.query_start_loc, num_rejected[:num_reqs]
        )
        for gid in self.attn_gids:
            pad_slots_past_limit(
                self.block_tables.slot_mappings[gid],
                seed_pos,
                block,
                self.max_model_len,
                PAD_SLOT_ID,
            )
        # The placeholders become uniform noise ids, as in training.
        noise = torch.randint(
            0, self.vocab_size, (num_reqs, block - 1),
            generator=self.noise_generator, device=self.device, dtype=torch.int32,
        )
        self.input_buffers.input_ids[: num_reqs * block].view(num_reqs, block)[:, 1:] = noise

        # A request that sampled nothing this step is still prefilling: its
        # rows run, but against null state blocks.
        self._lap("limit+noise")
        active = num_sampled[:num_reqs] > 0
        self._prepare_state_tables(input_batch, active)
        self._lap("state tables")

        batch_desc, batch_sync = dispatch_cg_and_sync_dp(
            self.query_cudagraph_manager,
            num_reqs,
            num_reqs * block,
            uniform_token_count=block,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
            dp_sync=None,
        )
        num_tokens_padded = batch_desc.num_tokens
        if self.positions_nd is not None:
            write_mrope_positions(
                self.positions_nd[:, :num_tokens_padded],
                self.input_buffers.positions[:num_tokens_padded],
                self.rope_state.prefill_delta.gpu[input_batch.idx_mapping.to(torch.int64)],
                block,
            )
        self._lap("dispatch+positions")
        draft_attn_metadata = self._build_draft_attn_metadata(
            num_reqs, batch_desc, input_batch.seq_lens_cpu_upper_bound
        )
        self._lap("attn metadata")
        draft_slot_mappings = build_slot_mappings_by_layer(
            self.block_tables.slot_mappings[:, :num_tokens_padded], self.kv_cache_config
        )
        self._lap("slot mappings")

        snaps = (
            self._snapshot_committed(num_reqs, seed_pos) if self.debug_mode == "check" else None
        )
        t1 = self._tick()

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.query_cudagraph_manager is not None
            self.query_cudagraph_manager.run_fullgraph(batch_desc)
        else:
            self._generate_draft(
                num_reqs,
                num_tokens_padded,
                draft_attn_metadata,
                draft_slot_mappings,
                num_tokens_across_dp=(
                    batch_sync.num_tokens_across_dp if batch_sync is not None else None
                ),
                cudagraph_runtime_mode=batch_desc.cg_mode,
            )

        if self.profile:
            t2 = self._tick()
            self._prof[0] += t1 - t0
            self._prof[1] += t2 - t1
            self._prof[3] += 1
            if self._prof[3] % 200 == 0:
                logger.info(
                    "%s: per round, prepare %.2f ms, draft forward %.2f ms (%d rounds, "
                    "LoRA %s, debug %r); prepare by stage, ms: %s",
                    self._speculator_name,
                    1e3 * self._prof[0] / self._prof[3],
                    1e3 * self._prof[1] / self._prof[3],
                    self._prof[3],
                    self.lora_dtype,
                    self.debug_mode,
                    {k: round(1e3 * v / self._prof[3], 2) for k, v in self._laps.items()},
                )
            if (
                self.profile_kernels
                and self._prof[3] == 100
                and batch_desc.cg_mode == CUDAGraphMode.FULL
            ):
                self._log_kernel_table(batch_desc)
        if snaps is not None:
            self._assert_untouched(
                snaps, self.model_state.num_accepted_tokens_gpu[input_batch.idx_mapping]
            )
        if self.debug_mode == "discard":
            self.draft_tokens[:num_reqs].zero_()
        return self.draft_tokens[:num_reqs]
