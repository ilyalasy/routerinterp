"""Stream a corpus into token, SAE-latent, and router files, plus optional residuals and
expert writes.

Files that share a shard number hold the same tokens.
A rerun skips shard numbers already listed in ``progress.json``.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from loguru import logger
from safetensors import safe_open
from torch import Tensor
from tqdm import tqdm

from routerinterp import core
from routerinterp.backends import get_model_backend, get_sae_backend
from routerinterp.core import (
    EXPERT_WRITES_EXPERT_IDX,
    EXPERT_WRITES_LOGIT_LENS,
    EXPERT_WRITES_NORMS,
    EXPERT_WRITES_TOKEN_IDX,
    TOKENS_SPLIT,
    CollectConfig,
    DatasetConfig,
    router_layer_for_hookpoint,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from transformers import PreTrainedTokenizerBase

    from routerinterp.backends import SAEBackend

__all__ = [
    "COLLECT_INFO_FILENAME",
    "EXPERT_WRITES_EXPERT_IDX",
    "EXPERT_WRITES_LOGIT_LENS",
    "EXPERT_WRITES_NORMS",
    "EXPERT_WRITES_TOKEN_IDX",
    "PROGRESS_FILENAME",
    "RESIDUALS",
    "TOKENS_SPLIT",
    "iter_sequence_batches",
    "iter_token_blocks",
    "load_collect_config",
    "load_collect_info",
    "load_pile_set_names",
    "run_collect",
]

# Residual shard key. Token / router / latent / expert-write keys live in core.
RESIDUALS = "residuals"  # float32 [T, H]

PROGRESS_FILENAME = "progress.json"
COLLECT_INFO_FILENAME = "collect_info.json"
SHUFFLE_BUFFER_SIZE = 10_000  # streaming shuffle buffer (documents)
PILE_SET_NAME_KEY = "pile_set_name"
UNKNOWN_PILE_SET = "unknown"
# monology/pile-uncopyrighted subsets. Any corpus that carries the same
# names (flat ``pile_set_name`` or ``meta.pile_set_name``) uses this list.
_PILE_SUBSETS: tuple[str, ...] = (
    "ArXiv",
    "DM Mathematics",
    "Enron Emails",
    "EuroParl",
    "FreeLaw",
    "Github",
    "Gutenberg (PG-19)",
    "HackerNews",
    "NIH ExPorter",
    "PhilPapers",
    "Pile-CC",
    "PubMed Abstracts",
    "PubMed Central",
    "StackExchange",
    "USPTO Backgrounds",
    "Wikipedia (en)",
)
# Rows scanned with no yield before the per-source cap is dropped.
_PILE_STRATIFIED_STALE_THRESHOLD = 1_000_000


# ---------------------------------------------------------------------------
# Deterministic block stream
# ---------------------------------------------------------------------------


def _row_pile_set_name(row: dict[str, Any]) -> str | None:
    """Read the Pile subset name from ``pile_set_name`` or ``meta.pile_set_name``."""
    if PILE_SET_NAME_KEY in row and row[PILE_SET_NAME_KEY] is not None:
        text = str(row[PILE_SET_NAME_KEY]).strip()
        return text or UNKNOWN_PILE_SET
    meta = row.get("meta")
    if isinstance(meta, dict) and PILE_SET_NAME_KEY in meta:
        text = str(meta.get(PILE_SET_NAME_KEY) or "").strip()
        return text or UNKNOWN_PILE_SET
    return None


def _iter_stratified_pile_rows(
    rows: Iterable[dict[str, Any]],
    *,
    num_examples: int,
    sources: Sequence[str] = _PILE_SUBSETS,
    stale_threshold: int = _PILE_STRATIFIED_STALE_THRESHOLD,
) -> Iterator[dict[str, Any]]:
    """Yield up to ``num_examples`` rows, equally capped per Pile subset.

    The cap is dropped once ``stale_threshold`` rows pass with nothing yielded.
    """
    num_per_source = num_examples // len(sources)
    sources_set = set(sources)
    counts: dict[str, int] = dict.fromkeys(sources, 0)
    total = 0
    rows_without_yield = 0
    relaxed = False

    for row in rows:
        if total >= num_examples:
            return
        pile_name = _row_pile_set_name(row) or UNKNOWN_PILE_SET
        if relaxed:
            if pile_name in sources_set:
                yield row
                total += 1
            continue
        if pile_name in sources_set and counts[pile_name] < num_per_source:
            yield row
            counts[pile_name] += 1
            total += 1
            rows_without_yield = 0
            continue
        rows_without_yield += 1
        if rows_without_yield >= stale_threshold:
            relaxed = True
            logger.info(
                f"No progress for {stale_threshold} rows "
                f"for source {pile_name}, taking any remaining to reach target"
            )


def _load_text_stream(dataset: DatasetConfig) -> Any:
    return load_dataset(
        dataset.name,
        revision=dataset.revision,
        split=dataset.split,
        streaming=True,
    )


def _iter_dataset_rows(dataset: DatasetConfig) -> Iterator[dict[str, Any]]:
    """Yield document rows in collect order.

    Shuffled, unless ``load_stratified`` is set, in which case each known Pile subset
    contributes an equal count.
    Rows with no subset name are shuffled instead.
    """
    stream = _load_text_stream(dataset)
    if not dataset.load_stratified:
        yield from stream.shuffle(seed=dataset.seed, buffer_size=SHUFFLE_BUFFER_SIZE)
        return

    iterator = iter(stream)
    try:
        first = next(iterator)
    except StopIteration:
        return
    if _row_pile_set_name(first) is None:
        logger.warning(
            f"load_stratified is set but {dataset.name} has no pile subset "
            "names; falling back to shuffle"
        )
        yield from _load_text_stream(dataset).shuffle(
            seed=dataset.seed, buffer_size=SHUFFLE_BUFFER_SIZE
        )
        return

    num_examples = max(dataset.n_tokens // dataset.seq_len, 1)
    per_source = num_examples // len(_PILE_SUBSETS)
    logger.info(
        f"Stratified sample of {dataset.name}: {num_examples} documents "
        f"({per_source} per subset)"
    )
    yield from _iter_stratified_pile_rows(
        chain((first,), iterator),
        num_examples=num_examples,
    )


def iter_token_blocks(
    dataset: DatasetConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> Iterator[tuple[np.ndarray, int, str | None]]:
    """Yield blocks of ``seq_len`` tokens as ``(tokens, doc_id, pile_set_name)``.

    Documents are concatenated, with an end-of-sequence token between them, and then cut
    into blocks.
    One block can therefore contain the end of one document and the start of the next.
    ``doc_id`` and the subset name come from the document of the first token.
    The same dataset always produces the same blocks.
    """
    sep_ids = tokenizer.encode(
        tokenizer.eos_token or "<|endoftext|>", add_special_tokens=False
    )
    seq_len = dataset.seq_len
    segments: deque[tuple[int, np.ndarray, str | None]] = deque()
    n_pending = 0
    doc_id = -1
    for row in _iter_dataset_rows(dataset):
        doc_id += 1
        ids = np.asarray(
            sep_ids
            + tokenizer(row[dataset.text_column], add_special_tokens=False).input_ids,
            dtype=np.int64,
        )
        segments.append((doc_id, ids, _row_pile_set_name(row)))
        n_pending += ids.shape[0]
        while n_pending >= seq_len:
            block = np.empty(seq_len, dtype=np.int64)
            first_doc_id, _, first_pile = segments[0]
            pos = 0
            while pos < seq_len:
                seg_doc_id, seg_ids, seg_pile = segments[0]
                take = min(seq_len - pos, seg_ids.shape[0])
                block[pos : pos + take] = seg_ids[:take]
                pos += take
                n_pending -= take
                if take == seg_ids.shape[0]:
                    segments.popleft()
                else:
                    segments[0] = (seg_doc_id, seg_ids[take:], seg_pile)
            yield block, first_doc_id, first_pile


def _collate_batch(
    blocks: list[np.ndarray],
    doc_ids: list[int],
    pile_names: list[str | None],
    config: CollectConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str] | None]:
    """Stack blocks and attach deterministic doc-level split labels."""
    tokens_NS = np.stack(blocks)
    doc_ids_N = np.asarray(doc_ids, dtype=np.int64)
    split_N = np.asarray(
        [
            core.doc_is_train(
                int(doc_id),
                seed=config.seed,
                train_frac=config.train_frac,
                fingerprint=config.dataset.fingerprint,
            )
            for doc_id in doc_ids
        ],
        dtype=bool,
    )
    labels = (
        None
        if all(name is None for name in pile_names)
        else [name or UNKNOWN_PILE_SET for name in pile_names]
    )
    return tokens_NS, doc_ids_N, split_N, labels


def iter_sequence_batches(
    config: CollectConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    n_sequences: int,
    skip_sequences: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, list[str] | None]]:
    """Yield batches of token blocks.

    On a rerun, blocks that were already written are still tokenized and then dropped,
    so document ids match the first run.
    """
    blocks: list[np.ndarray] = []
    doc_ids: list[int] = []
    pile_names: list[str | None] = []
    seen = 0
    for tokens_S, doc_id, pile_name in iter_token_blocks(config.dataset, tokenizer):
        if seen >= n_sequences:
            break
        seen += 1
        if seen <= skip_sequences:
            continue
        blocks.append(tokens_S)
        doc_ids.append(doc_id)
        pile_names.append(pile_name)
        if len(blocks) == config.batch_size:
            yield _collate_batch(blocks, doc_ids, pile_names, config)
            blocks, doc_ids, pile_names = [], [], []
    if blocks:
        yield _collate_batch(blocks, doc_ids, pile_names, config)


# ---------------------------------------------------------------------------
# Capture state (fed to ModelBackend.install_captures callbacks)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExpertWritePart:
    """One batch of expert writes: token index, expert index, and the write vector."""

    token_idx: Tensor
    expert_idx: Tensor
    writes: Tensor


@dataclass
class _Captures:
    """Per-batch capture state for one forward pass."""

    activations: dict[str, Tensor] = field(default_factory=dict)
    router_logits: dict[int, Tensor] = field(default_factory=dict)
    expert_parts: dict[int, list[ExpertWritePart]] = field(default_factory=dict)

    def on_activation(self, hookpoint: str, hidden_BSH: Tensor) -> None:
        self.activations[hookpoint] = hidden_BSH.detach()

    def on_router_logits(self, layer_idx: int, logits_TE: Tensor) -> None:
        self.router_logits[layer_idx] = logits_TE.detach()

    def on_expert_write(
        self,
        layer_idx: int,
        expert_idx: int,
        token_idx_R: Tensor,
        writes_RH: Tensor,
    ) -> None:
        if token_idx_R.numel() == 0 or writes_RH.numel() == 0:
            return
        expert_idx_R = torch.full_like(token_idx_R, expert_idx)
        self.expert_parts.setdefault(layer_idx, []).append(
            ExpertWritePart(
                token_idx_R.detach(), expert_idx_R.detach(), writes_RH.detach()
            )
        )

    def expert_writes(self, layer_idx: int) -> ExpertWritePart | None:
        """Return concatenated token indices, expert indices, and writes for a layer."""
        parts = self.expert_parts.get(layer_idx)
        if not parts:
            return None
        return ExpertWritePart(
            torch.cat([part.token_idx for part in parts], dim=0),
            torch.cat([part.expert_idx for part in parts], dim=0),
            torch.cat([part.writes for part in parts], dim=0),
        )

    def clear(self) -> None:
        self.activations.clear()
        self.router_logits.clear()
        self.expert_parts.clear()


# ---------------------------------------------------------------------------
# Logit lens over expert writes
# ---------------------------------------------------------------------------


def _promoted_token_ids(
    writes_RH: Tensor,
    unembed_weight_VH: Tensor,
    num_logits: int,
    *,
    chunk_size: int = 256,
) -> Tensor:
    """Return the top logit-lens token ids for expert writes, as int64 ``[R, P]``."""
    n_rows = writes_RH.shape[0]
    if n_rows == 0:
        return torch.zeros((0, num_logits), dtype=torch.int64)
    device = unembed_weight_VH.device
    dtype = unembed_weight_VH.dtype
    chunks: list[Tensor] = []
    for start in range(0, n_rows, chunk_size):
        chunk_RH = writes_RH[start : start + chunk_size].to(device=device, dtype=dtype)
        logits_RV = F.linear(chunk_RH, unembed_weight_VH)
        chunks.append(logits_RV.topk(num_logits, dim=-1).indices)
    return torch.cat(chunks, dim=0).to(dtype=torch.int64)


# ---------------------------------------------------------------------------
# Aligned shard writer
# ---------------------------------------------------------------------------


def _shard_path(directory: Path, shard_idx: int) -> Path:
    return directory / f"shard-{shard_idx:05d}.safetensors"


class _ShardWriter:
    """Buffer one pass of tokens, latents, router outputs, and any optional captures, then
    write one file of each for those same tokens.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        hookpoints: list[str],
        router_layers: list[int],
        expert_write_layers: list[int],
        sequences_per_shard: int,
        seq_len: int,
        top_k: int,
        store_latents: bool,
        store_router_logits: bool,
        store_dense_residuals: bool,
        logit_lens_top_p: int,
        logit_lens_chunk_size: int,
        unembed_weight_VH: Tensor | None,
    ) -> None:
        self.output_dir = output_dir
        self.hookpoints = hookpoints
        self.router_layers = router_layers
        self.expert_write_layers = expert_write_layers
        self.sequences_per_shard = sequences_per_shard
        self.seq_len = seq_len
        self.top_k = top_k
        self.store_latents = store_latents
        self.store_router_logits = store_router_logits
        self.store_dense_residuals = store_dense_residuals
        self.logit_lens_top_p = logit_lens_top_p
        self.logit_lens_chunk_size = logit_lens_chunk_size
        self.unembed_weight_VH = unembed_weight_VH

        self.n_sequences = 0
        self._tokens: list[np.ndarray] = []
        self._doc_ids: list[np.ndarray] = []
        self._split: list[np.ndarray] = []
        self._pile_names: list[list[str]] | None = None
        self._latent_counts: dict[str, list[np.ndarray]] = {hp: [] for hp in hookpoints}
        self._latent_ids: dict[str, list[np.ndarray]] = {hp: [] for hp in hookpoints}
        self._latent_vals: dict[str, list[np.ndarray]] = {hp: [] for hp in hookpoints}
        self._topk_ids: dict[int, list[np.ndarray]] = {
            layer: [] for layer in router_layers
        }
        self._topk_vals: dict[int, list[np.ndarray]] = {
            layer: [] for layer in router_layers
        }
        self._logits: dict[int, list[np.ndarray]] = {
            layer: [] for layer in router_layers
        }
        self._residuals: dict[str, list[np.ndarray]] = {hp: [] for hp in hookpoints}
        self._write_token_idx: dict[int, list[np.ndarray]] = {
            layer: [] for layer in expert_write_layers
        }
        self._write_expert_idx: dict[int, list[np.ndarray]] = {
            layer: [] for layer in expert_write_layers
        }
        self._write_norms: dict[int, list[np.ndarray]] = {
            layer: [] for layer in expert_write_layers
        }
        self._write_lens: dict[int, list[np.ndarray]] = {
            layer: [] for layer in expert_write_layers
        }

    # -- appenders (one batch at a time) --

    def append_tokens(
        self,
        tokens_NS: np.ndarray,
        doc_ids_N: np.ndarray,
        split_N: np.ndarray,
        pile_names: list[str] | None = None,
    ) -> None:
        self._tokens.append(tokens_NS.astype(np.int32))
        self._doc_ids.append(doc_ids_N.astype(np.int64))
        self._split.append(split_N.astype(bool))
        if pile_names is not None:
            if self._pile_names is None:
                self._pile_names = []
            self._pile_names.append(pile_names)
        self.n_sequences += tokens_NS.shape[0]

    def append_latents(self, hookpoint: str, z_BSF: Tensor) -> None:
        """Encode CSR rows: nonzero latents per token, sorted feature ids."""
        z_TF = z_BSF.reshape(-1, z_BSF.shape[-1])
        nz_A2 = (z_TF > 0).nonzero()
        counts_T = torch.bincount(nz_A2[:, 0], minlength=z_TF.shape[0])
        vals_A = z_TF[nz_A2[:, 0], nz_A2[:, 1]]
        self._latent_counts[hookpoint].append(counts_T.cpu().numpy().astype(np.int64))
        self._latent_ids[hookpoint].append(nz_A2[:, 1].cpu().numpy().astype(np.int32))
        self._latent_vals[hookpoint].append(vals_A.half().cpu().numpy())

    def append_router(self, layer_idx: int, logits_TE: Tensor) -> None:
        logits_TE = logits_TE.detach().float()
        topk_TK = logits_TE.topk(self.top_k, dim=-1)
        self._topk_ids[layer_idx].append(topk_TK.indices.cpu().numpy().astype(np.int32))
        self._topk_vals[layer_idx].append(topk_TK.values.half().cpu().numpy())
        if self.store_router_logits:
            self._logits[layer_idx].append(logits_TE.half().cpu().numpy())

    def append_residuals(self, hookpoint: str, acts_BSH: Tensor) -> None:
        acts_TH = acts_BSH.detach().reshape(-1, acts_BSH.shape[-1])
        self._residuals[hookpoint].append(acts_TH.float().cpu().numpy())

    def append_expert_writes(
        self,
        layer_idx: int,
        token_idx_R: Tensor,
        expert_idx_R: Tensor,
        writes_RH: Tensor,
        *,
        token_offset: int,
    ) -> None:
        """Append one batch of expert writes.
        ``token_offset`` counts from the first token of this shard.
        """
        if token_idx_R.numel() == 0:
            return
        if self.unembed_weight_VH is None:
            msg = "expert-write capture requires an unembedding matrix"
            raise RuntimeError(msg)
        norms_R = writes_RH.float().norm(dim=-1).half()
        lens_RP = _promoted_token_ids(
            writes_RH,
            self.unembed_weight_VH,
            self.logit_lens_top_p,
            chunk_size=self.logit_lens_chunk_size,
        )
        self._write_token_idx[layer_idx].append(
            (token_idx_R.long() + token_offset).cpu().numpy()
        )
        self._write_expert_idx[layer_idx].append(
            expert_idx_R.cpu().numpy().astype(np.int32)
        )
        self._write_norms[layer_idx].append(norms_R.cpu().numpy())
        self._write_lens[layer_idx].append(lens_RP.cpu().numpy().astype(np.int32))

    # -- flushing --

    def ready(self) -> bool:
        return self.n_sequences >= self.sequences_per_shard

    def flush(self, shard_idx: int, n_sequences: int | None = None) -> None:
        """Write shard ``shard_idx`` of every kind; keep the remainder buffered."""
        take = self.sequences_per_shard if n_sequences is None else n_sequences
        take = min(take, self.n_sequences)
        if take <= 0:
            return
        n_rows = take * self.seq_len
        self._flush_tokens(shard_idx, take)
        if self.store_latents:
            for hp in self.hookpoints:
                self._flush_latents(shard_idx, hp, n_rows)
        for layer in self.router_layers:
            self._flush_router(shard_idx, layer, n_rows)
        if self.store_dense_residuals:
            for hp in self.hookpoints:
                self._flush_residuals(shard_idx, hp, n_rows)
        for layer in self.expert_write_layers:
            self._flush_expert_writes(shard_idx, layer, n_rows)
        self.n_sequences -= take
        logger.info(f"Flushed shard {shard_idx:05d} ({take} sequences)")

    def _flush_tokens(self, shard_idx: int, take: int) -> None:
        tokens_NS = np.concatenate(self._tokens)
        doc_ids_N = np.concatenate(self._doc_ids)
        split_N = np.concatenate(self._split)
        core.write_safetensors(
            _shard_path(self.output_dir / "tokens", shard_idx),
            {
                core.TOKENS: tokens_NS[:take],
                core.TOKENS_DOC_IDS: doc_ids_N[:take],
                TOKENS_SPLIT: split_N[:take],
            },
        )
        if self._pile_names is not None:
            flat = [name for batch in self._pile_names for name in batch]
            core.write_parquet(
                _shard_path(
                    self.output_dir / "tokens" / "pile_set_names", shard_idx
                ).with_suffix(".parquet"),
                [{"pile_set_name": name} for name in flat[:take]],
            )
            self._pile_names = [flat[take:]]
        self._tokens = [tokens_NS[take:]]
        self._doc_ids = [doc_ids_N[take:]]
        self._split = [split_N[take:]]

    def _flush_latents(self, shard_idx: int, hookpoint: str, n_rows: int) -> None:
        counts_T = np.concatenate(self._latent_counts[hookpoint])
        ids_A = np.concatenate(self._latent_ids[hookpoint])
        vals_A = np.concatenate(self._latent_vals[hookpoint])
        csum_T = np.cumsum(counts_T)
        nnz_cut = int(csum_T[n_rows - 1]) if n_rows > 0 else 0
        offsets = np.zeros(n_rows + 1, dtype=np.int64)
        offsets[1:] = csum_T[:n_rows]
        core.write_csr_shard(
            _shard_path(
                self.output_dir / "latents" / core.hookpoint_slug(hookpoint),
                shard_idx,
            ),
            offsets,
            ids_A[:nnz_cut],
            vals_A[:nnz_cut],
        )
        self._latent_counts[hookpoint] = [counts_T[n_rows:]]
        self._latent_ids[hookpoint] = [ids_A[nnz_cut:]]
        self._latent_vals[hookpoint] = [vals_A[nnz_cut:]]

    def _flush_router(self, shard_idx: int, layer_idx: int, n_rows: int) -> None:
        topk_ids_TK = np.concatenate(self._topk_ids[layer_idx])
        topk_vals_TK = np.concatenate(self._topk_vals[layer_idx])
        tensors: dict[str, np.ndarray] = {
            core.ROUTER_TOPK_IDS: topk_ids_TK[:n_rows],
            core.ROUTER_TOPK_VALS: topk_vals_TK[:n_rows],
        }
        if self.store_router_logits:
            logits_TE = np.concatenate(self._logits[layer_idx])
            tensors[core.ROUTER_LOGITS] = logits_TE[:n_rows]
            self._logits[layer_idx] = [logits_TE[n_rows:]]
        core.write_safetensors(
            _shard_path(self.output_dir / "router" / f"layers_{layer_idx}", shard_idx),
            tensors,
        )
        self._topk_ids[layer_idx] = [topk_ids_TK[n_rows:]]
        self._topk_vals[layer_idx] = [topk_vals_TK[n_rows:]]

    def _flush_residuals(self, shard_idx: int, hookpoint: str, n_rows: int) -> None:
        residuals_TH = np.concatenate(self._residuals[hookpoint])
        core.write_safetensors(
            _shard_path(
                self.output_dir / "residuals" / core.hookpoint_slug(hookpoint),
                shard_idx,
            ),
            {RESIDUALS: residuals_TH[:n_rows]},
        )
        self._residuals[hookpoint] = [residuals_TH[n_rows:]]

    def _flush_expert_writes(self, shard_idx: int, layer_idx: int, n_rows: int) -> None:
        if self._write_token_idx[layer_idx]:
            token_idx_R = np.concatenate(self._write_token_idx[layer_idx])
            expert_idx_R = np.concatenate(self._write_expert_idx[layer_idx])
            norms_R = np.concatenate(self._write_norms[layer_idx])
            lens_RP = np.concatenate(self._write_lens[layer_idx])
        else:  # no routed writes in this shard (degenerate; keeps alignment)
            token_idx_R = np.zeros(0, dtype=np.int64)
            expert_idx_R = np.zeros(0, dtype=np.int32)
            norms_R = np.zeros(0, dtype=np.float16)
            lens_RP = np.zeros((0, self.logit_lens_top_p), dtype=np.int32)
        in_shard = token_idx_R < n_rows
        core.write_safetensors(
            _shard_path(
                self.output_dir / "expert_writes" / f"layers_{layer_idx}",
                shard_idx,
            ),
            {
                EXPERT_WRITES_TOKEN_IDX: token_idx_R[in_shard].astype(np.int32),
                EXPERT_WRITES_EXPERT_IDX: expert_idx_R[in_shard],
                EXPERT_WRITES_NORMS: norms_R[in_shard],
                EXPERT_WRITES_LOGIT_LENS: lens_RP[in_shard],
            },
        )
        self._write_token_idx[layer_idx] = [token_idx_R[~in_shard] - n_rows]
        self._write_expert_idx[layer_idx] = [expert_idx_R[~in_shard]]
        self._write_norms[layer_idx] = [norms_R[~in_shard]]
        self._write_lens[layer_idx] = [lens_RP[~in_shard]]


# ---------------------------------------------------------------------------
# Progress / resume
# ---------------------------------------------------------------------------


def _read_progress(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / PROGRESS_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning(f"Ignoring unreadable {path}")
        return None


def _write_progress(
    output_dir: Path,
    *,
    config_hash: str,
    completed_shards: int,
    sequences_written: int,
    total_sequences: int,
) -> None:
    core.write_json(
        output_dir / PROGRESS_FILENAME,
        {
            "config_hash": config_hash,
            "completed_shards": completed_shards,
            "sequences_written": sequences_written,
            "total_sequences": total_sequences,
        },
    )


def _cleanup_stale_shards(
    directories: list[Path], *, keep: int, pattern: str = "shard-*.safetensors"
) -> None:
    """Remove shard files with index >= ``keep`` (shrunk rerun leftovers)."""
    for directory in directories:
        for path in sorted(directory.glob(pattern)):
            try:
                idx = int(path.stem.removeprefix("shard-"))
            except ValueError:
                continue
            if idx >= keep:
                logger.warning(f"Removing stale shard {path}")
                path.unlink()


# ---------------------------------------------------------------------------
# Downstream readers (select.py / explain / probes.py)
# ---------------------------------------------------------------------------


def load_collect_config(input_dir: Path) -> CollectConfig:
    """Reconstruct the CollectConfig of a completed run from its manifest."""
    manifest = core.read_manifest(input_dir)
    if manifest.stage != "collect":
        msg = f"{input_dir} is a {manifest.stage!r} run, not 'collect'"
        raise ValueError(msg)
    return CollectConfig.model_validate(manifest.resolved_config)


def load_collect_info(input_dir: Path) -> dict[str, Any]:
    """Load ``collect_info.json`` (layers, dims, shard layout, store flags)."""
    path = input_dir / COLLECT_INFO_FILENAME
    if not path.exists():
        msg = f"No {COLLECT_INFO_FILENAME} in {input_dir} (incomplete collect run?)"
        raise FileNotFoundError(msg)
    return json.loads(path.read_text())


def load_pile_set_names(input_dir: Path) -> np.ndarray:
    """Return one Pile subset name per sequence, in the same order as the token files.

    Raises when collect did not write those names.
    """
    token_shards = sorted((input_dir / "tokens").glob("shard-*.safetensors"))
    if not token_shards:
        msg = f"No token shards under {input_dir / 'tokens'}"
        raise FileNotFoundError(msg)
    names_dir = input_dir / "tokens" / "pile_set_names"
    parts: list[str] = []
    for shard in token_shards:
        sidecar = names_dir / f"{shard.stem}.parquet"
        if not sidecar.exists():
            msg = (
                f"Missing {sidecar.name}; re-run collect on a dataset with "
                "pile_set_name or meta.pile_set_name"
            )
            raise FileNotFoundError(msg)
        with safe_open(str(shard), framework="np") as handle:
            n_seq = int(handle.get_slice(core.TOKENS).get_shape()[0])
        labels = core.read_parquet(sidecar).column(PILE_SET_NAME_KEY).to_pylist()
        if len(labels) != n_seq:
            msg = f"{sidecar} has {len(labels)} labels for {n_seq} sequences"
            raise ValueError(msg)
        parts.extend(str(label) for label in labels)
    return np.asarray(parts, dtype=object)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_collect(
    config: CollectConfig,
    *,
    run_name: str = "collect",
    logit_lens_top_p: int = 3,
    logit_lens_chunk_size: int = 256,
) -> Path:
    """Run collection and return the output directory.

    Shard numbers already listed in ``progress.json`` are skipped.
    If the manifest already matches this config, the run returns immediately.
    """
    output_dir = config.output_dir
    if core.manifest_matches(output_dir, config, {}):
        logger.info(f"collect up to date -> {output_dir}")
        return output_dir
    core.set_global_seed(config.seed)

    hookpoints = list(config.sae.hookpoints) if config.sae is not None else []
    if config.store_dense_residuals and not hookpoints:
        msg = "store_dense_residuals requires sae.hookpoints to be set"
        raise ValueError(msg)
    router_layers = sorted({router_layer_for_hookpoint(hp) for hp in hookpoints})
    expert_write_layers = router_layers if config.store_expert_writes else []
    if config.store_expert_writes and not router_layers:
        msg = "store_expert_writes requires sae.hookpoints to be set"
        raise ValueError(msg)
    store_latents = config.sae is not None
    activation_hookpoints = (
        hookpoints if (store_latents or config.store_dense_residuals) else []
    )

    seq_len = config.dataset.seq_len
    if config.shard_tokens % seq_len != 0:
        msg = (
            f"shard_tokens ({config.shard_tokens}) must be a multiple of "
            f"seq_len ({seq_len}) so shards align across artifact kinds"
        )
        raise ValueError(msg)
    sequences_per_shard = config.shard_tokens // seq_len
    total_sequences = config.dataset.n_tokens // seq_len
    if total_sequences == 0:
        msg = (
            f"Token budget {config.dataset.n_tokens} < seq_len {seq_len}: "
            "nothing to collect"
        )
        raise ValueError(msg)

    config_hash = core.canonical_config_hash(config)
    progress = _read_progress(output_dir)
    completed_shards = 0
    if progress is not None:
        if progress.get("config_hash") == config_hash:
            completed_shards = int(progress.get("completed_shards", 0))
            if completed_shards > 0:
                logger.info(
                    f"Resuming collect: {completed_shards} shards already written"
                )
        else:
            logger.warning(
                "progress.json config hash mismatch; re-writing from shard 0"
            )
    skip_sequences = completed_shards * sequences_per_shard

    logger.info(f"Loading model backend for {config.model.name}...")
    backend = get_model_backend(config.model)
    model = backend.load_model()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    tokenizer = backend.load_tokenizer()
    backend.validate(model, hookpoints)
    device = next(model.parameters()).device

    saes: dict[str, SAEBackend] = {}
    if config.sae is not None:
        for hp in hookpoints:
            saes[hp] = get_sae_backend(config.sae, hp).to(str(device))

    unembed_weight_VH: Tensor | None = None
    if config.store_expert_writes:
        embed = model.get_output_embeddings()
        if embed is None or not hasattr(embed, "weight"):
            msg = "Model has no output embeddings weight for logit lens"
            raise RuntimeError(msg)
        unembed_weight_VH = embed.weight

    writer = _ShardWriter(
        output_dir,
        hookpoints=hookpoints,
        router_layers=router_layers,
        expert_write_layers=expert_write_layers,
        sequences_per_shard=sequences_per_shard,
        seq_len=seq_len,
        top_k=backend.top_k(model),
        store_latents=store_latents,
        store_router_logits=config.store_router_logits,
        store_dense_residuals=config.store_dense_residuals,
        logit_lens_top_p=logit_lens_top_p,
        logit_lens_chunk_size=logit_lens_chunk_size,
        unembed_weight_VH=unembed_weight_VH,
    )

    captures = _Captures()
    closer = backend.install_captures(
        model,
        activation_hookpoints=activation_hookpoints,
        router_layers=router_layers,
        expert_write_layers=expert_write_layers,
        on_activation=captures.on_activation,
        on_router_logits=captures.on_router_logits,
        on_expert_write=captures.on_expert_write,
    )

    logger.info(
        f"Collect: {total_sequences:,} sequences "
        f"({total_sequences * seq_len:,} tokens), "
        f"{sequences_per_shard} seqs/shard, resume at shard {completed_shards}"
    )
    shard_idx = completed_shards
    n_written = skip_sequences
    batches = iter_sequence_batches(
        config,
        tokenizer,
        n_sequences=total_sequences,
        skip_sequences=skip_sequences,
    )
    total_batches = math.ceil((total_sequences - skip_sequences) / config.batch_size)
    try:
        for tokens_NS, doc_ids_N, split_N, pile_names in tqdm(
            batches, total=total_batches, desc="collect", unit="batch"
        ):
            input_ids_BS = torch.from_numpy(tokens_NS).to(device)
            with torch.no_grad():
                model(input_ids_BS)

                token_offset = writer.n_sequences * seq_len
                writer.append_tokens(tokens_NS, doc_ids_N, split_N, pile_names)
                for hp in activation_hookpoints:
                    acts_BSH = captures.activations.get(hp)
                    if acts_BSH is None:
                        msg = f"Missing activation capture for {hp!r}"
                        raise RuntimeError(msg)
                    if saes:
                        writer.append_latents(hp, saes[hp].encode(acts_BSH))
                    if config.store_dense_residuals:
                        writer.append_residuals(hp, acts_BSH)
                for layer_idx in router_layers:
                    logits_TE = captures.router_logits.get(layer_idx)
                    if logits_TE is None:
                        msg = f"Missing router logits for layer {layer_idx}"
                        raise RuntimeError(msg)
                    writer.append_router(layer_idx, logits_TE)
                for layer_idx in expert_write_layers:
                    packed = captures.expert_writes(layer_idx)
                    if packed is not None:
                        writer.append_expert_writes(
                            layer_idx,
                            packed.token_idx,
                            packed.expert_idx,
                            packed.writes,
                            token_offset=token_offset,
                        )
                captures.clear()
            n_written += tokens_NS.shape[0]
            while writer.ready():
                writer.flush(shard_idx)
                shard_idx += 1
                _write_progress(
                    output_dir,
                    config_hash=config_hash,
                    completed_shards=shard_idx,
                    sequences_written=shard_idx * sequences_per_shard,
                    total_sequences=total_sequences,
                )
        if writer.n_sequences > 0:
            writer.flush(shard_idx, writer.n_sequences)
            shard_idx += 1
            _write_progress(
                output_dir,
                config_hash=config_hash,
                completed_shards=shard_idx,
                sequences_written=n_written,
                total_sequences=total_sequences,
            )
    finally:
        closer()

    shard_dirs = [output_dir / "tokens"]
    _cleanup_stale_shards(
        [output_dir / "tokens" / "pile_set_names"],
        keep=shard_idx,
        pattern="shard-*.parquet",
    )
    if store_latents:
        shard_dirs += [
            output_dir / "latents" / core.hookpoint_slug(hp) for hp in hookpoints
        ]
    shard_dirs += [output_dir / "router" / f"layers_{layer}" for layer in router_layers]
    if config.store_dense_residuals:
        shard_dirs += [
            output_dir / "residuals" / core.hookpoint_slug(hp) for hp in hookpoints
        ]
    shard_dirs += [
        output_dir / "expert_writes" / f"layers_{layer}"
        for layer in expert_write_layers
    ]
    _cleanup_stale_shards(shard_dirs, keep=shard_idx)

    sae_revisions = {hp: sae.revision for hp, sae in saes.items()}
    unique_revisions = set(sae_revisions.values())
    info: dict[str, Any] = {
        "hookpoints": hookpoints,
        "router_layers": {hp: router_layer_for_hookpoint(hp) for hp in hookpoints},
        "n_experts": backend.num_experts(model),
        "top_k": backend.top_k(model),
        "seq_len": seq_len,
        "batch_size": config.batch_size,
        "shard_tokens": config.shard_tokens,
        "sequences_per_shard": sequences_per_shard,
        "n_sequences": total_sequences,
        "n_tokens": total_sequences * seq_len,
        "n_shards": shard_idx,
        "routing_style": backend.routing_style,
        "model_type": backend.model_type,
        "dataset_fingerprint": config.dataset.fingerprint,
        "train_frac": config.train_frac,
        "seed": config.seed,
        "store_router_logits": config.store_router_logits,
        "store_dense_residuals": config.store_dense_residuals,
        "store_expert_writes": config.store_expert_writes,
        "store_pile_set_names": writer._pile_names is not None,
        "logit_lens_top_p": logit_lens_top_p if config.store_expert_writes else None,
        "sae": (
            {
                hp: {
                    "n_features": saes[hp].n_features,
                    "revision": saes[hp].revision,
                }
                for hp in hookpoints
            }
            if saes
            else None
        ),
    }
    core.write_json(output_dir / COLLECT_INFO_FILENAME, info)

    manifest = core.build_manifest(
        stage="collect",
        run_name=run_name,
        config=config,
        dataset_revision=config.dataset.revision,
        model_revision=config.model.revision,
        sae_revision=(unique_revisions.pop() if len(unique_revisions) == 1 else None),
        provenance={
            "sae_hookpoint_revisions": sae_revisions,
            "logit_lens_top_p": logit_lens_top_p,
        },
    )
    core.write_manifest(output_dir, manifest)
    logger.success(f"collect completed -> {output_dir}")
    return output_dir
