"""Readers for the token, latent, router, and expert-write files written by collect.

A file with the same shard number in each of those directories holds the same stretch of
the corpus.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from routerinterp.core import (
    EXPERT_WRITES_EXPERT_IDX,
    EXPERT_WRITES_LOGIT_LENS,
    EXPERT_WRITES_NORMS,
    EXPERT_WRITES_TOKEN_IDX,
    ROUTER_TOPK_IDS,
    ROUTER_TOPK_VALS,
    TOKENS,
    TOKENS_DOC_IDS,
    TOKENS_SPLIT,
    RunManifest,
    csr_shard_paths,
    doc_is_train,
    hookpoint_slug,
    read_csr_shard,
    read_manifest,
    read_safetensors,
    router_layer_for_hookpoint,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

__all__ = [
    "Corpus",
    "ExpertWrites",
    "SparseActivations",
    "collect_expert_activations",
    "collect_feature_activations",
    "load_corpus",
    "load_expert_writes",
    "load_tokenizer",
]


# ---------------------------------------------------------------------------
# Corpus (token shards)
# ---------------------------------------------------------------------------


@dataclass
class Corpus:
    """Token shards + document-level train/test membership."""

    tokens: np.ndarray  # [N, S] int32
    doc_ids: np.ndarray  # [N] int64
    is_train: np.ndarray  # [N] bool

    @property
    def seq_len(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def n_tokens(self) -> int:
        return int(self.tokens.shape[0]) * self.seq_len

    @property
    def flat_tokens(self) -> np.ndarray:
        return self.tokens.reshape(-1)

    def n_windows(self, ctx_len: int) -> int:
        return self.n_tokens // ctx_len

    def window_tokens(self, window_id: int, ctx_len: int) -> np.ndarray:
        """Return the token row of length ``ctx_len`` for corpus window ``window_id``."""
        flat_start = window_id * ctx_len
        doc = flat_start // self.seq_len
        pos = flat_start % self.seq_len
        return self.tokens[doc, pos : pos + ctx_len]

    def windows_are_train(self, window_ids: np.ndarray, ctx_len: int) -> np.ndarray:
        """Return whether each window's sequence is in the train split.

        The label is the sequence that contains the window's first token.
        """
        docs = (window_ids.astype(np.int64) * ctx_len) // self.seq_len
        return self.is_train[docs]


def _derive_splits(doc_ids: np.ndarray, collect_manifest: RunManifest) -> np.ndarray:
    """Derive train membership from the collect manifest's split config."""
    cfg = collect_manifest.resolved_config
    dataset = cfg.get("dataset") or {}
    name = dataset.get("name")
    if not name:
        msg = "Collect manifest is missing dataset.name; cannot derive splits"
        raise ValueError(msg)
    fingerprint = f"{name}@{dataset.get('revision') or 'main'}"
    train_frac = float(cfg.get("train_frac", 0.8))
    seed = int(cfg.get("seed", 0))
    logger.info(
        f"Token shards carry no split column; deriving splits via doc_is_train "
        f"(fingerprint={fingerprint}, seed={seed}, train_frac={train_frac})"
    )
    return np.array(
        [
            doc_is_train(
                int(d), seed=seed, train_frac=train_frac, fingerprint=fingerprint
            )
            for d in doc_ids
        ],
        dtype=bool,
    )


def load_corpus(input_dir: Path, collect_manifest: RunManifest | None = None) -> Corpus:
    """Load ``tokens/shard-*.safetensors`` (tokens/doc_ids[/split]) concatenated."""
    tokens_dir = input_dir / "tokens"
    shard_paths = sorted(tokens_dir.glob("shard-*.safetensors"))
    if not shard_paths:
        msg = f"No token shards found at {tokens_dir}/shard-*.safetensors"
        raise FileNotFoundError(msg)
    tokens_parts: list[np.ndarray] = []
    doc_id_parts: list[np.ndarray] = []
    split_parts: list[np.ndarray] = []
    for path in shard_paths:
        tensors = read_safetensors(path)
        tokens_parts.append(np.asarray(tensors[TOKENS], dtype=np.int32))
        doc_id_parts.append(np.asarray(tensors[TOKENS_DOC_IDS], dtype=np.int64))
        if TOKENS_SPLIT in tensors:
            split_parts.append(np.asarray(tensors[TOKENS_SPLIT]))
    tokens = np.concatenate(tokens_parts)
    doc_ids = np.concatenate(doc_id_parts)
    n_docs = int(tokens.shape[0])
    if len(split_parts) == len(shard_paths):
        split = np.concatenate(split_parts).reshape(-1)
        if split.size == 1:  # shard-level scalar label: broadcast to all docs
            split = np.full(n_docs, split[0])
        if split.size != n_docs:
            msg = f"split column has {split.size} entries, expected {n_docs} docs"
            raise ValueError(msg)
        is_train = split.astype(bool)  # convention: truthy = train
    else:
        if collect_manifest is None:
            collect_manifest = read_manifest(input_dir)
        is_train = _derive_splits(doc_ids, collect_manifest)
    logger.info(
        f"Loaded corpus: {n_docs} docs x {tokens.shape[1]} tokens "
        f"({int(is_train.sum())} train / {int((~is_train).sum())} test docs)"
    )
    return Corpus(tokens=tokens, doc_ids=doc_ids, is_train=is_train)


def load_tokenizer(collect_manifest: RunManifest) -> PreTrainedTokenizerBase:
    """Load the tokenizer named by the collect run's model config."""
    from transformers import AutoTokenizer

    model_cfg = collect_manifest.resolved_config.get("model") or {}
    name = model_cfg.get("name")
    if not name:
        msg = "Collect manifest is missing model.name; cannot load tokenizer"
        raise ValueError(msg)
    return AutoTokenizer.from_pretrained(
        name,
        revision=model_cfg.get("revision"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )


# ---------------------------------------------------------------------------
# Per-id activation scans
# ---------------------------------------------------------------------------


def _layer_shard_dir(input_dir: Path, kind: str, hookpoint: str) -> Path:
    """Return ``<kind>/layers_<idx>`` for the router layer paired with ``hookpoint``."""
    layer_idx = router_layer_for_hookpoint(hookpoint)
    return input_dir / kind / f"layers_{layer_idx}"


def _require_shards(directory: Path, kind: str) -> list[Path]:
    paths = sorted(directory.glob("shard-*.safetensors"))
    if not paths:
        msg = f"No {kind} shards found at {directory}/shard-*.safetensors"
        raise FileNotFoundError(msg)
    return paths


@dataclass(frozen=True, slots=True)
class SparseActivations:
    """Corpus positions and the activation at each position."""

    indices: np.ndarray
    values: np.ndarray


def collect_feature_activations(
    input_dir: Path,
    hookpoint: str,
    feature_ids: Sequence[int],
    *,
    expected_tokens: int | None = None,
) -> dict[int, SparseActivations]:
    """Collect each feature's activations across the corpus.

    Positions count from the start of the corpus, in shard-file order.
    """
    wanted = sorted({int(f) for f in feature_ids})
    if not wanted:
        return {}
    latents_dir = input_dir / "latents" / hookpoint_slug(hookpoint)
    paths = csr_shard_paths(latents_dir)
    if not paths:
        msg = f"No CSR latent shards found at {latents_dir}/shard-*.safetensors"
        raise FileNotFoundError(msg)
    wanted_arr = np.asarray(wanted, dtype=np.int32)
    parts: dict[int, list[SparseActivations]] = {f: [] for f in wanted}
    token_offset = 0
    for path in paths:
        shard = read_csr_shard(path)
        counts = np.diff(shard.offsets)
        rows = np.repeat(np.arange(shard.n_tokens, dtype=np.int64), counts)
        mask = np.isin(shard.feature_ids, wanted_arr)
        sel_rows = rows[mask] + token_offset
        sel_fids = shard.feature_ids[mask]
        sel_vals = shard.values[mask].astype(np.float32)
        for fid in wanted:
            fmask = sel_fids == fid
            if fmask.any():
                parts[fid].append(SparseActivations(sel_rows[fmask], sel_vals[fmask]))
        token_offset += shard.n_tokens
    if expected_tokens is not None and token_offset != expected_tokens:
        msg = (
            f"CSR latents cover {token_offset} tokens but corpus has "
            f"{expected_tokens}; misaligned collect artifacts"
        )
        raise ValueError(msg)
    out: dict[int, SparseActivations] = {}
    for fid in wanted:
        if parts[fid]:
            idx = np.concatenate([part.indices for part in parts[fid]])
            vals = np.concatenate([part.values for part in parts[fid]])
            out[fid] = SparseActivations(idx, vals)
        else:
            out[fid] = SparseActivations(
                np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
            )
    return out


def collect_expert_activations(
    input_dir: Path,
    hookpoint: str,
    expert_ids: Sequence[int],
    *,
    expected_tokens: int | None = None,
) -> dict[int, SparseActivations]:
    """Collect each expert's activations across the corpus.

    Positions count from the start of the corpus, in shard-file order.
    """
    wanted = sorted({int(e) for e in expert_ids})
    if not wanted:
        return {}
    router_dir = _layer_shard_dir(input_dir, "router", hookpoint)
    paths = _require_shards(router_dir, "router")
    wanted_arr = np.asarray(wanted, dtype=np.int32)
    parts: dict[int, list[SparseActivations]] = {e: [] for e in wanted}
    token_offset = 0
    for path in paths:
        tensors = read_safetensors(path)
        topk_ids = np.asarray(tensors[ROUTER_TOPK_IDS], dtype=np.int32)
        topk_vals = np.asarray(tensors[ROUTER_TOPK_VALS], dtype=np.float32)
        n_tokens = int(topk_ids.shape[0])
        mask = np.isin(topk_ids, wanted_arr)
        rows, cols = np.nonzero(mask)
        if rows.size:
            hit_experts = topk_ids[rows, cols]
            hit_vals = topk_vals[rows, cols]
            hit_flat = rows.astype(np.int64) + token_offset
            for eid in wanted:
                emask = hit_experts == eid
                if emask.any():
                    parts[eid].append(
                        SparseActivations(hit_flat[emask], hit_vals[emask])
                    )
        token_offset += n_tokens
    if expected_tokens is not None and token_offset != expected_tokens:
        msg = (
            f"Router shards cover {token_offset} tokens but corpus has "
            f"{expected_tokens}; misaligned collect artifacts"
        )
        raise ValueError(msg)
    out: dict[int, SparseActivations] = {}
    for eid in wanted:
        if parts[eid]:
            idx = np.concatenate([part.indices for part in parts[eid]])
            vals = np.concatenate([part.values for part in parts[eid]])
            order = np.argsort(idx, kind="stable")
            out[eid] = SparseActivations(idx[order], vals[order])
        else:
            out[eid] = SparseActivations(
                np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
            )
    return out


def _routing_values(
    topk_ids: np.ndarray,
    topk_vals: np.ndarray,
    token_idx: np.ndarray,
    expert_idx: np.ndarray,
) -> np.ndarray:
    """Return the router logit for ``expert_idx`` at ``token_idx`` inside this shard.
    0 if that expert was not selected there.
    """
    if token_idx.size == 0:
        return np.empty(0, dtype=np.float32)
    n_tokens = int(topk_ids.shape[0])
    if int(token_idx.min()) < 0 or int(token_idx.max()) >= n_tokens:
        msg = (
            f"expert-write token_idx {int(token_idx.min())}..{int(token_idx.max())} "
            f"falls outside router shard length {n_tokens}"
        )
        raise ValueError(msg)
    chosen = topk_ids[token_idx]
    match = chosen == expert_idx.reshape(-1, 1)
    found = match.any(axis=1)
    cols = np.argmax(match, axis=1)
    gates = np.zeros(token_idx.shape[0], dtype=np.float32)
    if found.any():
        gates[found] = topk_vals[token_idx[found], cols[found]].astype(np.float32)
    n_missing = int((~found).sum())
    if n_missing:
        logger.warning(
            f"{n_missing} expert writes have no matching router top-k entry; "
            "gate set to 0"
        )
    return gates


@dataclass
class ExpertWrites:
    """One row per expert write.
    ``token_indices`` count from the start of the corpus.

    ``gate_values`` are the router logits at those positions, taken from the matching
    router file.
    """

    expert_ids: np.ndarray  # [R] int32
    token_indices: np.ndarray  # [R] int64 flat token positions
    gate_values: np.ndarray  # [R] float32 router top-k logit
    write_norms: np.ndarray  # [R] float32 L2 norm of the expert write
    promoted_token_ids: np.ndarray  # [R, P] int32 logit-lens tokens

    def for_expert(self, expert_id: int) -> ExpertWrites:
        mask = self.expert_ids == expert_id
        return ExpertWrites(
            expert_ids=self.expert_ids[mask],
            token_indices=self.token_indices[mask],
            gate_values=self.gate_values[mask],
            write_norms=self.write_norms[mask],
            promoted_token_ids=self.promoted_token_ids[mask],
        )


def load_expert_writes(
    input_dir: Path,
    hookpoint: str,
    *,
    expected_tokens: int | None = None,
) -> ExpertWrites:
    """Load one hookpoint's expert writes.

    Each file stores positions counting from its own first token.
    Earlier files' lengths are added so the returned positions count from the start of
    the corpus.
    """
    writes_dir = _layer_shard_dir(input_dir, "expert_writes", hookpoint)
    router_dir = _layer_shard_dir(input_dir, "router", hookpoint)
    write_paths = _require_shards(writes_dir, "expert_writes")
    router_paths = {path.name: path for path in _require_shards(router_dir, "router")}
    missing = [path.name for path in write_paths if path.name not in router_paths]
    if missing or len(router_paths) != len(write_paths):
        msg = (
            f"expert_writes shards under {writes_dir} are not aligned with "
            f"router shards under {router_dir} "
            f"(write={[p.name for p in write_paths]}, "
            f"router={sorted(router_paths)})"
        )
        raise FileNotFoundError(msg)

    expert_parts: list[np.ndarray] = []
    token_parts: list[np.ndarray] = []
    gate_parts: list[np.ndarray] = []
    norm_parts: list[np.ndarray] = []
    lens_parts: list[np.ndarray] = []
    token_offset = 0
    for path in write_paths:
        writes = read_safetensors(path)
        for required in (
            EXPERT_WRITES_TOKEN_IDX,
            EXPERT_WRITES_EXPERT_IDX,
            EXPERT_WRITES_NORMS,
            EXPERT_WRITES_LOGIT_LENS,
        ):
            if required not in writes:
                msg = f"{path} is missing required key {required!r}"
                raise ValueError(msg)
        router = read_safetensors(router_paths[path.name])
        topk_ids = np.asarray(router[ROUTER_TOPK_IDS], dtype=np.int32)
        topk_vals = np.asarray(router[ROUTER_TOPK_VALS], dtype=np.float32)
        token_idx = np.asarray(writes[EXPERT_WRITES_TOKEN_IDX], dtype=np.int64)
        expert_idx = np.asarray(writes[EXPERT_WRITES_EXPERT_IDX], dtype=np.int32)
        expert_parts.append(expert_idx)
        token_parts.append(token_idx + token_offset)
        gate_parts.append(_routing_values(topk_ids, topk_vals, token_idx, expert_idx))
        norm_parts.append(np.asarray(writes[EXPERT_WRITES_NORMS], dtype=np.float32))
        lens_parts.append(np.asarray(writes[EXPERT_WRITES_LOGIT_LENS], dtype=np.int32))
        token_offset += int(topk_ids.shape[0])
    if expected_tokens is not None and token_offset != expected_tokens:
        msg = (
            f"Router shards cover {token_offset} tokens but corpus has "
            f"{expected_tokens}; misaligned collect artifacts"
        )
        raise ValueError(msg)

    expert_ids = np.concatenate(expert_parts) if expert_parts else np.empty(0, np.int32)
    token_indices = (
        np.concatenate(token_parts) if token_parts else np.empty(0, np.int64)
    )
    order = np.lexsort((expert_ids, token_indices))
    return ExpertWrites(
        expert_ids=expert_ids[order],
        token_indices=token_indices[order],
        gate_values=(
            np.concatenate(gate_parts)[order] if gate_parts else np.empty(0, np.float32)
        ),
        write_norms=(
            np.concatenate(norm_parts)[order] if norm_parts else np.empty(0, np.float32)
        ),
        promoted_token_ids=(
            np.concatenate(lens_parts)[order]
            if lens_parts
            else np.empty((0, 0), np.int32)
        ),
    )
