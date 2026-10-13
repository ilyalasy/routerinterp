"""Config models, deterministic document splits, run manifests, and artifact IO."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from safetensors.numpy import load_file as st_load_file
from safetensors.numpy import save_file as st_save_file

FORMAT_VERSION = 1

# ---------------------------------------------------------------------------
# Seeds & hashing
# ---------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    """Seed numpy / torch / python-random for reproducible pipeline stages."""
    import random

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_hash(text: str) -> str:
    """Return the SHA256 hex digest of ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_config_hash(config: BaseModel | dict[str, Any]) -> str:
    """Hash a resolved config as canonical JSON, then SHA256."""
    payload = (
        config.model_dump(mode="json") if isinstance(config, BaseModel) else config
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return stable_hash(canonical)


# ---------------------------------------------------------------------------
# Deterministic document-level splits
# ---------------------------------------------------------------------------


def doc_is_train(
    doc_id: int, *, seed: int, train_frac: float, fingerprint: str
) -> bool:
    """Return whether this document is in the train split.

    The answer depends only on the dataset fingerprint, the seed, and the document id.
    It is not stored, and streaming the corpus in a different order does not change it.
    """
    if not 0.0 <= train_frac <= 1.0:
        msg = f"train_frac must be in [0, 1], got {train_frac}"
        raise ValueError(msg)
    digest = hashlib.blake2b(
        f"{fingerprint}:{seed}:{doc_id}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / 2**64 < train_frac


def split_doc_ids(
    n_docs: int, *, seed: int, train_frac: float, fingerprint: str
) -> tuple[np.ndarray, np.ndarray]:
    """Partition ``range(n_docs)`` into sorted train and test id arrays."""
    if n_docs < 0:
        msg = f"n_docs must be non-negative, got {n_docs}"
        raise ValueError(msg)
    ids = np.arange(n_docs, dtype=np.int64)
    if n_docs == 0:
        return ids, ids.copy()
    # Vectorised equivalent of doc_is_train over the id range.
    keys = np.array(
        [
            int.from_bytes(
                hashlib.blake2b(
                    f"{fingerprint}:{seed}:{int(i)}".encode(), digest_size=8
                ).digest(),
                "big",
            )
            for i in ids
        ],
        dtype=np.uint64,
    )
    mask = keys.astype(np.float64) / 2**64 < train_frac
    return ids[mask], ids[~mask]


# ---------------------------------------------------------------------------
# Hookpoint helpers
# ---------------------------------------------------------------------------


def hookpoint_slug(hookpoint: str) -> str:
    """Turn a hookpoint into a filesystem-safe slug, so ``layers.7.mlp`` becomes
    ``layers_7_mlp``.
    """
    return hookpoint.replace("/", "_").replace(".", "_")


def parse_layer_index(hookpoint: str) -> int:
    """Read the layer index in a hookpoint, so ``layers.7.mlp`` is 7."""
    parts = hookpoint.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            return int(parts[i + 1])
    msg = f"Cannot parse layer index from hookpoint {hookpoint!r}"
    raise ValueError(msg)


_LAYERS_MLP_RE = re.compile(r"^layers\.(\d+)\.mlp$")
_LAYERS_RE = re.compile(r"^layers\.(\d+)$")


def normalize_hookpoint(hookpoint: str) -> str:
    """Strip an optional ``model.`` prefix: ``model.layers.7.mlp`` -> ``layers.7.mlp``."""
    return hookpoint.removeprefix("model.")


def residual_is_input(hookpoint: str) -> bool:
    """Return whether the hookpoint captures a module input (``layers.N.mlp``)."""
    return _LAYERS_MLP_RE.fullmatch(normalize_hookpoint(hookpoint)) is not None


def router_layer_for_hookpoint(hookpoint: str) -> int:
    """Return the router layer that belongs with an activation hookpoint.

    ``layers.N.mlp`` is the input of layer N's mixture of experts, so the router is
    layer N.
    ``layers.N`` is the residual after layer N, so the router is layer N + 1.
    """
    hp = normalize_hookpoint(hookpoint)
    if _LAYERS_MLP_RE.fullmatch(hp):
        return parse_layer_index(hp)
    if _LAYERS_RE.fullmatch(hp):
        return parse_layer_index(hp) + 1
    msg = (
        f"Unsupported hookpoint {hookpoint!r}; expected "
        "'layers.N' or 'layers.N.mlp' (optional 'model.' prefix)"
    )
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Machine-local env (.env / ARTIFACTS_ROOT)
# ---------------------------------------------------------------------------


def resolve_run_path(path: Path) -> Path:
    """Prefix a relative path with ``ARTIFACTS_ROOT`` when that variable is set.

    Absolute paths are returned unchanged, so one relative path in a config works on a
    laptop and on the cluster.
    """
    expanded = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if expanded.is_absolute():
        return expanded
    root = os.environ.get("ARTIFACTS_ROOT")
    if not root:
        return expanded
    return Path(os.path.expandvars(os.path.expanduser(root))) / expanded


def _resolve_path_value(value: Any) -> Any:
    if isinstance(value, Path):
        return resolve_run_path(value)
    return value


# ---------------------------------------------------------------------------
# Config models (resolved from configs/*.yaml — no paper constants here)
# ---------------------------------------------------------------------------


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    @field_validator("*", mode="after")
    @classmethod
    def _resolve_artifact_paths(cls, value: Any) -> Any:
        return _resolve_path_value(value)


class ModelConfig(_Config):
    name: str  # HF repo id, e.g. "allenai/OLMoE-1B-7B-0125"
    revision: str | None = None
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    device: str = "auto"
    trust_remote_code: bool = False


class SAESourceConfig(_Config):
    """Where SAE weights come from (exactly one variant populated)."""

    type: Literal["sparsify_wandb", "hf_dictionary_learning", "local"]
    # sparsify_wandb: W&B artifact produced by train_sae.py
    wandb_entity: str | None = None
    wandb_project: str | None = None
    wandb_artifact: str | None = None  # defaults to latest
    # hf_dictionary_learning: HF repo of dictionary_learning checkpoints
    repo_id: str | None = None
    trainer_id: int | None = None
    revision: str | None = None
    # Unpickle ae.pt once into the routerinterp cache. Off unless set:
    # unpickling can execute code, so only enable it for a repo you trust.
    trust_pickle: bool = False
    # local: directory with SafeTensors + config.json
    path: Path | None = None


class SAEConfig(_Config):
    source: SAESourceConfig
    hookpoints: list[str]
    expansion_factor: int | None = None
    k: int | None = None  # top-k latents (Top-K SAE)


class DatasetConfig(_Config):
    name: str  # HF dataset repo id
    revision: str | None = None
    split: str = "train"
    text_column: str = "text"
    n_tokens: int
    seq_len: int
    seed: int = 0
    num_proc: int = 8
    # Equal-count Pile (or mix) subsets instead of a shuffled stream.
    load_stratified: bool = False

    @property
    def fingerprint(self) -> str:
        """Return the id that binds splits to this dataset version and sampling mode."""
        extra = ":stratified" if self.load_stratified else ""
        return f"{self.name}@{self.revision or 'main'}{extra}"


class CollectConfig(_Config):
    model: ModelConfig
    dataset: DatasetConfig
    sae: SAEConfig | None = None
    output_dir: Path
    batch_size: int = 32
    shard_tokens: int = 250_000  # tokens per resumable CSR shard
    store_dense_residuals: bool = False  # only for probe runs
    store_router_logits: bool = False  # full logits; else top-k only
    store_expert_writes: bool = False  # esb example writer
    train_frac: float = 0.8
    seed: int = 0


class SelectConfig(_Config):
    method: Literal["routing_margin", "rho_useful", "cosine"]
    input_dir: Path  # collect run dir
    output_dir: Path
    hookpoints: list[str]
    n: int = 45  # features per expert (paper operating point)
    lambda_: float = 1.0  # routing-margin S = A - lambda * L
    exclusive: bool = True  # exclusive feature assignment across experts
    batch_size: int | None = None
    n_tokens: int | None = None  # cap for attribution pass
    seed: int = 0


class ExplainConfig(_Config):
    method: Literal[
        "feature_autointerp",
        "routerinterp",
        "unigram_autointerp",
        "bigram_autointerp",
        "unigram_lookup",
        "bigram_lookup",
        "expert_activation",
        "esb",
    ]
    input_dir: Path  # collect run dir (evidence source)
    selection_dir: Path  # select run dir
    output_dir: Path
    hookpoints: list[str]
    explainer_model: str
    train_positives: int = 15
    train_negatives: int = 5
    ctx_len: int = 32
    max_concurrent: int = 10
    example_seed: int = 0


class EvaluateConfig(_Config):
    explanations_dir: Path  # explain run dir
    output_dir: Path
    scorer_model: str
    eval_positives: int = 15  # held-out leakage-free windows
    eval_negatives: int = 85
    threshold: int = 2  # 0-4 scorer binarisation threshold
    scorer_batch_size: int = 8
    max_concurrent: int = 10
    scoring_seed: int = 0


NgramType = Literal["unigram", "bigram"]


class ProbeConfig(_Config):
    input_dir: Path  # collect run dir with dense residuals
    output_dir: Path
    bases: list[Literal["sae", "neuron", "pca", "ngram"]] = ["sae"]
    hookpoints: list[str]
    n_tokens: int = 1_000_000
    train_frac: float = 0.8
    min_tokens_per_expert: int = 100
    seed: int = 0
    pca_dim: int | None = None  # None -> hidden size
    ngram_type: NgramType = "bigram"
    ngram_artifact_top_k: int = 10_000  # per-expert entries in the .npz lookup
    solver: Literal["liblinear", "saga"] = "liblinear"  # sparse-friendly
    logreg_c: float = 1.0
    max_iter: int = 1_000
    # Neuron/PCA: keep this many activations per token (paper: SAE s).
    # None -> SAE k from the parent collect config.
    keep_top_s: int | None = None


# ---------------------------------------------------------------------------
# Run manifests
# ---------------------------------------------------------------------------


class RunManifest(BaseModel):
    """The ``manifest.json`` written at the root of a run directory.

    A stage skips its work when an existing manifest has the same config hash and the
    same hashes for the runs it reads.
    """

    manifest_version: int = FORMAT_VERSION
    stage: str  # collect | select | explain | evaluate | probe | plot
    run_name: str
    config_hash: str
    resolved_config: dict[str, Any]
    dataset_revision: str | None = None
    model_revision: str | None = None
    sae_revision: str | None = None
    parents: dict[str, str] = Field(default_factory=dict)  # stage -> config_hash
    provenance: dict[str, Any] = Field(default_factory=dict)  # stage-specific metadata
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def matches(
        self, config: BaseModel | dict[str, Any], parents: dict[str, str]
    ) -> bool:
        return (
            self.config_hash == canonical_config_hash(config)
            and self.parents == parents
        )


MANIFEST_FILENAME = "manifest.json"


def write_manifest(run_dir: Path, manifest: RunManifest) -> Path:
    """Write ``manifest.json`` and log the run when ``WANDB_PROJECT`` is set."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / MANIFEST_FILENAME
    path.write_text(manifest.model_dump_json(indent=2) + "\n")
    from routerinterp.wandb_io import wandb_log

    wandb_log(run_dir)
    return path


def read_manifest(run_dir: Path) -> RunManifest:
    path = run_dir / MANIFEST_FILENAME
    if not path.exists():
        msg = f"No {MANIFEST_FILENAME} in {run_dir}"
        raise FileNotFoundError(msg)
    return RunManifest.model_validate_json(path.read_text())


def manifest_matches(
    run_dir: Path, config: BaseModel | dict[str, Any], parents: dict[str, str]
) -> bool:
    """Return whether an up-to-date manifest for this invocation already exists."""
    try:
        return read_manifest(run_dir).matches(config, parents)
    except (FileNotFoundError, ValueError):
        return False


def build_manifest(
    stage: str,
    run_name: str,
    config: BaseModel,
    *,
    parents: dict[str, str] | None = None,
    dataset_revision: str | None = None,
    model_revision: str | None = None,
    sae_revision: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> RunManifest:
    return RunManifest(
        stage=stage,
        run_name=run_name,
        config_hash=canonical_config_hash(config),
        resolved_config=config.model_dump(mode="json"),
        dataset_revision=dataset_revision,
        model_revision=model_revision,
        sae_revision=sae_revision,
        parents=parents or {},
        provenance=provenance or {},
    )


# ---------------------------------------------------------------------------
# SafeTensors / Parquet / JSONL artifact contracts
# ---------------------------------------------------------------------------


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    st_save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, str(path))
    return path


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    return st_load_file(str(path))


def write_parquet(path: Path, records: list[dict[str, Any]] | pa.Table) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = records if isinstance(records, pa.Table) else pa.Table.from_pylist(records)
    pq.write_table(table, path)
    return path


def read_parquet(path: Path) -> pa.Table:
    return pq.read_table(path)


def write_json(path: Path, payload: BaseModel | dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        payload.model_dump_json(indent=2)
        if isinstance(payload, BaseModel)
        else json.dumps(payload, indent=2)
    )
    path.write_text(text + "\n")
    return path


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# --- CSR latent shards (collect.py writes; select.py / probes.py read) ---
#
# One SafeTensors file per shard with keys:
#   offsets     int64  [n_tokens + 1]   row pointers into feature_ids/values
#   feature_ids int32  [nnz]            SAE latent ids, sorted within a row
#   values      float16 [nnz]           latent activations (> 0)
# Tokens are flattened document-major: token t = doc d, position p has
# t = d * seq_len + p. Shards cover contiguous token ranges -> resumable.

CSR_OFFSETS = "offsets"
CSR_FEATURE_IDS = "feature_ids"
CSR_VALUES = "values"


def write_csr_shard(
    path: Path,
    offsets: np.ndarray,
    feature_ids: np.ndarray,
    values: np.ndarray,
) -> Path:
    if offsets.dtype != np.int64:
        offsets = offsets.astype(np.int64)
    if feature_ids.dtype != np.int32:
        feature_ids = feature_ids.astype(np.int32)
    if values.dtype not in (np.float16, np.float32):
        values = values.astype(np.float16)
    if offsets.ndim != 1 or offsets[-1] != feature_ids.shape[0]:
        msg = "CSR offsets inconsistent with nnz"
        raise ValueError(msg)
    return write_safetensors(
        path,
        {CSR_OFFSETS: offsets, CSR_FEATURE_IDS: feature_ids, CSR_VALUES: values},
    )


class CSRShard:
    """In-memory CSR matrix over tokens; row ``t`` = active latents at token t."""

    def __init__(
        self, offsets: np.ndarray, feature_ids: np.ndarray, values: np.ndarray
    ):
        self.offsets = offsets
        self.feature_ids = feature_ids
        self.values = values

    @property
    def n_tokens(self) -> int:
        return int(self.offsets.shape[0] - 1)

    @property
    def nnz(self) -> int:
        return int(self.feature_ids.shape[0])

    def row(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        start, end = int(self.offsets[t]), int(self.offsets[t + 1])
        return self.feature_ids[start:end], self.values[start:end]

    def to_torch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.offsets),
            torch.from_numpy(self.feature_ids),
            torch.from_numpy(self.values),
        )


def read_csr_shard(path: Path) -> CSRShard:
    tensors = read_safetensors(path)
    return CSRShard(tensors[CSR_OFFSETS], tensors[CSR_FEATURE_IDS], tensors[CSR_VALUES])


def csr_shard_paths(latents_dir: Path) -> list[Path]:
    """Return one hookpoint's shard files, in token order."""
    return sorted(latents_dir.glob("shard-*.safetensors"))


# --- Router shards ---
# keys: topk_ids int32 [T, K], topk_vals float16 [T, K]; optional logits float16 [T, E]
ROUTER_TOPK_IDS = "topk_ids"
ROUTER_TOPK_VALS = "topk_vals"
ROUTER_LOGITS = "logits"

# --- Tokens ---
# keys: tokens int32 [N, S], doc_ids int64 [N], split bool [N] (True = train)
TOKENS = "tokens"
TOKENS_DOC_IDS = "doc_ids"
TOKENS_SPLIT = "split"

# --- Expert-write shards (collect.py writes under expert_writes/layers_<idx>/) ---
# token_idx is shard-local: a flat index within that shard's token rows.
EXPERT_WRITES_TOKEN_IDX = "token_idx"  # int32 [R]
EXPERT_WRITES_EXPERT_IDX = "expert_idx"  # int32 [R]
EXPERT_WRITES_NORMS = "norms"  # float16 [R] L2 norm of the unscaled expert write
EXPERT_WRITES_LOGIT_LENS = "logit_lens_ids"  # int32 [R, P] promoted token ids

# --- Score matrices (select.py writes) ---
# scores safetensors: scores float32 [E, F]; diagnostics: coverage/effect [E, F]
SCORES = "scores"
COVERAGE = "coverage"
EFFECT = "effect"


# ---------------------------------------------------------------------------
# Tidy metrics (plot.py input)
# ---------------------------------------------------------------------------

TIDY_COLUMNS = [
    "figure",  # e.g. "feature_clustering", "explanations_eval", "explanation_length"
    "panel",  # subplot / row group within the figure
    "series",  # legend entry: method / basis name
    "x",  # x value (layer, n, sparsity, ...)
    "metric",  # e.g. "f1", "precision", "recall", "ari"
    "value",  # float
    "split",  # train | test | ""
]


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)
