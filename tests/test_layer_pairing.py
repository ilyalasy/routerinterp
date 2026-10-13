"""Which router layer an SAE hookpoint belongs with, and that the two capture points do not
share a finished run.

``layers.N.mlp`` is the input of layer N's mixture of experts, so the router is N.
``layers.N`` is the residual after layer N, so the router is N + 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from routerinterp import artifacts, collect, core, probes
from routerinterp.collect import load_collect_info, run_collect
from routerinterp.core import (
    CollectConfig,
    DatasetConfig,
    ModelConfig,
    ProbeConfig,
    SAEConfig,
    SAESourceConfig,
    build_manifest,
    canonical_config_hash,
    hookpoint_slug,
    manifest_matches,
    parse_layer_index,
    router_layer_for_hookpoint,
    write_manifest,
)
from routerinterp.probes import _load_labels, _load_latents, run_probes
from routerinterp.select import compute_rho_useful_scores

# Toy router fill uses the router index as the sole top-1 expert id.
_N_EXPERTS = 32
_LATENT_FEATURE = {
    "layers.7.mlp": 1,
    "layers.7": 2,
    "layers.19": 3,
}

# hookpoint, router layer, parse_layer_index, slug (None when prefixed).
# One MLP input, one residual, one ``model.`` prefix.
_PAIRING = [
    ("layers.7.mlp", 7, 7, "layers_7_mlp"),
    ("layers.7", 8, 7, "layers_7"),
    ("model.layers.7", 8, 7, None),
]


def _collect_config(
    output_dir: Path,
    hookpoints: list[str],
    *,
    residuals: bool = False,
) -> CollectConfig:
    return CollectConfig(
        model=ModelConfig(name="toy-model"),
        dataset=DatasetConfig(name="toy-corpus", n_tokens=2, seq_len=2),
        sae=SAEConfig(
            source=SAESourceConfig(type="local", path=output_dir / "unused-sae"),
            hookpoints=list(hookpoints),
        ),
        output_dir=output_dir,
        batch_size=1,
        shard_tokens=2,
        store_dense_residuals=residuals,
        seed=0,
    )


@pytest.mark.parametrize(
    ("hookpoint", "router_layer", "layer_index", "slug"),
    _PAIRING,
)
def test_hookpoint_pairs_with_router_layer(
    hookpoint: str,
    router_layer: int,
    layer_index: int,
    slug: str | None,
) -> None:
    normalized = hookpoint.removeprefix("model.")
    if normalized.endswith(".mlp"):
        assert router_layer == layer_index
    else:
        assert router_layer == layer_index + 1
    assert router_layer_for_hookpoint(hookpoint) == router_layer
    assert parse_layer_index(hookpoint) == layer_index
    if slug is not None:
        assert hookpoint_slug(hookpoint) == slug


def test_unsupported_hookpoint_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported hookpoint"):
        router_layer_for_hookpoint("layers.7.resid")


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Linear(1, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids


class _ToyBackend:
    """Satisfies the collect loop without loading weights."""

    model_type = "toy"
    routing_style = "olmoe"

    def load_model(self) -> _ToyModel:
        return _ToyModel()

    def load_tokenizer(self) -> object:
        return object()

    def validate(self, model: object, hookpoints: list[str]) -> None:
        return None

    def num_experts(self, model: object) -> int:
        return _N_EXPERTS

    def top_k(self, model: object) -> int:
        return 1

    def install_captures(
        self,
        model: _ToyModel,
        *,
        activation_hookpoints: list[str],
        router_layers: list[int],
        expert_write_layers: list[int],
        on_activation,
        on_router_logits,
        on_expert_write,
    ):
        del expert_write_layers, on_expert_write

        def _pre(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            input_ids = args[0]
            batch, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
            hidden = torch.ones(batch, seq_len, 4)
            for hookpoint in activation_hookpoints:
                on_activation(hookpoint, hidden)
            for layer in router_layers:
                logits = torch.full((batch * seq_len, _N_EXPERTS), -1.0)
                logits[:, int(layer)] = 10.0
                on_router_logits(int(layer), logits)

        handle = model.register_forward_pre_hook(_pre)
        return handle.remove


class _ToySAE:
    def __init__(self, hookpoint: str) -> None:
        self.hookpoint = hookpoint
        self.n_features = 8
        self.revision = "toy"

    def encode(self, acts_BSH: torch.Tensor) -> torch.Tensor:
        latents = torch.zeros(*acts_BSH.shape[:-1], self.n_features)
        latents[..., _LATENT_FEATURE[self.hookpoint]] = 1
        return latents

    def to(self, device: str) -> _ToySAE:
        del device
        return self


def _batches(config, tokenizer, *, n_sequences: int, skip_sequences: int = 0):
    del tokenizer, skip_sequences
    if n_sequences <= 0:
        return
    seq_len = config.dataset.seq_len
    yield (
        np.ones((1, seq_len), dtype=np.int32),
        np.array([0], dtype=np.int64),
        np.array([True]),
        None,
    )


def test_collect_writes_paired_router_layers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hookpoints = ["layers.7.mlp", "layers.7", "layers.19"]
    output = tmp_path / "collect"
    config = _collect_config(output, hookpoints, residuals=True)
    monkeypatch.setattr(collect, "get_model_backend", lambda _config: _ToyBackend())
    monkeypatch.setattr(
        collect,
        "get_sae_backend",
        lambda _config, hookpoint: _ToySAE(hookpoint),
    )
    monkeypatch.setattr(collect, "iter_sequence_batches", _batches)

    assert run_collect(config) == output

    assert {path.name for path in (output / "latents").iterdir()} == {
        "layers_7_mlp",
        "layers_7",
        "layers_19",
    }
    assert {path.name for path in (output / "residuals").iterdir()} == {
        "layers_7_mlp",
        "layers_7",
        "layers_19",
    }
    # layers.7 must not be stored as router 7, and layers.19 not as router 19.
    # router/layers_7 is only the mlp hook's router.
    assert {path.name for path in (output / "router").iterdir()} == {
        "layers_7",
        "layers_8",
        "layers_20",
    }
    assert not (output / "router" / "layers_19").exists()

    info = load_collect_info(output)
    assert info["router_layers"] == {
        "layers.7.mlp": 7,
        "layers.7": 8,
        "layers.19": 20,
    }
    assert info["router_layers"]["layers.7"] != parse_layer_index("layers.7")
    assert info["router_layers"]["layers.19"] != parse_layer_index("layers.19")

    for hookpoint, feature in _LATENT_FEATURE.items():
        shard = core.read_csr_shard(
            output / "latents" / hookpoint_slug(hookpoint) / "shard-00000.safetensors"
        )
        assert set(shard.feature_ids.tolist()) == {feature}
    for layer in (7, 8, 20):
        tensors = core.read_safetensors(
            output / "router" / f"layers_{layer}" / "shard-00000.safetensors"
        )
        assert set(tensors[core.ROUTER_TOPK_IDS].reshape(-1).tolist()) == {layer}

    assert manifest_matches(output, config, {})


def _write_router(root: Path, layer: int, expert: int) -> None:
    core.write_safetensors(
        root / "router" / f"layers_{layer}" / "shard-00000.safetensors",
        {
            core.ROUTER_TOPK_IDS: np.array([[expert]], dtype=np.int32),
            core.ROUTER_TOPK_VALS: np.array([[1.0]], dtype=np.float16),
        },
    )


def _write_latents(root: Path, slug: str, feature: int) -> None:
    core.write_csr_shard(
        root / "latents" / slug / "shard-00000.safetensors",
        np.array([0, 1], dtype=np.int64),
        np.array([feature], dtype=np.int32),
        np.array([1.0], dtype=np.float16),
    )


def _ambiguous_tree(tmp_path: Path) -> Path:
    """Build a directory where both router layers exist, plus decoy directories."""
    root = tmp_path / "collect"
    _write_router(root, 7, 70)
    _write_router(root, 8, 80)
    _write_router(root, 19, 19)
    _write_latents(root, "layers_7_mlp", 11)
    _write_latents(root, "layers_7", 22)
    _write_latents(root, "layers_8", 30)
    core.write_safetensors(
        root / "tokens" / "shard-00000.safetensors",
        {
            core.TOKENS: np.array([[1]], dtype=np.int32),
            core.TOKENS_DOC_IDS: np.array([0], dtype=np.int64),
            core.TOKENS_SPLIT: np.array([True]),
        },
    )
    core.write_json(
        root / collect.COLLECT_INFO_FILENAME,
        {
            "n_experts": 128,
            "top_k": 1,
            "batch_size": 1,
            "sae": {
                "layers.7.mlp": {"n_features": 64},
                "layers.7": {"n_features": 64},
            },
        },
    )
    write_manifest(
        root,
        build_manifest("collect", "collect", _collect_config(root, ["layers.7.mlp"])),
    )
    return root


def _probe_sample() -> probes.ProbeSample:
    corpus = artifacts.Corpus(
        tokens=np.zeros((1, 1), dtype=np.int32),
        doc_ids=np.zeros(1, dtype=np.int64),
        is_train=np.ones(1, dtype=bool),
    )
    return probes.ProbeSample(
        corpus=corpus,
        train_mask_T=np.ones(1, dtype=bool),
        selected=np.zeros(1, dtype=np.int64),
    )


def _spy(monkeypatch: pytest.MonkeyPatch, module: object, name: str) -> list[Path]:
    opened: list[Path] = []
    real = getattr(module, name)

    def wrapped(path: Path, *args: object, **kwargs: object):
        opened.append(Path(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(module, name, wrapped)
    return opened


def _parents(paths: list[Path]) -> list[str]:
    return [path.parent.name for path in paths]


def test_readers_open_the_paired_router_and_latent_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _ambiguous_tree(tmp_path)
    sample = _probe_sample()
    expert_reads = _spy(monkeypatch, artifacts, "read_safetensors")
    latent_reads = _spy(monkeypatch, artifacts, "read_csr_shard")
    core_reads = _spy(monkeypatch, core, "read_safetensors")

    cases = (
        ("layers.7.mlp", 7, "layers_7", "layers_7_mlp", 70, 11),
        ("layers.7", 8, "layers_8", "layers_7", 80, 22),
    )
    for hookpoint, layer, router_dir, latent_dir, expert, feature in cases:
        mark = len(expert_reads)
        acts = artifacts.collect_expert_activations(root, hookpoint, [70, 80])
        assert _parents(expert_reads[mark:]) == [router_dir]
        assert acts[expert].indices.size == 1
        other = 80 if expert == 70 else 70
        assert acts[other].indices.size == 0

        mark = len(latent_reads)
        feats = artifacts.collect_feature_activations(root, hookpoint, [11, 22, 30])
        assert _parents(latent_reads[mark:]) == [latent_dir]
        assert feats[feature].indices.size == 1
        for decoy in (11, 22, 30):
            if decoy != feature:
                assert feats[decoy].indices.size == 0

        mark = len(core_reads)
        labels, _top_k, _n_experts = _load_labels(
            root, router_layer_for_hookpoint(hookpoint), sample
        )
        assert router_layer_for_hookpoint(hookpoint) == layer
        assert _parents(core_reads[mark:]) == [router_dir]
        assert set(np.flatnonzero(labels.any(axis=0)).tolist()) == {expert}

        mark = len(core_reads)
        matrix = _load_latents(root, hookpoint, sample)
        assert _parents(core_reads[mark:]) == [latent_dir]
        assert matrix.indices.tolist() == [feature]

    opened_routers = {
        path.parent.name for path in expert_reads + core_reads if "router" in path.parts
    }
    assert "layers_19" not in opened_routers
    assert "layers_8" not in {path.parent.name for path in latent_reads}

    mark = len(core_reads)
    run_probes(
        ProbeConfig(
            input_dir=root,
            output_dir=tmp_path / "probes",
            bases=[],
            hookpoints=["layers.7.mlp", "layers.7"],
            n_tokens=1,
        )
    )
    assert _parents(core_reads[mark:]) == ["layers_7", "layers_8"]

    mark = len(core_reads)
    mlp_scores = compute_rho_useful_scores(root, "layers.7.mlp")
    rho_paths = core_reads[mark:]
    assert float(mlp_scores[70, 11]) == pytest.approx(1.0)
    assert float(mlp_scores[80, 11]) == pytest.approx(-1.0)
    assert [path.parent.name for path in rho_paths if "router" in path.parts] == [
        "layers_7"
    ]
    assert [path.parent.name for path in rho_paths if "latents" in path.parts] == [
        "layers_7_mlp"
    ]

    mark = len(core_reads)
    residual_scores = compute_rho_useful_scores(root, "layers.7")
    rho_paths = core_reads[mark:]
    assert float(residual_scores[80, 22]) == pytest.approx(1.0)
    assert float(residual_scores[70, 22]) == pytest.approx(-1.0)
    assert [path.parent.name for path in rho_paths if "router" in path.parts] == [
        "layers_8"
    ]
    assert [path.parent.name for path in rho_paths if "latents" in path.parts] == [
        "layers_7"
    ]


def test_rho_useful_ignores_a_test_split_fire(tmp_path: Path) -> None:
    """Check that a huge latent on the test split leaves the train-only score unchanged."""
    root = tmp_path / "collect"
    hookpoint = "layers.7.mlp"
    n_experts, n_features = 4, 4
    # Sequence 0 is train and selects expert 0. Sequence 1 is test, selects
    # expert 3, and fires a huge copy of feature 1 plus a test-only feature.
    core.write_safetensors(
        root / "tokens" / "shard-00000.safetensors",
        {
            core.TOKENS: np.array([[1], [2]], dtype=np.int32),
            core.TOKENS_DOC_IDS: np.array([0, 1], dtype=np.int64),
            core.TOKENS_SPLIT: np.array([True, False]),
        },
    )
    router_layer = router_layer_for_hookpoint(hookpoint)
    core.write_safetensors(
        root / "router" / f"layers_{router_layer}" / "shard-00000.safetensors",
        {
            core.ROUTER_TOPK_IDS: np.array([[0], [3]], dtype=np.int32),
            core.ROUTER_TOPK_VALS: np.array([[1.0], [1.0]], dtype=np.float16),
        },
    )
    core.write_csr_shard(
        root / "latents" / hookpoint_slug(hookpoint) / "shard-00000.safetensors",
        np.array([0, 1, 3], dtype=np.int64),
        np.array([1, 1, 2], dtype=np.int32),
        np.array([1.0, 1.0e4, 1.0e4], dtype=np.float32),
    )
    core.write_json(
        root / collect.COLLECT_INFO_FILENAME,
        {
            "n_experts": n_experts,
            "top_k": 1,
            "sae": {hookpoint: {"n_features": n_features}},
        },
    )

    scores = compute_rho_useful_scores(root, hookpoint, device="cpu")
    assert tuple(scores.shape) == (n_experts, n_features)
    # Train token: feature 1 fires only when expert 0 is selected.
    assert float(scores[0, 1]) == pytest.approx(1.0)
    for expert in range(1, n_experts):
        assert float(scores[expert, 1]) == pytest.approx(-1.0)
    # Feature 2 fires only on the test sequence.
    assert float(scores[:, 2].abs().max()) == pytest.approx(0.0)


def test_collect_cache_identity_splits_hookpoint_forms(tmp_path: Path) -> None:
    output = tmp_path / "collect"
    mlp = _collect_config(output, ["layers.7.mlp"])
    residual = _collect_config(output, ["layers.7"])
    mlp_dump = mlp.model_dump(mode="json")
    residual_dump = residual.model_dump(mode="json")
    assert mlp_dump["sae"]["hookpoints"] == ["layers.7.mlp"]
    assert residual_dump["sae"]["hookpoints"] == ["layers.7"]
    mlp_dump["sae"]["hookpoints"] = residual_dump["sae"]["hookpoints"]
    assert mlp_dump == residual_dump

    assert canonical_config_hash(mlp) != canonical_config_hash(residual)
    cache = tmp_path / "cache"
    write_manifest(cache, build_manifest("collect", "collect", mlp))
    assert manifest_matches(cache, mlp, {})
    assert manifest_matches(cache, residual, {}) is False
