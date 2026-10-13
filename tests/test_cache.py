"""Finished work is skipped, and a changed input runs again."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from routerinterp import collect, core
from routerinterp.artifacts import Corpus, SparseActivations
from routerinterp.experiments import feature_clustering, feature_selection_sweep
from routerinterp.experiments.feature_selection_sweep import probe_selections
from routerinterp.explain import evaluate, explain
from routerinterp.explain.schema import ChatResponse, EvalWindow, ExplanationKey
from routerinterp.probes import run_probes
from routerinterp.select import run_select

HP = "layers.0.mlp"
_SENTINEL = b"finished-shard-0"


def _collect_config(
    output_dir: Path, *, n_tokens: int = 4, seq_len: int = 2
) -> core.CollectConfig:
    return core.CollectConfig(
        model=core.ModelConfig(name="toy/model"),
        dataset=core.DatasetConfig(name="toy/pile", n_tokens=n_tokens, seq_len=seq_len),
        output_dir=output_dir,
        batch_size=1,
        shard_tokens=n_tokens if n_tokens == seq_len else seq_len,
    )


def _write_collect_manifest(config: core.CollectConfig) -> core.RunManifest:
    manifest = core.build_manifest(stage="collect", run_name="collect", config=config)
    core.write_manifest(config.output_dir, manifest)
    return manifest


def _patch_collect(monkeypatch):
    """Stand in for the model and the dataset stream.
    Record loads and skips.
    """
    loads: list[int] = []
    skips: list[int] = []

    class _Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return input_ids

    class _Backend:
        model_type = "toy"
        routing_style = "olmoe"

        def load_model(self) -> _Toy:
            loads.append(1)
            return _Toy()

        def load_tokenizer(self) -> object:
            return object()

        def validate(self, model: object, hookpoints: list[str]) -> None:
            return None

        def install_captures(self, *args: object, **kwargs: object):
            return lambda: None

        def num_experts(self, model: object) -> int:
            return 1

        def top_k(self, model: object) -> int:
            return 1

    def _batches(config, tokenizer, *, n_sequences: int, skip_sequences: int = 0):
        skips.append(skip_sequences)
        seq_len = config.dataset.seq_len
        for i in range(n_sequences - skip_sequences):
            yield (
                np.full((1, seq_len), i + 1, dtype=np.int64),
                np.asarray([i], dtype=np.int64),
                np.ones(1, dtype=bool),
                None,
            )

    monkeypatch.setattr(collect, "get_model_backend", lambda model: _Backend())
    monkeypatch.setattr(collect, "iter_sequence_batches", _batches)
    return loads, skips


def _sentinel_shard(output_dir: Path) -> Path:
    path = output_dir / "tokens" / "shard-00000.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_SENTINEL)
    return path


def _write_progress(output_dir: Path, config_hash: str, completed_shards: int) -> None:
    core.write_json(
        output_dir / collect.PROGRESS_FILENAME,
        {
            "config_hash": config_hash,
            "completed_shards": completed_shards,
            "sequences_written": completed_shards,
            "total_sequences": 2,
        },
    )


def test_collect_matching_manifest_does_not_load_model(
    tmp_path: Path, monkeypatch
) -> None:
    loads, skips = _patch_collect(monkeypatch)
    config = _collect_config(tmp_path / "collect")
    manifest_path = _write_collect_manifest(config)
    shard = _sentinel_shard(config.output_dir)
    _write_progress(config.output_dir, manifest_path.config_hash, 1)
    manifest_bytes = (config.output_dir / "manifest.json").read_bytes()

    assert collect.run_collect(config) == config.output_dir
    assert loads == []
    assert skips == []
    assert shard.read_bytes() == _SENTINEL
    assert (config.output_dir / "manifest.json").read_bytes() == manifest_bytes


def test_collect_resumes_without_rewriting_finished_shards(
    tmp_path: Path, monkeypatch
) -> None:
    loads, skips = _patch_collect(monkeypatch)
    config = _collect_config(tmp_path / "collect")
    shard = _sentinel_shard(config.output_dir)
    _write_progress(config.output_dir, core.canonical_config_hash(config), 1)

    collect.run_collect(config)

    # One shard was already done, so the stream starts at the next shard.
    assert skips == [1]
    assert loads == [1]
    assert shard.read_bytes() == _SENTINEL
    assert (config.output_dir / "tokens" / "shard-00001.safetensors").is_file()


def test_collect_progress_hash_mismatch_starts_at_shard_zero(
    tmp_path: Path, monkeypatch
) -> None:
    loads, skips = _patch_collect(monkeypatch)
    config = _collect_config(tmp_path / "collect")
    shard = _sentinel_shard(config.output_dir)
    _write_progress(config.output_dir, "different-config-hash", 1)

    collect.run_collect(config)

    assert skips == [0]
    assert loads == [1]
    assert shard.read_bytes() != _SENTINEL


def _select_setup(tmp_path: Path) -> tuple[core.SelectConfig, core.CollectConfig, str]:
    collect_dir = tmp_path / "collect"
    collect_cfg = _collect_config(collect_dir)
    collect_manifest = _write_collect_manifest(collect_cfg)
    core.write_json(collect_dir / collect.COLLECT_INFO_FILENAME, {"batch_size": 1})
    select_cfg = core.SelectConfig(
        method="rho_useful",
        input_dir=collect_dir,
        output_dir=tmp_path / "select",
        hookpoints=[HP],
        n=1,
    )
    return select_cfg, collect_cfg, collect_manifest.config_hash


def _patch_select_scores(monkeypatch) -> list[str]:
    calls: list[str] = []

    def _scores(input_dir, hookpoint, **kwargs):
        calls.append(hookpoint)
        return torch.zeros(2, 3)

    monkeypatch.setattr(select_module(), "compute_rho_useful_scores", _scores)
    return calls


def select_module():
    import routerinterp.select as select

    return select


def _write_select_manifest(config: core.SelectConfig, collect_hash: str) -> None:
    core.write_manifest(
        config.output_dir,
        core.build_manifest(
            stage="select",
            run_name="select",
            config=config,
            parents={"collect": collect_hash},
        ),
    )


def test_select_skips_when_manifest_and_collect_parent_match(
    tmp_path: Path, monkeypatch
) -> None:
    scores = _patch_select_scores(monkeypatch)
    config, _collect_cfg, collect_hash = _select_setup(tmp_path)
    _write_select_manifest(config, collect_hash)
    manifest_bytes = (config.output_dir / "manifest.json").read_bytes()

    assert run_select(config) == config.output_dir
    assert scores == []
    assert (config.output_dir / "manifest.json").read_bytes() == manifest_bytes


def test_select_rescores_when_n_or_collect_parent_changes(
    tmp_path: Path, monkeypatch
) -> None:
    scores = _patch_select_scores(monkeypatch)
    config, collect_cfg, collect_hash = _select_setup(tmp_path)
    _write_select_manifest(config, collect_hash)

    wider = config.model_copy(update={"n": 2})
    run_select(wider)
    assert scores == [HP]

    rewritten = collect_cfg.model_copy(update={"seed": collect_cfg.seed + 1})
    _write_collect_manifest(rewritten)
    run_select(wider)
    assert scores == [HP, HP]


def _tokenizer():
    class _Tok:
        def batch_decode(self, ids, skip_special_tokens=True):
            return [str(int(token_id)) for token_id in ids]

        def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
            return text.split()

    return _Tok()


def _patch_explainer(monkeypatch) -> list[list[dict[str, str]]]:
    generated: list[list[dict[str, str]]] = []

    class _Client:
        def __init__(self, model: str, **kwargs: object) -> None:
            self.model = model

        async def generate(self, messages, **kwargs):
            generated.append(messages)
            return ChatResponse(text="explained", request_id="req")

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(explain, "OpenRouterChatClient", _Client)
    monkeypatch.setattr(explain, "load_tokenizer", lambda manifest: _tokenizer())
    return generated


def _write_explain_collect(collect_dir: Path) -> str:
    """Write token, latent, and router shards for two features and one expert."""
    n_docs, seq_len = 8, 4
    n_tokens = n_docs * seq_len
    tokens = np.arange(n_tokens, dtype=np.int32).reshape(n_docs, seq_len)
    core.write_safetensors(
        collect_dir / "tokens" / "shard-00000.safetensors",
        {
            core.TOKENS: tokens,
            core.TOKENS_DOC_IDS: np.arange(n_docs, dtype=np.int64),
            core.TOKENS_SPLIT: np.ones(n_docs, dtype=bool),
        },
    )
    # Feature 9 is shifted one token so its highlighted window differs from 7.
    fires = {7: [0, 4, 8, 12, 16], 9: [1, 5, 9, 13, 17]}
    offsets = [0]
    feature_ids: list[int] = []
    values: list[float] = []
    for token in range(n_tokens):
        for feature_id, positions in fires.items():
            if token in positions:
                feature_ids.append(feature_id)
                values.append(float(feature_id))
        offsets.append(len(feature_ids))
    core.write_csr_shard(
        collect_dir / "latents" / core.hookpoint_slug(HP) / "shard-00000.safetensors",
        np.asarray(offsets, dtype=np.int64),
        np.asarray(feature_ids, dtype=np.int32),
        np.asarray(values, dtype=np.float16),
    )
    routed = {0, 1, 4, 5, 8, 9}
    topk_ids = np.ones((n_tokens, 1), dtype=np.int32)
    for token in routed:
        topk_ids[token, 0] = 0
    core.write_safetensors(
        collect_dir / "router" / "layers_0" / "shard-00000.safetensors",
        {
            core.ROUTER_TOPK_IDS: topk_ids,
            core.ROUTER_TOPK_VALS: np.ones((n_tokens, 1), dtype=np.float16),
        },
    )
    config = _collect_config(collect_dir, n_tokens=n_tokens, seq_len=seq_len)
    # shard_tokens must be a multiple of seq_len; the manifest is not executed.
    config = config.model_copy(update={"shard_tokens": n_tokens})
    return _write_collect_manifest(config).config_hash


def _write_feature_selection(
    selection_dir: Path,
    collect_dir: Path,
    collect_hash: str,
    feature_ids: list[int],
) -> None:
    n = len(feature_ids)
    rows = [
        {
            "method": "routing_margin",
            "hookpoint": HP,
            "expert_id": 0,
            "n": n,
            "rank": rank,
            "feature_id": feature_id,
            "score": float(n - rank),
        }
        for rank, feature_id in enumerate(feature_ids)
    ]
    core.write_parquet(
        selection_dir / "selections" / "routing_margin" / f"n{n}.parquet",
        rows,
    )
    config = core.SelectConfig(
        method="routing_margin",
        input_dir=collect_dir,
        output_dir=selection_dir,
        hookpoints=[HP],
        n=n,
    )
    core.write_manifest(
        selection_dir,
        core.build_manifest(
            stage="select",
            run_name="select",
            config=config,
            parents={"collect": collect_hash},
        ),
    )


def _explain_config(
    collect_dir: Path,
    selection_dir: Path,
    output_dir: Path,
    *,
    method: str,
    train_positives: int,
    train_negatives: int,
    explainer_model: str = "explainer-a",
) -> core.ExplainConfig:
    return core.ExplainConfig(
        method=method,
        input_dir=collect_dir,
        selection_dir=selection_dir,
        output_dir=output_dir,
        hookpoints=[HP],
        explainer_model=explainer_model,
        train_positives=train_positives,
        train_negatives=train_negatives,
        ctx_len=4,
        max_concurrent=1,
    )


def _explanation_rows(output_dir: Path) -> list[dict]:
    path = output_dir / explain.EXPLANATIONS_FILENAME
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_explain_skips_cached_feature_and_generates_the_new_one(
    tmp_path: Path, monkeypatch
) -> None:
    generated = _patch_explainer(monkeypatch)
    collect_dir = tmp_path / "collect"
    selection_dir = tmp_path / "select"
    output_dir = tmp_path / "explain"
    collect_hash = _write_explain_collect(collect_dir)
    _write_feature_selection(selection_dir, collect_dir, collect_hash, [7])
    config = _explain_config(
        collect_dir,
        selection_dir,
        output_dir,
        method="feature_autointerp",
        train_positives=2,
        train_negatives=0,
    )

    explain.run_explain(config)
    assert len(generated) == 1
    first = _explanation_rows(output_dir)
    assert [row["target_id"] for row in first] == [7]
    cached_hash = first[0]["prompt_hash"]
    assert cached_hash == explain.prompt_hash(config.explainer_model, generated[0])

    # A larger selection adds feature 9. Feature 7's own top windows are unchanged,
    # so its prompt_hash still hits and only the new feature is sent.
    _write_feature_selection(selection_dir, collect_dir, collect_hash, [7, 9])
    explain.run_explain(config)

    assert len(generated) == 2
    assert generated[1] != generated[0]
    rows = _explanation_rows(output_dir)
    assert [row["target_id"] for row in rows if row["target_id"] == 7] == [7]
    assert rows[0]["prompt_hash"] == cached_hash
    assert [row["target_id"] for row in rows if row["target_id"] == 9] == [9]


def test_explain_resends_when_model_or_prompt_changes(
    tmp_path: Path, monkeypatch
) -> None:
    generated = _patch_explainer(monkeypatch)
    collect_dir = tmp_path / "collect"
    selection_dir = tmp_path / "select"
    output_dir = tmp_path / "explain"
    collect_hash = _write_explain_collect(collect_dir)
    _write_feature_selection(selection_dir, collect_dir, collect_hash, [7])
    config = _explain_config(
        collect_dir,
        selection_dir,
        output_dir,
        method="feature_autointerp",
        train_positives=2,
        train_negatives=0,
    )
    explain.run_explain(config)
    original_hash = _explanation_rows(output_dir)[0]["prompt_hash"]

    other_model = config.model_copy(update={"explainer_model": "explainer-b"})
    explain.run_explain(other_model)
    assert len(generated) == 2
    after_model = [
        row for row in _explanation_rows(output_dir) if row["target_id"] == 7
    ]
    assert len(after_model) == 2
    assert after_model[-1]["explainer_model"] == "explainer-b"
    assert after_model[-1]["prompt_hash"] != original_hash

    shorter_prompt = other_model.model_copy(update={"train_positives": 1})
    explain.run_explain(shorter_prompt)
    assert len(generated) == 3
    assert generated[2] != generated[1]
    after_prompt = [
        row for row in _explanation_rows(output_dir) if row["target_id"] == 7
    ]
    assert len(after_prompt) == 3
    assert after_prompt[-1]["prompt_hash"] != after_model[-1]["prompt_hash"]


def test_explain_resume_is_per_output_dir(tmp_path: Path, monkeypatch) -> None:
    generated = _patch_explainer(monkeypatch)
    collect_dir = tmp_path / "collect"
    selection_dir = tmp_path / "select"
    collect_hash = _write_explain_collect(collect_dir)
    _write_feature_selection(selection_dir, collect_dir, collect_hash, [7])
    config = _explain_config(
        collect_dir,
        selection_dir,
        tmp_path / "explain-a",
        method="feature_autointerp",
        train_positives=2,
        train_negatives=0,
    )
    explain.run_explain(config)
    saved = _explanation_rows(config.output_dir)

    other = config.model_copy(update={"output_dir": tmp_path / "explain-b"})
    explain.run_explain(other)

    assert len(generated) == 2
    assert _explanation_rows(config.output_dir) == saved
    assert [row["target_id"] for row in _explanation_rows(other.output_dir)] == [7]


def test_explain_routerinterp_resends_expert_when_selection_grows(
    tmp_path: Path, monkeypatch
) -> None:
    # The expert prompt is the selected feature set, so a larger n changes it.
    generated = _patch_explainer(monkeypatch)
    collect_dir = tmp_path / "collect"
    selection_dir = tmp_path / "select"
    output_dir = tmp_path / "explain"
    collect_hash = _write_explain_collect(collect_dir)
    _write_feature_selection(selection_dir, collect_dir, collect_hash, [7])
    config = _explain_config(
        collect_dir,
        selection_dir,
        output_dir,
        method="routerinterp",
        train_positives=3,
        train_negatives=2,
    )
    explain.run_explain(config)
    assert len(generated) == 1
    first_hash = _explanation_rows(output_dir)[0]["prompt_hash"]
    assert _explanation_rows(output_dir)[0]["target_id"] == 0

    _write_feature_selection(selection_dir, collect_dir, collect_hash, [7, 9])
    explain.run_explain(config)

    assert len(generated) == 2
    expert_rows = _explanation_rows(output_dir)
    assert [row["target_id"] for row in expert_rows] == [0, 0]
    assert expert_rows[-1]["prompt_hash"] != first_hash


def _score_record(target_id: int, *, precision: float = 0.25) -> dict:
    score = {
        "method": "feature_autointerp",
        "hookpoint": HP,
        "target_kind": "feature",
        "target_id": target_id,
        "scorer": "llm0-4",
        "threshold": 2,
        "precision": precision,
        "recall": 0.5,
        "f1": 0.33,
        "n_pos": 1,
        "n_neg": 1,
        "scorer_model": "scorer-a",
        "request_ids": [],
    }
    prediction = {
        "method": "feature_autointerp",
        "hookpoint": HP,
        "target_kind": "feature",
        "target_id": target_id,
        "scorer": "llm0-4",
        "window_id": 40 + target_id,
        "ground_truth": True,
        "prediction": True,
        "raw_score": 3,
    }
    return {"scores": [score], "predictions": [prediction]}


def _write_evaluate_inputs(
    tmp_path: Path, texts: dict[int, str]
) -> tuple[core.EvaluateConfig, Path]:
    collect_dir = tmp_path / "collect"
    explain_dir = tmp_path / "explain"
    collect_cfg = _collect_config(collect_dir, n_tokens=8, seq_len=4)
    _write_collect_manifest(collect_cfg.model_copy(update={"shard_tokens": 8}))
    explain_cfg = core.ExplainConfig(
        method="feature_autointerp",
        input_dir=collect_dir,
        selection_dir=tmp_path / "select",
        output_dir=explain_dir,
        hookpoints=[HP],
        explainer_model="explainer-a",
        ctx_len=4,
    )
    core.write_manifest(
        explain_dir,
        core.build_manifest(stage="explain", run_name="explain", config=explain_cfg),
    )
    core.append_jsonl(
        explain_dir / explain.EXPLANATIONS_FILENAME,
        [
            {
                "method": "feature_autointerp",
                "hookpoint": HP,
                "target_kind": "feature",
                "target_id": target_id,
                "text": text,
                "prompt_hash": f"hash-{target_id}",
            }
            for target_id, text in texts.items()
        ],
    )
    output_dir = tmp_path / "eval"
    config = core.EvaluateConfig(
        explanations_dir=explain_dir,
        output_dir=output_dir,
        scorer_model="scorer-b",
        threshold=4,
        scoring_seed=1,
    )
    return config, output_dir


def _patch_evaluate(monkeypatch) -> dict:
    state = {"corpus": 0, "clients": 0, "scored": []}

    def _corpus(input_dir, manifest=None):
        state["corpus"] += 1
        return Corpus(
            tokens=np.zeros((2, 4), dtype=np.int32),
            doc_ids=np.arange(2),
            is_train=np.array([True, False]),
        )

    def _fires(input_dir, pending, *, expected_tokens):
        return {
            key.target: SparseActivations(
                np.zeros(1, dtype=np.int64),
                np.ones(1, dtype=np.float32),
            )
            for key in pending
        }

    def _windows(*args, **kwargs):
        return [
            EvalWindow(
                window_id=1,
                tokens=[1],
                str_tokens=["a"],
                activations=[1.0],
                ground_truth=True,
                highlight={0},
            ),
            EvalWindow(
                window_id=2,
                tokens=[2],
                str_tokens=["b"],
                activations=[0.0],
                ground_truth=False,
                highlight=set(),
            ),
        ]

    async def _score(client, explanation, windows, **kwargs):
        state["scored"].append(explanation)
        n = len(windows)
        return evaluate.WindowGrades([True] * n, [3] * n, ["req"])

    class _Client:
        def __init__(self, model: str, **kwargs: object) -> None:
            state["clients"] += 1

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(evaluate, "load_corpus", _corpus)
    monkeypatch.setattr(evaluate, "load_tokenizer", lambda manifest: object())
    monkeypatch.setattr(evaluate, "_collect_fires", _fires)
    monkeypatch.setattr(evaluate, "build_eval_windows", _windows)
    monkeypatch.setattr(evaluate, "score_windows_llm", _score)
    monkeypatch.setattr(evaluate, "OpenRouterChatClient", _Client)
    return state


def _feature_scores(output_dir: Path) -> list[dict]:
    table = core.read_parquet(output_dir / evaluate.SCORES_FILENAME).to_pylist()
    return [row for row in table if row["target_kind"] == "feature"]


def test_evaluate_finished_targets_skip_scoring_and_corpus_load(
    tmp_path: Path, monkeypatch
) -> None:
    # The checkpoint key is the target identity, not the scoring config.
    # A new threshold or scorer model in the same directory still skips.
    state = _patch_evaluate(monkeypatch)
    config, output_dir = _write_evaluate_inputs(
        tmp_path, {7: "keep-seven", 9: "keep-nine"}
    )
    core.append_jsonl(
        output_dir / evaluate.EVAL_PROGRESS_FILENAME,
        [_score_record(7, precision=0.25), _score_record(9, precision=0.75)],
    )
    core.write_parquet(
        output_dir / evaluate.SCORES_FILENAME,
        [{"precision": 999.0, "target_id": -99}],
    )

    evaluate.run_evaluate(config)

    assert state["scored"] == []
    assert state["clients"] == 0
    assert state["corpus"] == 0
    scores = _feature_scores(output_dir)
    by_id = {int(row["target_id"]): row for row in scores}
    assert set(by_id) == {7, 9}
    assert by_id[7]["threshold"] == 2
    assert by_id[7]["scorer_model"] == "scorer-a"
    assert by_id[7]["precision"] == 0.25
    assert all(row["precision"] != 999.0 for row in scores)
    predictions = core.read_parquet(
        output_dir / evaluate.PREDICTIONS_FILENAME
    ).to_pylist()
    assert {int(row["window_id"]) for row in predictions} == {47, 49}


def test_evaluate_scores_only_targets_missing_from_the_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    state = _patch_evaluate(monkeypatch)
    config, output_dir = _write_evaluate_inputs(
        tmp_path, {7: "keep-seven", 9: "score-nine"}
    )
    core.append_jsonl(
        output_dir / evaluate.EVAL_PROGRESS_FILENAME,
        [_score_record(7)],
    )

    evaluate.run_evaluate(config)

    assert state["scored"] == ["score-nine"]
    assert state["clients"] == 1
    assert state["corpus"] == 1
    by_id = {int(row["target_id"]): row for row in _feature_scores(output_dir)}
    assert by_id[7]["threshold"] == 2
    assert by_id[7]["scorer_model"] == "scorer-a"
    assert by_id[9]["threshold"] == 4
    assert by_id[9]["scorer_model"] == "scorer-b"


def test_evaluate_checkpoint_is_per_output_dir(tmp_path: Path, monkeypatch) -> None:
    state = _patch_evaluate(monkeypatch)
    config, output_dir = _write_evaluate_inputs(tmp_path, {7: "score-seven"})
    core.append_jsonl(
        output_dir / evaluate.EVAL_PROGRESS_FILENAME,
        [_score_record(7)],
    )
    progress = (output_dir / evaluate.EVAL_PROGRESS_FILENAME).read_bytes()
    other = config.model_copy(update={"output_dir": tmp_path / "eval-b"})

    evaluate.run_evaluate(other)

    assert state["scored"] == ["score-seven"]
    assert progress == (output_dir / evaluate.EVAL_PROGRESS_FILENAME).read_bytes()
    assert [int(row["target_id"]) for row in _feature_scores(other.output_dir)] == [7]


def test_evaluate_parquet_seeds_checkpoint_and_drops_macros(tmp_path: Path) -> None:
    score = {
        "method": "routerinterp",
        "hookpoint": "layers.19",
        "target_kind": "expert",
        "target_id": 3,
        "scorer": "llm0-4",
        "threshold": 2,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "n_pos": 1,
        "n_neg": 1,
        "scorer_model": "m",
        "request_ids": [],
    }
    macro = {**score, "target_kind": "all", "target_id": -1}
    core.write_parquet(tmp_path / evaluate.SCORES_FILENAME, [score, macro])
    core.write_parquet(
        tmp_path / evaluate.PREDICTIONS_FILENAME,
        [
            {
                "method": "routerinterp",
                "hookpoint": "layers.19",
                "target_kind": "expert",
                "target_id": 3,
                "scorer": "llm0-4",
                "window_id": 7,
                "ground_truth": True,
                "prediction": True,
                "raw_score": 3,
            }
        ],
    )

    evaluate._seed_progress_from_parquet(tmp_path)
    loaded = evaluate.load_eval_progress(tmp_path / evaluate.EVAL_PROGRESS_FILENAME)

    key = ExplanationKey("routerinterp", "layers.19", "expert", 3)
    assert list(loaded) == [key]
    assert loaded[key]["predictions"][0]["window_id"] == 7
    evaluate._seed_progress_from_parquet(tmp_path)
    assert (
        evaluate.load_eval_progress(tmp_path / evaluate.EVAL_PROGRESS_FILENAME)
        == loaded
    )


def _probe_setup(tmp_path: Path) -> tuple[Path, core.ProbeConfig, core.CollectConfig]:
    collect_dir = tmp_path / "collect"
    collect_cfg = _collect_config(collect_dir, n_tokens=8, seq_len=4)
    collect_cfg = collect_cfg.model_copy(update={"shard_tokens": 8})
    _write_collect_manifest(collect_cfg)
    probe_cfg = core.ProbeConfig(
        input_dir=collect_dir,
        output_dir=tmp_path / "probes",
        hookpoints=[HP],
        bases=["sae"],
        n_tokens=8,
        seed=0,
        min_tokens_per_expert=1,
    )
    return collect_dir, probe_cfg, collect_cfg


def _patch_probe_fit(monkeypatch) -> list[str]:
    fits: list[str] = []

    def _corpus(input_dir, manifest=None):
        return Corpus(
            tokens=np.zeros((4, 2), dtype=np.int32),
            doc_ids=np.arange(4),
            is_train=np.array([True, True, False, False]),
        )

    def _labels(input_dir, layer, sample):
        n = int(sample.selected.shape[0])
        labels = np.zeros((n, 1), dtype=np.bool_)
        labels[: n // 2, 0] = True
        return labels, 1, 1

    def _latents(input_dir, hookpoint, sample):
        n = int(sample.selected.shape[0])
        return np.ones((n, 1), dtype=np.float32)

    def _fit(**kwargs):
        fits.append(kwargs["hookpoint"])
        return [
            {
                "basis": kwargs["basis"],
                "hookpoint": kwargs["hookpoint"],
                "expert_id": 0,
                "metric": "f1",
                "value": 1.0,
                "split": "test",
                "n_train": 1,
                "n_test": 1,
            }
        ]

    import routerinterp.probes as probes

    monkeypatch.setattr(probes, "load_corpus", _corpus)
    monkeypatch.setattr(probes, "_load_labels", _labels)
    monkeypatch.setattr(probes, "_load_latents", _latents)
    monkeypatch.setattr(probes, "_run_linear_basis", _fit)
    return fits


def test_probes_skip_when_manifest_and_collect_parent_match(
    tmp_path: Path, monkeypatch
) -> None:
    fits = _patch_probe_fit(monkeypatch)
    _collect_dir, probe_cfg, collect_cfg = _probe_setup(tmp_path)
    collect_hash = core.read_manifest(probe_cfg.input_dir).config_hash
    core.write_manifest(
        probe_cfg.output_dir,
        core.build_manifest(
            stage="probe",
            run_name="probe",
            config=probe_cfg,
            parents={"collect": collect_hash},
        ),
    )
    manifest_bytes = (probe_cfg.output_dir / "manifest.json").read_bytes()

    assert run_probes(probe_cfg) == probe_cfg.output_dir
    assert fits == []
    assert (probe_cfg.output_dir / "manifest.json").read_bytes() == manifest_bytes
    del collect_cfg


def test_probes_refit_when_config_or_collect_parent_changes(
    tmp_path: Path, monkeypatch
) -> None:
    fits = _patch_probe_fit(monkeypatch)
    _collect_dir, probe_cfg, collect_cfg = _probe_setup(tmp_path)
    collect_hash = core.read_manifest(probe_cfg.input_dir).config_hash
    core.write_manifest(
        probe_cfg.output_dir,
        core.build_manifest(
            stage="probe",
            run_name="probe",
            config=probe_cfg,
            parents={"collect": collect_hash},
        ),
    )

    changed = probe_cfg.model_copy(update={"logreg_c": 0.1})
    run_probes(changed)
    assert fits == [HP]

    _write_collect_manifest(collect_cfg.model_copy(update={"seed": 1}))
    run_probes(changed)
    assert fits == [HP, HP]


def _patch_embedder(monkeypatch) -> list[str]:
    names: list[str] = []

    class _Embedder:
        def __init__(self, name: str) -> None:
            names.append(name)

        def encode(self, texts, **kwargs):
            n = len(list(texts))
            vectors = np.zeros((n, max(n, 1)), dtype=np.float32)
            for i in range(n):
                vectors[i, i] = 1.0
            return vectors

    monkeypatch.setattr(feature_clustering, "SentenceTransformer", _Embedder)
    return names


def _clustering_inputs(tmp_path: Path, explainer_model: str) -> tuple[Path, Path, Path]:
    explain_dir = tmp_path / "explain"
    explanations = explain_dir / "explanations.jsonl"
    selection = tmp_path / "selection.parquet"
    metrics = tmp_path / "figures" / "clustering.parquet"
    core.append_jsonl(
        explanations,
        [
            {
                "method": "feature_autointerp",
                "hookpoint": HP,
                "target_kind": "feature",
                "target_id": feature_id,
                "text": f"feature {feature_id} is about topic {feature_id}",
            }
            for feature_id in range(3)
        ],
    )
    core.write_parquet(
        selection,
        [
            {"hookpoint": HP, "expert_id": 0, "feature_id": feature_id}
            for feature_id in range(3)
        ],
    )
    parent = core.ExplainConfig(
        method="feature_autointerp",
        input_dir=tmp_path / "collect",
        selection_dir=tmp_path / "select",
        output_dir=explain_dir,
        hookpoints=[HP],
        explainer_model=explainer_model,
        ctx_len=4,
    )
    core.write_manifest(
        explain_dir,
        core.build_manifest(stage="explain", run_name="explain", config=parent),
    )
    return explanations, selection, metrics


def test_cluster_skips_when_manifest_matches(tmp_path: Path, monkeypatch) -> None:
    embedded = _patch_embedder(monkeypatch)
    explanations, selection, metrics = _clustering_inputs(tmp_path, "explainer-a")
    kwargs = {
        "n": 2,
        "k_max": 3,
        "embedding_model": "embed-a",
    }
    feature_clustering.cluster(explanations, selection, metrics, **kwargs)
    assert embedded == ["embed-a"]
    blob = metrics.read_bytes()

    feature_clustering.cluster(explanations, selection, metrics, **kwargs)
    assert embedded == ["embed-a"]
    assert metrics.read_bytes() == blob


def test_cluster_reembeds_when_n_model_or_explain_parent_changes(
    tmp_path: Path, monkeypatch
) -> None:
    embedded = _patch_embedder(monkeypatch)
    explanations, selection, metrics = _clustering_inputs(tmp_path, "explainer-a")
    feature_clustering.cluster(
        explanations, selection, metrics, n=2, k_max=3, embedding_model="embed-a"
    )
    assert embedded == ["embed-a"]

    feature_clustering.cluster(
        explanations, selection, metrics, n=4, k_max=3, embedding_model="embed-a"
    )
    assert embedded == ["embed-a", "embed-a"]

    feature_clustering.cluster(
        explanations, selection, metrics, n=4, k_max=3, embedding_model="embed-b"
    )
    assert embedded == ["embed-a", "embed-a", "embed-b"]

    parent = core.ExplainConfig(
        method="feature_autointerp",
        input_dir=tmp_path / "collect",
        selection_dir=tmp_path / "select",
        output_dir=explanations.parent,
        hookpoints=[HP],
        explainer_model="explainer-b",
        ctx_len=4,
    )
    core.write_manifest(
        explanations.parent,
        core.build_manifest(stage="explain", run_name="explain", config=parent),
    )
    feature_clustering.cluster(
        explanations, selection, metrics, n=4, k_max=3, embedding_model="embed-b"
    )
    assert embedded == ["embed-a", "embed-a", "embed-b", "embed-b"]


def _sweep_selection(path: Path) -> None:
    core.write_parquet(
        path,
        [
            {
                "hookpoint": HP,
                "method": "routing_margin",
                "n": 1,
                "expert_id": 0,
                "feature_id": 0,
                "rank": 0,
                "score": 1.0,
            }
        ],
    )


def _patch_sweep_probe(monkeypatch) -> list[str]:
    fits: list[str] = []

    def _corpus(input_dir, manifest=None):
        return Corpus(
            tokens=np.zeros((4, 2), dtype=np.int32),
            doc_ids=np.arange(4),
            is_train=np.array([True, True, False, False]),
        )

    def _labels(input_dir, layer, sample):
        n = int(sample.selected.shape[0])
        labels = np.zeros((n, 1), dtype=np.bool_)
        labels[: n // 2, 0] = True
        return labels, 1, 1

    def _latents(input_dir, hookpoint, sample):
        n = int(sample.selected.shape[0])
        return sp.csr_matrix(np.ones((n, 1), dtype=np.float32))

    def _fit(**kwargs):
        fits.append(str(kwargs["hookpoint"]))
        return [{"metric": "f1", "value": 0.5}]

    monkeypatch.setattr(feature_selection_sweep, "load_corpus", _corpus)
    monkeypatch.setattr(feature_selection_sweep, "_load_labels", _labels)
    monkeypatch.setattr(feature_selection_sweep, "_load_latents", _latents)
    monkeypatch.setattr(feature_selection_sweep, "_run_linear_basis", _fit)
    return fits


def _sweep_config(tmp_path: Path) -> tuple[core.ProbeConfig, Path]:
    collect_dir = tmp_path / "collect"
    collect_cfg = _collect_config(collect_dir, n_tokens=8, seq_len=4).model_copy(
        update={"shard_tokens": 8}
    )
    _write_collect_manifest(collect_cfg)
    config = core.ProbeConfig(
        input_dir=collect_dir,
        output_dir=tmp_path / "unused",
        hookpoints=[HP],
        n_tokens=8,
        seed=0,
    )
    return config, tmp_path / "figures" / "sweep.parquet"


def test_feature_selection_sweep_skips_matching_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    fits = _patch_sweep_probe(monkeypatch)
    config, metrics = _sweep_config(tmp_path)
    selection = tmp_path / "n1.parquet"
    _sweep_selection(selection)

    probe_selections(config, [selection], metrics)
    assert fits == [HP]
    blob = metrics.read_bytes()

    probe_selections(config, [selection], metrics)
    assert fits == [HP]
    assert metrics.read_bytes() == blob


def test_feature_selection_sweep_refits_when_selection_paths_change(
    tmp_path: Path, monkeypatch
) -> None:
    fits = _patch_sweep_probe(monkeypatch)
    config, metrics = _sweep_config(tmp_path)
    first = tmp_path / "n1.parquet"
    second = tmp_path / "n1-other.parquet"
    _sweep_selection(first)
    _sweep_selection(second)

    probe_selections(config, [first], metrics)
    probe_selections(config, [second], metrics)
    assert fits == [HP, HP]
