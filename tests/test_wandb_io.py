"""W&B logging hook.
No network: wandb.init is mocked.
"""

from __future__ import annotations

from routerinterp import core, wandb_io


def _forbid_init(**kwargs: object) -> None:
    raise AssertionError(kwargs)


def test_wandb_log_noop_without_project(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.setattr(wandb_io.wandb, "init", _forbid_init)
    assert wandb_io.wandb_log(tmp_path) is None


def test_wandb_log_uses_project_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    monkeypatch.setenv("WANDB_ENTITY", "ent")

    def fake_upload(run_dir, project, entity, **kwargs):
        assert run_dir == tmp_path
        assert project == "proj"
        assert entity == "ent"
        assert kwargs.get("extra_tags") is None
        return "https://wandb.ai/ent/proj/runs/new"

    monkeypatch.setattr(wandb_io, "_upload", fake_upload)
    assert wandb_io.wandb_log(tmp_path) == "https://wandb.ai/ent/proj/runs/new"


def test_table_dataframe_stringifies_ragged_request_ids() -> None:
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame(
        {
            "f1": [0.0, 1.0],
            "request_ids": [
                np.array(["a", "b"], dtype=object),
                np.array(["c"], dtype=object),
            ],
        }
    )
    out = wandb_io._table_dataframe(frame)
    assert out["f1"].tolist() == [0.0, 1.0]
    assert out["request_ids"].tolist() == ['["a", "b"]', '["c"]']


def test_wandb_log_swallows_upload_errors(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WANDB_PROJECT", "proj")

    def boom(*_a, **_k):
        raise TypeError("ragged request_ids")

    monkeypatch.setattr(wandb_io, "_upload", boom)
    assert wandb_io.wandb_log(tmp_path) is None


def test_write_manifest_unset_project_does_not_init(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.setattr(wandb_io.wandb, "init", _forbid_init)
    config = core.CollectConfig(
        model=core.ModelConfig(name="toy/model"),
        dataset=core.DatasetConfig(name="toy/data", n_tokens=1, seq_len=1),
        output_dir=tmp_path,
    )
    manifest = core.build_manifest("collect", "collect", config)
    path = core.write_manifest(tmp_path, manifest)
    assert path.read_text().startswith("{")
    assert core.read_manifest(tmp_path).run_name == "collect"
