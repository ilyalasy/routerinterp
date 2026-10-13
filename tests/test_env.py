"""ARTIFACTS_ROOT path prefix.
No network.
"""

from __future__ import annotations

from pathlib import Path

from routerinterp.core import CollectConfig, resolve_run_path


def test_resolve_run_path_keeps_relative_without_root() -> None:
    assert resolve_run_path(Path("artifacts/olmoe/collect")) == Path(
        "artifacts/olmoe/collect"
    )


def test_resolve_run_path_prefixes_relative(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ARTIFACTS_ROOT", str(tmp_path))
    assert resolve_run_path(Path("artifacts/olmoe/collect")) == tmp_path / (
        "artifacts/olmoe/collect"
    )


def test_resolve_run_path_keeps_absolute(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ARTIFACTS_ROOT", str(tmp_path / "scratch"))
    absolute = tmp_path / "elsewhere"
    assert resolve_run_path(absolute) == absolute


def test_collect_config_uses_artifacts_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ARTIFACTS_ROOT", str(tmp_path))
    config = CollectConfig.model_validate(
        {
            "model": {"name": "toy/model"},
            "dataset": {"name": "toy/data", "n_tokens": 1, "seq_len": 1},
            "output_dir": "artifacts/olmoe/collect",
        }
    )
    assert config.output_dir == tmp_path / "artifacts/olmoe/collect"
