"""Keep cluster .env from leaking into path resolution during tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_cluster_env(monkeypatch) -> None:
    monkeypatch.delenv("ARTIFACTS_ROOT", raising=False)
