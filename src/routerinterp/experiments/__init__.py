"""Jobs that read finished pipeline outputs and write one metrics parquet.

Three results live here. Feature clustering embeds feature explanations and counts how many clusters each expert's features span. The feature-selection sweep probes each top-n feature set and records F1. Pile domains record how strongly each expert prefers each Pile domain.

A missing input file raises ``MissingInputs``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from routerinterp.core import TIDY_COLUMNS, write_parquet


class MissingInputs(FileNotFoundError):
    """A figure job was started before its inputs were on disk."""


def require(paths: list[Path]) -> None:
    """Raise ``MissingInputs`` listing every absent path."""
    missing = [
        path
        for path in paths
        if not path.exists() or (path.is_dir() and not any(path.iterdir()))
    ]
    if not missing:
        return
    lines = ["Missing inputs:"]
    lines.extend(f"  {path}" for path in missing)
    raise MissingInputs("\n".join(lines))


def write_tidy(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write ``core.TIDY_COLUMNS`` rows.
    ``x`` is stored as text.
    """
    slim: list[dict[str, Any]] = []
    for row in rows:
        missing = [col for col in TIDY_COLUMNS if col not in row]
        if missing:
            msg = f"Tidy row missing {missing}"
            raise ValueError(msg)
        item = {col: row[col] for col in TIDY_COLUMNS}
        item["x"] = "" if item["x"] is None else str(item["x"])
        item["value"] = float(item["value"])
        slim.append(item)
    if not slim:
        msg = f"No tidy rows to write at {path}"
        raise ValueError(msg)
    return write_parquet(path, slim)
