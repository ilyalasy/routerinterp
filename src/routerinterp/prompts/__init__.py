"""Jinja templates for explanation and scoring prompts."""

from __future__ import annotations

import json
from functools import cache
from importlib.resources import files
from typing import Any

from jinja2 import Environment, StrictUndefined

_ENV = Environment(
    autoescape=False,
    undefined=StrictUndefined,
    keep_trailing_newline=True,
)


@cache
def _template(name: str):
    source = (files("routerinterp.prompts") / f"{name}.jinja").read_text(
        encoding="utf-8"
    )
    return _ENV.from_string(source)


def render_messages(name: str, **kwargs: Any) -> list[dict[str, str]]:
    """Render ``prompts/<name>.jinja`` to a list of ``{role, content}`` messages."""
    rendered = _template(name).render(**kwargs)
    messages = json.loads(rendered)
    if not isinstance(messages, list):
        msg = f"Prompt template {name!r} must render a JSON list of messages"
        raise TypeError(msg)
    return messages
