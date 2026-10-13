"""OpenRouter chat client used by the explain and evaluate stages."""

from __future__ import annotations

import os

from loguru import logger
from openrouter.components import (
    ChatAssistantMessage,
    ChatAssistantMessageTypedDict,
    ChatMessagesTypedDict,
    ChatResult,
    ChatSystemMessageTypedDict,
    ChatUserMessageTypedDict,
)
from openrouter.sdk import OpenRouter

from routerinterp.explain.schema import ChatResponse

EXPLAINER_TEMPERATURE = 1.0


def _chat_messages(messages: list[dict[str, str]]) -> list[ChatMessagesTypedDict]:
    """Build SDK message objects from ``{role, content}`` prompt dicts."""
    typed: list[ChatMessagesTypedDict] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            typed.append(ChatSystemMessageTypedDict(role=role, content=content))
        elif role == "user":
            typed.append(ChatUserMessageTypedDict(role=role, content=content))
        elif role == "assistant":
            typed.append(ChatAssistantMessageTypedDict(role=role, content=content))
        else:
            msg = f"Unsupported chat role {role!r}"
            raise ValueError(msg)
    return typed


def _assistant_text(message: ChatAssistantMessage) -> str:
    """Return the visible assistant text.
    Non-text payloads are stringified.
    """
    content = message.content
    if isinstance(content, str):
        return content
    if not content:
        return ""
    return str(content)


def _reasoning_sizes(message: ChatAssistantMessage) -> tuple[int, int]:
    """Return the reasoning character count and the number of reasoning-detail items."""
    reasoning = message.reasoning
    n_reasoning = len(reasoning) if isinstance(reasoning, str) else 0
    details = message.reasoning_details
    n_details = 0 if details is None else len(details)
    return n_reasoning, n_details


class OpenRouterChatClient:
    """Chat client for OpenRouter.
    The API key is ``OPENROUTER_API_KEY``.

    One client may be called concurrently.
    Timeouts and retries belong to the SDK.
    """

    _client: OpenRouter

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float = EXPLAINER_TEMPERATURE,
        timeout_ms: int = 100_000,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._client = OpenRouter(
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY", ""),
            timeout_ms=timeout_ms,
        )

    async def aclose(self) -> None:
        await self._client.__aexit__(None, None, None)

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
    ) -> ChatResponse:
        response: ChatResult = await self._client.chat.send_async(
            model=self.model,
            messages=_chat_messages(messages),
            temperature=temperature if temperature is not None else self.temperature,
            stream=False,
        )
        request_id = response.id or None
        if request_id:
            logger.info(f"OpenRouter request_id={request_id} model={self.model}")
        else:
            logger.warning(
                f"OpenRouter response has no request id (model={self.model})"
            )
        if not response.choices:
            msg = "OpenRouter returned no choices"
            raise ValueError(msg)
        message = response.choices[0].message
        text = _assistant_text(message)
        n_reasoning, n_details = _reasoning_sizes(message)
        if n_reasoning or n_details:
            logger.info(
                f"OpenRouter reasoning request_id={request_id or '-'} "
                f"chars={n_reasoning} details={n_details}"
            )
        if not text.strip():
            suffix = f" request_id={request_id}" if request_id else ""
            extra = " (reasoning-only)" if n_reasoning or n_details else ""
            msg = f"OpenRouter returned empty assistant content{suffix}{extra}"
            raise ValueError(msg)
        return ChatResponse(text=text, request_id=request_id)
