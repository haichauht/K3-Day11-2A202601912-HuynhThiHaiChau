"""Small OpenAI Responses API runtime used by the Lab 11 agents.

The starter repository was written around Google ADK callbacks.  This adapter
keeps those callback contracts so the existing guardrail plugins remain useful,
while all model inference is performed by OpenAI.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from types import SimpleNamespace
from uuid import uuid4

from google.genai import types
from openai import AsyncOpenAI

from core.config import setup_api_key


def _content_text(content: types.Content | None) -> str:
    """Extract text from a Google Content object used by starter callbacks."""
    if not content or not content.parts:
        return ""
    return "".join(
        part.text for part in content.parts if getattr(part, "text", None)
    )


@dataclass(frozen=True)
class OpenAIAgent:
    """Provider-neutral agent definition backed by OpenAI at runtime."""

    model: str
    name: str
    instruction: str


@dataclass
class OpenAISession:
    """Minimal session object compatible with existing lab call sites."""

    id: str = field(default_factory=lambda: uuid4().hex)
    previous_response_id: str | None = None


@dataclass
class OpenAIModelResponse:
    """Mutable response wrapper expected by ADK-style output callbacks."""

    content: types.Content


class OpenAIRunner:
    """Run an OpenAIAgent and apply existing input/output plugin callbacks."""

    def __init__(
        self,
        *,
        agent: OpenAIAgent,
        app_name: str,
        plugins: list | None = None,
    ):
        self.agent = agent
        self.app_name = app_name
        self.plugins = list(plugins or [])
        self._client: AsyncOpenAI | None = None
        self._sessions: dict[str, OpenAISession] = {}

    def _get_client(self) -> AsyncOpenAI:
        """Create the SDK client lazily so imports and smoke tests need no key."""
        if self._client is None:
            setup_api_key()
            self._client = AsyncOpenAI()
        return self._client

    def _get_session(self, session_id: str | None) -> OpenAISession:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        session = OpenAISession(id=session_id or uuid4().hex)
        self._sessions[session.id] = session
        return session

    @staticmethod
    async def _invoke(callback, **kwargs):
        result = callback(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def chat(
        self,
        user_message: str,
        *,
        session_id: str | None = None,
        user_id: str = "student",
    ) -> tuple[str, OpenAISession]:
        """Run input callbacks, call Responses API, then run output callbacks."""
        session = self._get_session(session_id)
        callback_context = SimpleNamespace(
            user_id=user_id,
            app_name=self.app_name,
            session_id=session.id,
        )
        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )

        for plugin in self.plugins:
            callback = getattr(plugin, "on_user_message_callback", None)
            if callback is None:
                continue
            replacement = await self._invoke(
                callback,
                invocation_context=callback_context,
                user_message=user_content,
            )
            if replacement is not None:
                return _content_text(replacement), session

        request = {
            "model": self.agent.model,
            "instructions": self.agent.instruction,
            "input": user_message,
        }
        # Continue a conversation only when the caller explicitly supplies a
        # session ID. Normal lab attacks remain isolated from one another.
        if session_id and session.previous_response_id:
            request["previous_response_id"] = session.previous_response_id

        response = await self._get_client().responses.create(**request)
        session.previous_response_id = response.id
        model_response = OpenAIModelResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=response.output_text or "")],
            )
        )

        for plugin in self.plugins:
            callback = getattr(plugin, "after_model_callback", None)
            if callback is None:
                continue
            replacement = await self._invoke(
                callback,
                callback_context=callback_context,
                llm_response=model_response,
            )
            if replacement is not None:
                model_response = replacement

        return _content_text(model_response.content), session
