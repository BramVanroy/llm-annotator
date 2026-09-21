"""Anthropic Claude provider implementation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from llm_annotator.clients.base import (
    Client,
    OnError,
    Provider,
    ProviderRuntimeOptions,
    Response,
)
from llm_annotator.clients.exceptions import ProviderError
from llm_annotator.utils import add_schema_additional_properties_false


@dataclass(slots=True, frozen=True)
class ClaudeRuntimeOptions(ProviderRuntimeOptions):
    """Runtime options specific to the Claude provider."""

    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    """Controls the amount of effort Claude puts into generating the response, which can affect quality and latency. Higher effort levels may produce better responses but take more time and compute resources. If not specified, the provider default will be used."""
    thinking_type: Literal["enabled", "disabled", "adaptive"] | None = None
    """When enabled, responses include thinking content blocks showing Claude's thinking process before the final answer. Requires a minimum budget of 1,024 tokens and counts towards your max_completion_tokens limit."""
    thinking_budget: int | None = None
    """Determines how many tokens Claude can use for its internal reasoning process. Larger budgets can enable more thorough analysis for complex problems, improving response quality. Must be ≥1024 and less than max_completion_tokens."""
    thinking_display: Literal["summarized", "omitted", "full"] | None = None
    """When thinking is enabled (or adaptive). Controls how thinking content appears in the response. When set to summarized, thinking is returned normally. When set to omitted, thinking content is redacted but a signature is returned for multi-turn continuity. Defaults to summarized."""

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}

        if self.max_completion_tokens is not None:
            payload["max_tokens"] = self.max_completion_tokens

        if self.effort is not None:
            if "output_config" not in payload:
                payload["output_config"] = {}
            payload["output_config"]["effort"] = self.effort

        if self.thinking_type is not None:
            payload["thinking"] = {"type": self.thinking_type}

            if (
                self.thinking_type == "enabled"
                and self.thinking_budget is not None
            ):
                payload["thinking"]["budget_tokens"] = self.thinking_budget

            if (
                self.thinking_type in {"enabled", "adaptive"}
                and self.thinking_display is not None
            ):
                payload["thinking"]["display"] = self.thinking_display

        return payload


if TYPE_CHECKING:
    from anthropic.types.message import Message as ClaudeMessage


class ClaudeClient(Client[ClaudeRuntimeOptions]):
    """Client wrapper for Anthropic Claude APIs.

    The Messages API has no synchronous batch endpoint, so the inherited
    [`batch_generate`][llm_annotator.clients.base.Client.batch_generate] sends
    one request per sample over a thread pool of ``max_workers`` threads.
    """

    provider_type = Provider.CLAUDE

    def __init__(
        self,
        model: str,
        max_workers: int | None = 4,
        api_key: str | None = None,
        on_error: OnError = "warn",
    ) -> None:
        """Initialize the Claude client.

        Args:
            model: Claude model identifier.
            max_workers: Maximum number of concurrent worker threads for
                ``batch_generate``. ``None``, ``0`` and ``1`` send the requests
                of a batch one after another.
            api_key: Anthropic API key. If not provided, the client will attempt to read from the environment variable `ANTHROPIC_API_KEY`.
            on_error: Error behavior when generation fails.
        """
        from anthropic import Anthropic

        super().__init__(
            model=model, max_workers=max_workers, on_error=on_error
        )

        self._api_key = api_key
        self._client = Anthropic(api_key=self._api_key)

    def _process_response(self, response: ClaudeMessage) -> Response:
        num_output_tokens = getattr(response.usage, "output_tokens", None)

        text_chunks: list[str] = []
        thinking_chunks: list[str] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                block_text = getattr(block, "text", None)
                if isinstance(block_text, str):
                    text_chunks.append(block_text)
            elif block_type == "thinking":
                block_thinking = getattr(block, "thinking", None)
                if isinstance(block_thinking, str):
                    thinking_chunks.append(block_thinking)
        content = "\n".join(text_chunks).strip()
        thinking = "\n".join(thinking_chunks).strip()

        partial = Response(
            text=content,
            stop_reason=response.stop_reason,
            model=response.model,
            provider=self.provider_type,
            num_output_tokens=num_output_tokens,
            full_response=response,
            reasoning=thinking or None,
        )

        try:
            self._handle_stop_reason(
                stop_reason=response.stop_reason,
                num_output_tokens=num_output_tokens,
            )
        except Exception as exc:
            return self._handle_error(
                exc,
                context="Claude response stop reason",
                partial=partial,
            )

        return partial

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        options: ClaudeRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> Response:
        """Generate a response using Claude.

        Args:
            messages: List of message dictionaries.
            options: Provider-specific generation options.
            gen_kwargs: Additional provider-specific generation kwargs that are not covered by the standard options.
                Has precedence over ``options``.

        Returns:
            A Response object containing the generated response. A failed
            request is an error Response when ``on_error`` is ``"warn"`` or
            ``"ignore"``.

        Raises:
            ProviderError: If the request fails and ``on_error`` is
                ``"raise"``, or if ``messages`` holds more than one system
                message.
            ValueError: If a message has a role Claude does not take, or a
                system message is not the first message.
        """
        options = options or ClaudeRuntimeOptions()

        # The Messages API takes the system prompt as its own argument rather
        # than as a message.
        messages, system_instruction = _extract_system_instruction(messages)

        request_payload: dict[str, Any] = options.to_payload()
        request_payload.update(
            {
                "model": self.model,
                "messages": messages,
            }
        )

        if system_instruction:
            request_payload["system"] = system_instruction

        if options.json_schema is not None:
            if "output_config" not in request_payload:
                request_payload["output_config"] = {}

            schema = _sanitize_schema(
                add_schema_additional_properties_false(options.json_schema)
            )
            request_payload["output_config"]["format"] = {
                "type": "json_schema",
                "schema": schema,
            }

        request_payload.update(gen_kwargs or {})

        try:
            response = self._client.messages.create(**request_payload)
        except Exception as exc:
            return self._handle_error(exc, context="Claude request failed")

        try:
            return self._process_response(response=response)
        except Exception as exc:
            return self._handle_error(
                exc, context="Claude response processing failed"
            )

    def _handle_stop_reason(
        self, *, stop_reason: str | None, num_output_tokens: int | None
    ) -> None:
        """Raise a provider error when Claude ended for a non-success reason.

        Args:
            stop_reason: Claude stop reason.
            num_output_tokens: Number of output tokens generated.

        Raises:
            ProviderError: If the stop reason indicates an incomplete or blocked response.
        """
        token_suffix = (
            ""
            if num_output_tokens is None
            else f" (output tokens: {num_output_tokens:,})"
        )
        if stop_reason in {None, "end_turn", "stop_sequence"}:
            return

        if stop_reason == "max_tokens":
            raise ProviderError(
                f"Claude stopped because it hit the configured output token limit{token_suffix}."
            )
        if stop_reason == "tool_use":
            raise ProviderError(
                f"Claude attempted to emit a tool call instead of returning a regular response{token_suffix}."
                " Adjust the prompt or disable tool use for this request."
            )
        if stop_reason == "pause_turn":
            raise ProviderError(
                f"Claude paused the turn before completing the response{token_suffix}."
            )
        if stop_reason == "refusal":
            raise ProviderError(
                f"Claude refused to answer the request{token_suffix}."
            )
        if stop_reason == "model_context_window_exceeded":
            raise ProviderError(
                f"Claude exceeded the model context window before completing the response{token_suffix}."
            )
        raise ProviderError(
            f"Claude stopped for an unexpected reason {stop_reason!r}{token_suffix}."
        )


def _extract_system_instruction(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], str]:
    """Split a leading system message off an OpenAI-style message list.

    The Messages API rejects a ``system`` role inside ``messages``, so the
    system message is removed from the list whatever its content is. An empty
    system message therefore leaves an empty instruction, and
    [`ClaudeClient.generate`][llm_annotator.clients.claude_client.ClaudeClient.generate]
    then sends no ``system`` argument at all.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.

    Returns:
        The messages without the system message, and the system instruction
        (``""`` when there is none).

    Raises:
        ProviderError: If more than one system message is present.
        ValueError: If a system message is not first, or a role is one Claude
            does not take.

    Examples:
        >>> _extract_system_instruction(
        ...     [
        ...         {"role": "system", "content": ""},
        ...         {"role": "user", "content": "hi"},
        ...     ]
        ... )
        ([{'role': 'user', 'content': 'hi'}], '')
    """
    system_instruction = ""
    has_system = False
    remaining: list[dict[str, str]] = []

    for msg_idx, message in enumerate(messages):
        role = message["role"]

        if role == "system":
            if has_system:
                raise ProviderError(
                    "For Claude, only a single system message is supported."
                )
            if msg_idx != 0:
                raise ValueError(
                    "Make sure that the system message is the first message in the list."
                )
            has_system = True
            system_instruction = message["content"]
        elif role not in {"user", "assistant"}:
            raise ValueError(
                f"Unsupported message role {role!r} for Claude client. Only 'system', 'assistant', and 'user' roles are supported."
            )
        else:
            remaining.append(message)

    return remaining, system_instruction


def _sanitize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Claude does not support all JSON Schema features.
    - Claude does not support integer min/max values. Remove them.

    Args:
        schema: The original JSON schema.

    Returns:
        The sanitized JSON schema compatible with Claude.
    """
    sanitized = copy.deepcopy(schema)

    def walk(s: Any):
        if not isinstance(s, dict):
            return

        # Remove min/max for integers
        if s.get("type") == "integer":
            s.pop("minimum", None)
            s.pop("maximum", None)

        if s.get("type") == "object" and "properties" in s:
            for prop_schema in s["properties"].values():
                walk(prop_schema)

        if s.get("type") == "array" and "items" in s:
            items = s["items"]
            if isinstance(items, dict):
                walk(items)
            elif isinstance(items, list):
                for item_schema in items:
                    walk(item_schema)

    walk(sanitized)
    return sanitized


__all__ = ["ClaudeClient", "ClaudeRuntimeOptions"]
