"""Anthropic Claude provider implementation."""

from __future__ import annotations

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


@dataclass(slots=True, frozen=True)
class ClaudeRuntimeOptions(ProviderRuntimeOptions):
    """Runtime options specific to the Claude provider."""

    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    """Controls the amount of effort Claude puts into generating the response, which can affect quality and latency. Higher effort levels may produce better responses but take more time and compute resources. If not specified, the provider default will be used."""
    thinking_type: Literal["enabled", "disabled", "adaptive"] | None = None
    """When enabled, responses include thinking content blocks showing Claude's thinking process before the final answer. Requires a minimum budget of 1,024 tokens and counts towards your max_tokens limit."""
    thinking_budget: int | None = None
    """Determines how many tokens Claude can use for its internal reasoning process. Larger budgets can enable more thorough analysis for complex problems, improving response quality. Must be ≥1024 and less than max_tokens."""
    thinking_display: Literal["summarized", "ommitted", "full"] | None = None
    """When thinking is enabled (or adaptive). Controls how thinking content appears in the response. When set to summarized, thinking is returned normally. When set to omitted, thinking content is redacted but a signature is returned for multi-turn continuity. Defaults to summarized."""

    def to_payload(self):
        payload: dict[str, Any] = {}

        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        else:
            payload["max_tokens"] = (
                8192  # Set a high default max_tokens for Claude
            )

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
    """Client wrapper for Anthropic Claude APIs."""

    provider_type = Provider.CLAUDE

    def __init__(
        self,
        model: str,
        max_workers: int = 4,
        api_key: str | None = None,
        on_error: OnError = "raise",
    ) -> None:
        """Initialize the Claude client.

        Args:
            model: Claude model identifier.
            max_workers: Maximum number of concurrent worker threads for ``batch_generate``.
            api_key: Anthropic API key. If not provided, the client will attempt to read from the environment variable `ANTHROPIC_API_KEY`.
            on_error: Error behavior when generation fails.
        """
        from anthropic import Anthropic

        super().__init__(
            model=model, max_workers=max_workers, on_error=on_error
        )

        self._api_key = api_key
        self._client = Anthropic(api_key=self._api_key)

        self._running_batch_ids: set[str] = set()

    def _process_response(self, response: ClaudeMessage) -> Response:
        num_output_tokens = getattr(response.usage, "output_tokens", None)
        self._handle_stop_reason(
            stop_reason=response.stop_reason,
            num_output_tokens=num_output_tokens,
        )

        text_chunks: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) != "text":
                continue
            block_text = getattr(block, "text", None)
            if isinstance(block_text, str):
                text_chunks.append(block_text)
        content = "\n".join(text_chunks).strip()

        return Response(
            text=content,
            stop_reason=response.stop_reason,
            model=response.model,
            provider=self.provider_type,
            num_output_tokens=num_output_tokens,
            full_response=response,
        )

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        options: ClaudeRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> Response:
        """Generate a structured JSON response using Claude.

        Args:
            messages: List of message dictionaries.
            options: Provider-specific generation options.
            gen_kwargs: Additional provider-specific generation kwargs that are not covered by the standard options.
                Has precedence over ``options``.

        Returns:
            A Response object containing the generated response.

        Raises:
            ProviderError: If the provider call fails.
        """
        options = options or ClaudeRuntimeOptions()

        try:
            # Claude API requires separating the system prompt
            messages, system_instruction = _extract_system_instruction(
                messages
            )

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

                request_payload["output_config"]["format"] = {
                    "type": "json_schema",
                    "schema": options.json_schema,
                }

            request_payload.update(gen_kwargs or {})
            response = self._client.messages.create(**request_payload)
        except Exception as exc:
            return self._handle_error(exc, context="Claude request failed")
        else:
            try:
                return self._process_response(response=response)
            except Exception as exc:
                return self._handle_error(
                    exc, context="Claude response processing failed"
                )

    def batch_generate(
        self,
        *,
        messages: list[list[dict[str, str]]],
        options: ClaudeRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> list[Response]:
        """Generate responses for a batch of inputs concurrently.

        The Anthropic API has no native synchronous batch endpoint, so requests
        are dispatched in parallel using a thread pool.

        Args:
            messages: List of message lists, one per request.
            options: Provider-specific generation options.
            gen_kwargs: Additional provider-specific generation kwargs that are not covered by the standard options.
                Has precedence over ``options``.

        Returns:
            A list of Response objects in the same order as the input.

        Raises:
            ProviderError: If any individual request fails.
        """
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self.generate,
                    messages=msgs,
                    options=options,
                    gen_kwargs=gen_kwargs,
                )
                for msgs in messages
            ]

        responses: list[Response] = []
        for idx, future in enumerate(futures):
            try:
                responses.append(future.result())
            except Exception as exc:
                responses.append(
                    self._handle_error(
                        exc,
                        context=f"Claude batch request failed at index {idx}",
                    )
                )
        return responses

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

    def destroy(self) -> None:
        """Clean up any resources used by the client."""

        if self._client is not None and self._running_batch_ids:
            for batch_id in self._running_batch_ids:
                self._client.messages.batches.cancel(batch_id)


def _extract_system_instruction(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], str]:
    """Convert OpenAI-style messages to Claude input text and instruction.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.
    Returns:
        A tuple of (list[dict[str, str]], system_instruction) to be used for Claude generation.
    """
    system_instruction = ""
    for msg_idx, message in enumerate(messages):
        role = message["role"]
        content = message["content"]

        if role == "system":
            if system_instruction:
                raise ProviderError(
                    "For Claude, only a single system message is supported."
                )

            if msg_idx != 0:
                raise ValueError(
                    "Make sure that the system message is the first message in the list."
                )
            system_instruction = content
        elif role not in {"user", "assistant"}:
            raise ValueError(
                f"Unsupported message role {role!r} for Claude client. Only 'system', 'assistant', and 'user' roles are supported."
            )

    messages = messages[1:] if system_instruction else messages

    return messages, system_instruction


__all__ = ["ClaudeClient", "ClaudeRuntimeOptions"]
