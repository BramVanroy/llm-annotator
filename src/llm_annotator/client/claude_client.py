"""Anthropic Claude provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llm_annotator.client.base import (
    Client,
    Provider,
    ProviderRuntimeOptions,
    Response,
)
from llm_annotator.client.exceptions import ProviderError


if TYPE_CHECKING:
    from anthropic.types.message import Message as ClaudeMessage


class ClaudeClient(Client[ClaudeMessage]):
    """Client wrapper for Anthropic Claude APIs."""

    provider_type = Provider.CLAUDE

    def __init__(self, model: str, api_key: str | None = None) -> None:
        """Initialize the Claude client.

        Args:
            model: Claude model identifier.
            api_key: Anthropic API key. If not provided, the client will attempt to read from the environment variable `ANTHROPIC_API_KEY`.
        """
        from anthropic import Anthropic

        super().__init__(model=model)

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
        )

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        options: ProviderRuntimeOptions | None = None,
    ) -> Response:
        """Generate a structured JSON response using Claude.

        Args:
            messages: List of message dictionaries.
            options: Optional generation configuration.

        Returns:
            A Response object containing the generated response.

        Raises:
            ProviderError: If the provider call fails.
            ParsingError: If model output cannot be parsed as JSON.
        """
        options = options or ProviderRuntimeOptions()
        try:
            request_payload: dict[str, Any] = {
                "model": self.model,
                "max_tokens": options.max_tokens,
                "messages": messages,
            }

            response = self._client.messages.create(**request_payload)
        except Exception as exc:
            raise ProviderError(f"Claude request failed: {exc}") from exc
        else:
            return self._process_response(response=response)

    def batch_generate(
        self,
        *,
        messages: list[list[dict[str, str]]],
        options: ProviderRuntimeOptions | None = None,
    ) -> list[Response]:
        """Batch generation for Claude is not currently implemented.
        Info: https://platform.claude.com/docs/en/build-with-claude/batch-processing#how-the-message-batches-api-works
        Running batches can be cancelled: https://platform.claude.com/docs/en/build-with-claude/batch-processing#canceling-a-message-batch
        """
        raise NotImplementedError(
            "Batch generation is not yet implemented for the Claude client."
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

    def destroy(self) -> None:
        """Clean up any resources used by the client."""

        from anthropic import Anthropic

        if self._running_batch_ids:
            client = Anthropic(api_key=self._api_key)
            for batch_id in self._running_batch_ids:
                client.messages.batches.cancel(batch_id)
