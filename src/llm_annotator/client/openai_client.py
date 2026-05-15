"""OpenAI provider implementation."""

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
    from openai.types.chat.chat_completion import ChatCompletion


class OpenAIClient(Client[ChatCompletion]):
    """Client wrapper for OpenAI APIs."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        """Initialize the OpenAI client.

        Args:
            model: OpenAI model identifier.
            api_key: OpenAI API key. If omitted, the SDK will use
                ``OPENAI_API_KEY`` from the environment.
        """
        super().__init__(model=model)
        self._api_key = api_key

    def _process_response(self, response: ChatCompletion) -> Response:
        """Process OpenAI response and handle stop reasons.

        Args:
            response: Raw response object from the OpenAI SDK.

        Returns:
            A Response object with the generated text and metadata.
        """
        choice = response.choices[0] if response.choices else None

        finish_reason = choice.finish_reason if choice else None
        num_output_tokens = getattr(response.usage, "completion_tokens", None)
        text = (
            choice.message.content.strip()
            if choice and choice.message.content
            else ""
        )

        self._handle_stop_reason(
            stop_reason=finish_reason,
            num_output_tokens=num_output_tokens,
        )

        return Response(
            text=text,
            stop_reason=finish_reason,
            model=response.model,
            provider=Provider.OPENAI,
            num_output_tokens=num_output_tokens,
        )

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        options: ProviderRuntimeOptions | None = None,
    ) -> Response:
        """Generate a response using OpenAI.

        Args:
            messages: List of message dictionaries.
            options: Optional generation configuration.

        Returns:
            A Response object containing the generated response.

        Raises:
            ProviderError: If the provider call fails.
        """
        options = options or ProviderRuntimeOptions()
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self._api_key)

            request_payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": options.max_tokens,
            }
            response = client.chat.completions.create(**request_payload)
        except Exception as exc:
            raise ProviderError(f"OpenAI request failed: {exc}") from exc
        else:
            return self._process_response(response=response)

    def batch_generate(
        self,
        *,
        messages: list[list[dict[str, str]]],
        options: ProviderRuntimeOptions | None = None,
    ) -> list[Response]:
        """Batch generation for OpenAI is not currently implemented."""
        raise NotImplementedError(
            "Batch generation is not yet implemented for the OpenAI client."
        )

    def _handle_stop_reason(
        self,
        *,
        stop_reason: str | None,
        num_output_tokens: int | None,
    ) -> None:
        """Raise a provider error when OpenAI ended for a non-success reason."""
        token_suffix = (
            ""
            if num_output_tokens is None
            else f" (output tokens: {num_output_tokens:,})"
        )

        if stop_reason is None:
            raise ProviderError(
                f"OpenAI response is missing finish reason{token_suffix}."
            )
        elif stop_reason == "stop":
            return  # Normal completion, no error
        elif stop_reason == "length":
            raise ProviderError(
                f"OpenAI response stopped due to max token limit{token_suffix}."
            )
        elif stop_reason == "content_filter":
            raise ProviderError(
                f"OpenAI response was filtered due to content{token_suffix}."
            )
        elif stop_reason == "tool_calls":
            raise ProviderError(
                f"OpenAI response stopped after calling a tool{token_suffix}."
            )
        elif stop_reason == "function_call":
            raise ProviderError(
                f"OpenAI response stopped after calling a function{token_suffix}."
            )
        else:
            raise ProviderError(
                f"OpenAI response stopped for unknown reason '{stop_reason}'{token_suffix}."
            )
