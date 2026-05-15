"""Google Gemini provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_annotator.client.base import (
    Client,
    Provider,
    ProviderRuntimeOptions,
    Response,
)
from llm_annotator.client.exceptions import ProviderError


if TYPE_CHECKING:
    from google.genai.types import GenerateContentResponse


class GeminiClient(Client[GenerateContentResponse]):
    """Client wrapper for Gemini APIs."""
    provider_type = Provider.GEMINI
    def __init__(self, model: str, api_key: str | None = None) -> None:
        """Initialize the Gemini client.

        Args:
            model: Gemini model identifier.
            api_key: Gemini API key. If omitted, the SDK will use
                ``GEMINI_API_KEY`` from the environment.
        """

        from google import genai

        super().__init__(model=model)
        self._api_key = api_key
        self._client = genai.Client(api_key=self._api_key)

    def _process_response(self, response: GenerateContentResponse) -> Response:
        """Process Gemini response and handle stop reasons.

        Args:
            response: Raw response object from the Gemini SDK.
        Returns:
            A Response object with the generated text and metadata.
        """
        candidate = response.candidates[0] if response.candidates else None
        finish_reason = getattr(candidate, "finish_reason", None)
        finish_reason = finish_reason.value if finish_reason else None
        num_output_tokens = getattr(candidate, "token_count", None)

        self._handle_stop_reason(
            stop_reason=finish_reason,
            num_output_tokens=num_output_tokens,
        )

        return Response(
            text=(response.text or "").strip(),
            stop_reason=finish_reason,
            model=response.model_version,
            provider=self.provider_type,
            num_output_tokens=num_output_tokens,
        )

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        options: ProviderRuntimeOptions | None = None,
    ) -> Response:
        """Generate a response using Gemini.

        Args:
            messages: List of message dictionaries.
            options: Optional generation configuration.

        Returns:
            A Response object containing the generated response.

        Raises:
            ProviderError: If the provider call fails.
        """
        from google.genai import types

        options = options or ProviderRuntimeOptions()
        system_instruction, prompt_text = _messages_to_prompt(messages)

        config = types.GenerateContentConfig(
            max_output_tokens=options.max_tokens,
            system_instruction=system_instruction or None,
        )
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt_text,
                config=config,
            )
        except Exception as exc:
            raise ProviderError(f"Gemini request failed: {exc}") from exc
        else:
            return self._process_response(response=response)

    def batch_generate(
        self,
        *,
        messages: list[list[dict[str, str]]],
        options: ProviderRuntimeOptions | None = None,
    ) -> list[Response]:
        """Batch generation for Gemini is not currently implemented."""
        raise NotImplementedError(
            "Batch generation is not yet implemented for the Gemini client."
        )

    def _handle_stop_reason(
        self,
        *,
        stop_reason: str | None,
        num_output_tokens: int | None,
    ) -> None:
        """Raise a provider error when Gemini ended for a non-success reason."""
        token_suffix = (
            ""
            if num_output_tokens is None
            else f" (output tokens: {num_output_tokens:,})"
        )
        if stop_reason in {None, "STOP"}:
            return
        if stop_reason == "MAX_TOKENS":
            raise ProviderError(
                "Gemini stopped because it hit the configured output token "
                f"limit before completing the response{token_suffix}."
            )
        if stop_reason in {
            "SAFETY",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "RECITATION",
            "IMAGE_SAFETY",
            "IMAGE_PROHIBITED_CONTENT",
            "IMAGE_RECITATION",
        }:
            raise ProviderError(
                "Gemini blocked the response for safety or policy reasons"
                f"{token_suffix}."
            )
        if stop_reason in {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"}:
            raise ProviderError(
                "Gemini attempted a tool or function interaction instead of "
                f"returning plain text{token_suffix}."
            )
        if stop_reason == "LANGUAGE":
            raise ProviderError(
                "Gemini stopped because of a language handling issue before "
                f"completing the response{token_suffix}."
            )
        if stop_reason == "NO_IMAGE":
            raise ProviderError(
                "Gemini expected image input and could not continue with the "
                f"provided request{token_suffix}."
            )
        raise ProviderError(
            "Gemini stopped for unexpected finish reason "
            f"{stop_reason!r}{token_suffix}."
        )


def _messages_to_prompt(
    messages: list[dict[str, str]],
) -> tuple[str, str]:
    """Convert OpenAI-style messages to Gemini input text and instruction.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.
    Returns:
        A tuple of (system_instruction, prompt_text) to be used for Gemini generation.
    """
    system_instruction = ""
    prompt = ""
    had_system_msg = False
    for msg_idx, message in enumerate(messages):
        role = message["role"]
        content = message["content"]

        if role == "system":
            if had_system_msg:
                raise ProviderError(
                    "For Gemini, only a single system message is supported."
                )

            if msg_idx != 0:
                raise ValueError(
                    "Make sure that the system message is the first message in the list."
                )
            system_instruction = content
            had_system_msg = True
        elif role == "user":
            if msg_idx > 1:
                raise ValueError(
                    "For Gemini, only a single user message is supported."
                )
            prompt = content
        else:
            raise ValueError(
                f"Unsupported message role {role!r} for Gemini client. Only 'system' and 'user' roles are supported."
            )

    if not prompt:
        raise ValueError(
            "A user message with role 'user' is required for Gemini generation."
        )
    return system_instruction, prompt
