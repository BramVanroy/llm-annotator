"""OpenAI provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast

from llm_annotator.clients.base import (
    Client,
    OnError,
    Provider,
    ProviderRuntimeOptions,
    Response,
)
from llm_annotator.clients.exceptions import ProviderError


if TYPE_CHECKING:
    from openai.types.chat.chat_completion import ChatCompletion

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class OpenAIRuntimeOptions(ProviderRuntimeOptions):
    frequency_penalty: float | None = None
    """Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim."""
    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None
    ) = None
    """Only for supported [reasoning models](https://platform.openai.com/docs/guides/reasoning). Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response."""
    temperature: float | None = None
    """What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or `top_p` but not both."""
    top_p: float | None = None
    """An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered."""
    presence_penalty: float | None = None
    """Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics."""


# Annoying typing magic to make sure OpenAI and vLLMClients work
# VLLMClient uses its own RuntimeOptions but shares the same structured and most of its methods
# so we need to make the base Client generic in the options type, and then have OpenAIClient and VLLMClient
# specify their own options type while still being recognized as Clients for the shared code to work.
T_OpenAIOptions = TypeVar("T_OpenAIOptions", bound=ProviderRuntimeOptions)


class OpenAIClient(Client[T_OpenAIOptions], Generic[T_OpenAIOptions]):
    """Client wrapper for OpenAI APIs."""

    provider_type = Provider.OPENAI

    def __init__(
        self,
        model: str,
        max_workers: int = 4,
        base_url: str | None = None,
        api_key: str | None = None,
        on_error: OnError = "raise",
    ) -> None:
        """Initialize the OpenAI client.

        Args:
            model: OpenAI model identifier.
            max_workers: Maximum number of concurrent worker threads for ``batch_generate``. Lower this value if
                you are getting rate limited.
            base_url: Base URL for the OpenAI API endpoint.
            api_key: OpenAI API key. If omitted, the SDK will use
                ``OPENAI_API_KEY`` from the environment.
            on_error: Error behavior when generation fails. Valid options are:
                - ``"raise"``: raise a :class:`ProviderError` (default).
                - ``"ignore"``: return a :class:`Response` with ``error`` set.
                - ``"warn"``: log a warning and return an error :class:`Response`.
        """
        from openai import OpenAI

        super().__init__(
            model=model, max_workers=max_workers, on_error=on_error
        )
        self._api_key = api_key
        self._base_url = base_url
        self._client = OpenAI(api_key=self._api_key, base_url=base_url)

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
            provider=self.provider_type,
            num_output_tokens=num_output_tokens,
            full_response=response,
        )

    def _default_options(self) -> T_OpenAIOptions:
        """Return default runtime options for this OpenAI-compatible client."""
        return cast(T_OpenAIOptions, OpenAIRuntimeOptions())

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        options: T_OpenAIOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> Response:
        """Generate a response using OpenAI.

        Args:
            messages: List of message dictionaries.
            options: Optional generation configuration.
            gen_kwargs: Additional provider-specific generation kwargs that are not covered by the standard options.
                Has precedence over ``options``.

        Returns:
            A Response object containing the generated response.

        Raises:
            ProviderError: If the provider call fails.
        """
        options = options or self._default_options()
        try:
            request_payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": options.max_tokens,
            }
            if options.json_schema is not None:
                request_payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": options.json_schema,
                        "strict": True,
                    },
                }
            opts = options.dict(exclude_none=True)
            opts.pop("max_tokens", None)
            opts.pop("json_schema", None)
            request_payload.update(opts)
            request_payload.update(gen_kwargs or {})
            response = self._client.chat.completions.create(**request_payload)
        except Exception as exc:
            return self._handle_error(exc, context="OpenAI request failed")
        else:
            try:
                return self._process_response(response=response)
            except Exception as exc:
                return self._handle_error(
                    exc, context="OpenAI response processing failed"
                )

    def batch_generate(
        self,
        *,
        messages: list[list[dict[str, str]]],
        options: T_OpenAIOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> list[Response]:
        """Generate responses for a batch of inputs concurrently.

        The OpenAI API has no native synchronous batch endpoint, so requests
        are dispatched in parallel using a thread pool.

        Args:
            messages: List of message lists, one per request.
            options: Optional generation configuration.
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
                        context=f"OpenAI batch request failed at index {idx}",
                    )
                )
        return responses

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
                f"Response is missing finish reason{token_suffix}."
            )
        elif stop_reason == "stop":
            return  # Normal completion, no error
        elif stop_reason == "length":
            raise ProviderError(
                f"Response stopped due to max token limit{token_suffix}."
            )
        elif stop_reason == "content_filter":
            raise ProviderError(
                f"Response was filtered due to content{token_suffix}."
            )
        elif stop_reason == "tool_calls":
            raise ProviderError(
                f"Response stopped after calling a tool{token_suffix}."
            )
        elif stop_reason == "function_call":
            raise ProviderError(
                f"Response stopped after calling a function{token_suffix}."
            )
        else:
            raise ProviderError(
                f"Response stopped for unknown reason '{stop_reason}'{token_suffix}."
            )


__all__ = ["OpenAIClient", "OpenAIRuntimeOptions"]
