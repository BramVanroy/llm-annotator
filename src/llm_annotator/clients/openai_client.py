"""OpenAI provider implementation."""

from __future__ import annotations

import io
import json
import time
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from llm_annotator.clients.base import (
    Client,
    OnError,
    Provider,
    ProviderRuntimeOptions,
    Response,
)
from llm_annotator.clients.exceptions import ProviderError
from llm_annotator.utils import add_schema_additional_properties_false


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

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}

        if self.max_tokens is not None:
            payload["max_completion_tokens"] = self.max_tokens
        if self.frequency_penalty is not None:
            payload["frequency_penalty"] = self.frequency_penalty
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.presence_penalty is not None:
            payload["presence_penalty"] = self.presence_penalty
        return payload


# OpenAIClient is declared generic so that VLLMClient can specialise it with
# VLLMRuntimeOptions while still inheriting OpenAI's HTTP machinery. Without
# the TypeVar, overriding ``generate`` / ``batch_generate`` with a different
# options type would be a Liskov-unsafe parameter narrowing and mypy would
# flag it.
T_OpenAIOptions = TypeVar("T_OpenAIOptions", bound=ProviderRuntimeOptions)


class OpenAIClient(Client[T_OpenAIOptions]):
    """Client wrapper for OpenAI APIs."""

    provider_type = Provider.OPENAI

    def __init__(
        self,
        model: str,
        max_workers: int = 4,
        base_url: str | None = None,
        api_key: str | None = None,
        on_error: OnError = "warn",
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
        self._active_batch_ids: list[str] = []

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

        partial = Response(
            text=text,
            stop_reason=finish_reason,
            model=response.model,
            provider=self.provider_type,
            num_output_tokens=num_output_tokens,
            full_response=response,
        )

        try:
            self._handle_stop_reason(
                stop_reason=finish_reason,
                num_output_tokens=num_output_tokens,
            )
        except Exception as exc:
            return self._handle_error(
                exc,
                context="OpenAI response stop reason",
                partial=partial,
            )

        return partial

    def destroy(self) -> None:
        """Cancel any in-flight batches and clean up resources."""
        for batch_id in list(self._active_batch_ids):
            try:
                self._client.batches.cancel(batch_id)
            except Exception:
                pass
        self._active_batch_ids.clear()

    def _build_batch_request(
        self,
        idx: int,
        messages: list[dict[str, str]],
        options: OpenAIRuntimeOptions,
        gen_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build a single Batch API request line for the given messages.

        Args:
            idx: Zero-based index used as the ``custom_id`` suffix.
            messages: Chat messages for this request.
            options: Generation options.
            gen_kwargs: Extra kwargs merged into the body (highest precedence).

        Returns:
            A dict representing one line of the JSONL batch input file.
        """
        body: dict[str, Any] = options.to_payload()
        body["model"] = self.model
        body["messages"] = messages
        if options.json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": add_schema_additional_properties_false(
                        options.json_schema
                    ),
                    "strict": True,
                },
            }
        body.update(gen_kwargs or {})
        return {
            "custom_id": f"request-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

    def _execute_batch_api(
        self,
        messages: list[list[dict[str, str]]],
        options: OpenAIRuntimeOptions,
        gen_kwargs: dict[str, Any] | None,
        poll_interval: float,
    ) -> list[Response]:
        """Run the OpenAI Batch API path for ``batch_generate``.

        Uploads a JSONL file, creates a batch job, polls until the job
        finishes, then downloads and parses the output.

        Args:
            messages: One list of message dicts per request.
            options: Generation options applied to every request in the batch.
            gen_kwargs: Extra kwargs merged into every request body.
            poll_interval: Seconds to wait between status-poll calls.

        Returns:
            Responses in the same order as the input ``messages``.
        """
        from openai.types.chat.chat_completion import ChatCompletion

        # Build JSONL content in memory.
        lines = [
            json.dumps(
                self._build_batch_request(idx, msgs, options, gen_kwargs)
            )
            for idx, msgs in enumerate(messages)
        ]
        jsonl_bytes = "\n".join(lines).encode()
        jsonl_file = io.BytesIO(jsonl_bytes)

        # Upload input file.
        uploaded = self._client.files.create(
            file=("batch.jsonl", jsonl_file, "application/jsonl"),
            purpose="batch",
        )
        file_id: str = uploaded.id

        # Create the batch.
        batch = self._client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_id: str = batch.id
        self._active_batch_ids.append(batch_id)

        # Poll until the batch reaches a terminal state.
        terminal_statuses = {"completed", "failed", "expired", "cancelled"}
        while batch.status not in terminal_statuses:
            time.sleep(poll_interval)
            batch = self._client.batches.retrieve(batch_id)

        self._active_batch_ids.remove(batch_id)

        # Handle batch-level failure.
        if batch.status != "completed":
            error_msg = f"Batch {batch_id} ended with status '{batch.status}'."
            return [
                self._handle_error(
                    ProviderError(error_msg),
                    context=f"OpenAI batch API failed at index {idx}",
                )
                for idx in range(len(messages))
            ]

        # Download and parse the output JSONL.
        assert batch.output_file_id is not None
        output_content = self._client.files.content(batch.output_file_id)
        result_map: dict[str, dict[str, Any]] = {}
        for raw_line in output_content.text.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            parsed_line: dict[str, Any] = json.loads(raw_line)
            result_map[parsed_line["custom_id"]] = parsed_line

        responses: list[Response] = []
        for idx in range(len(messages)):
            custom_id = f"request-{idx}"
            entry: dict[str, Any] | None = result_map.get(custom_id)
            if entry is None:
                responses.append(
                    self._handle_error(
                        ProviderError(f"No output entry for '{custom_id}'."),
                        context=f"OpenAI batch API missing result at index {idx}",
                    )
                )
                continue

            if entry.get("error") is not None:
                responses.append(
                    self._handle_error(
                        ProviderError(str(entry["error"])),
                        context=f"OpenAI batch API item error at index {idx}",
                    )
                )
                continue

            item_response = entry.get("response", {})
            if item_response.get("status_code") != 200:
                responses.append(
                    self._handle_error(
                        ProviderError(
                            f"Unexpected status code {item_response.get('status_code')} "
                            f"for '{custom_id}'."
                        ),
                        context=f"OpenAI batch API bad status at index {idx}",
                    )
                )
                continue

            try:
                completion = ChatCompletion.model_validate(
                    item_response["body"]
                )
                responses.append(self._process_response(completion))
            except Exception as exc:
                responses.append(
                    self._handle_error(
                        exc,
                        context=f"OpenAI batch API response processing failed at index {idx}",
                    )
                )

        return responses

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
        resolved = cast(
            OpenAIRuntimeOptions, options or self._default_options()
        )
        try:
            request_payload: dict[str, Any] = resolved.to_payload()

            request_payload.update(
                {
                    "model": self.model,
                    "messages": messages,
                }
            )
            if resolved.json_schema is not None:
                request_payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": add_schema_additional_properties_false(
                            resolved.json_schema
                        ),
                        "strict": True,
                    },
                }
            request_payload.update(gen_kwargs or {})
            response = self._client.chat.completions.create(**request_payload)
        except Exception as exc:
            # API errors specifically can always be raised
            raise exc
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
        use_batch_api: bool = False,
        poll_interval: float = 10.0,
    ) -> list[Response]:
        """Generate responses for a batch of inputs.

        By default, requests are dispatched in parallel using a thread pool.
        When ``use_batch_api=True``, the OpenAI Batch API is used instead:
        all requests are submitted as a single batch job and results are
        retrieved once the job completes. The Batch API supports a completion
        window of up to 24 hours and offers lower cost, but adds latency.

        Args:
            messages: List of message lists, one per request.
            options: Optional generation configuration.
            gen_kwargs: Additional provider-specific generation kwargs that are not covered by the standard options.
                Has precedence over ``options``.
            use_batch_api: When ``True``, use the OpenAI Batch API instead of
                concurrent individual requests. Defaults to ``False``.
            poll_interval: Seconds between batch status polls. Only used when
                ``use_batch_api=True``. Defaults to ``10.0``.

        Returns:
            A list of Response objects in the same order as the input.

        Raises:
            ProviderError: If any individual request fails.
        """
        if use_batch_api:
            resolved = cast(
                OpenAIRuntimeOptions, options or self._default_options()
            )
            return self._execute_batch_api(
                messages, resolved, gen_kwargs, poll_interval
            )

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
