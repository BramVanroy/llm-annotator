"""OpenAI provider implementation."""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from llm_annotator.clients.base import (
    Client,
    OnError,
    Provider,
    ProviderRuntimeOptions,
    Response,
    reject_multiple_responses,
)
from llm_annotator.clients.exceptions import ProviderError
from llm_annotator.utils import add_schema_additional_properties_false


if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat.chat_completion import ChatCompletion

DEFAULT_TIMEOUT = 600.0
"""Seconds one request may take, the OpenAI SDK's own default.

``DEFAULT_TIMEOUT`` in ``openai/_constants.py`` is
``httpx.Timeout(timeout=600, connect=5.0)``."""

DEFAULT_MAX_RETRIES = 2
"""How often the OpenAI SDK retries a failed request, its own default
(``DEFAULT_MAX_RETRIES`` in ``openai/_constants.py``)."""

CONNECT_TIMEOUT = 5.0
"""Seconds to wait for the TCP connection of one request, apart from the
generation itself.

httpx reads a plain float timeout as all four of its limits, so a long
``timeout`` would also let a host that drops packets rather than refusing the
connection hold a request for the full duration. This is the OpenAI SDK's own
connect timeout (``DEFAULT_TIMEOUT.connect`` in ``openai/_constants.py``)."""


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

        if self.max_completion_tokens is not None:
            payload["max_completion_tokens"] = self.max_completion_tokens
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


# OpenAIClient is declared generic so that VLLMOnlineClient can specialise it with
# VLLMOnlineRuntimeOptions while still inheriting OpenAI's HTTP machinery. Without
# the TypeVar, overriding ``generate`` / ``batch_generate`` with a different
# options type would be a Liskov-unsafe parameter narrowing and mypy would
# flag it.
T_OpenAIOptions = TypeVar("T_OpenAIOptions", bound=ProviderRuntimeOptions)


REASONING_FIELDS = ("reasoning", "reasoning_content")
"""Names an OpenAI-compatible server may use for a reasoning trace it returns
outside the message content. Neither is in the OpenAI schema, so both arrive as
extra fields on the message model. vLLM emits ``reasoning``; several hosted
OpenAI-compatible APIs emit ``reasoning_content``."""


def _json_schema_response_format(schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap a JSON schema as an OpenAI ``response_format`` value.

    Args:
        schema: The JSON schema the response must match.

    Returns:
        The ``response_format`` value for a chat-completions request.

    Examples:
        >>> fmt = _json_schema_response_format({"type": "object"})
        >>> (
        ...     fmt["type"],
        ...     fmt["json_schema"]["name"],
        ...     fmt["json_schema"]["strict"],
        ... )
        ('json_schema', 'response', True)
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": add_schema_additional_properties_false(schema),
            "strict": True,
        },
    }


def _message_reasoning(message: object) -> str | None:
    """Read the reasoning trace off a chat message, whatever the server calls it.

    Args:
        message: The ``message`` of one chat-completion choice.

    Returns:
        The trace, stripped, or ``None`` when the server returned none.

    Examples:
        >>> from types import SimpleNamespace
        >>> _message_reasoning(SimpleNamespace(reasoning=" first I think "))
        'first I think'
        >>> _message_reasoning(SimpleNamespace(content="no trace")) is None
        True
    """
    for field in REASONING_FIELDS:
        value = getattr(message, field, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


class OpenAIClient(Client[T_OpenAIOptions]):
    """Client wrapper for OpenAI APIs."""

    provider_type = Provider.OPENAI

    def __init__(
        self,
        model: str,
        max_workers: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        use_batch_api: bool = False,
        batch_poll_interval: float = 10.0,
        on_error: OnError = "warn",
    ) -> None:
        """Initialize the OpenAI client.

        Args:
            model: OpenAI model identifier.
            max_workers: Maximum number of concurrent worker threads for ``batch_generate``. Lower this value if
                you are getting rate limited. If set to None, 1 or lower, multithreading will be disabled.
            base_url: Base URL for the OpenAI API endpoint.
            api_key: OpenAI API key. If omitted, the SDK will use
                ``OPENAI_API_KEY`` from the environment.
            timeout: Seconds one request may take. The default is the SDK's
                own (``DEFAULT_TIMEOUT`` in ``openai/_constants.py``).
            max_retries: How often the SDK retries a request it can retry
                (connection errors, timeouts, and the status codes 408, 409,
                429 and 5xx). The default is the SDK's own
                (``DEFAULT_MAX_RETRIES`` in ``openai/_constants.py``).
            use_batch_api: Whether
                [`batch_generate`][llm_annotator.clients.openai_client.OpenAIClient.batch_generate]
                submits its requests to the OpenAI Batch API instead of
                sending them over a thread pool. The Batch API has a
                completion window of up to 24 hours and costs less, at the
                price of latency.
            batch_poll_interval: Seconds between two status polls of a running
                Batch API job. Only read when ``use_batch_api`` is ``True``.
            on_error: Error behavior when generation fails. Valid options are:
                - ``"warn"``: log a warning and return a
                  [`Response`][llm_annotator.clients.base.Response] with
                  ``error`` set (the default).
                - ``"ignore"``: return that error ``Response`` without the
                  warning.
                - ``"raise"``: raise a
                  [`ProviderError`][llm_annotator.clients.exceptions.ProviderError].
        """
        super().__init__(
            model=model, max_workers=max_workers, on_error=on_error
        )
        self._api_key = api_key
        self._base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_batch_api = use_batch_api
        self.batch_poll_interval = batch_poll_interval
        self._client = self._build_sdk_client()
        self._active_batches: dict[str, list[str]] = {}
        """Batch API jobs that have not been cleaned up yet, each mapped to the
        ids of the files it owns: the uploaded input file, plus the output and
        error files once the job reports them."""

    def _build_sdk_client(self) -> OpenAI:
        """Build the OpenAI SDK client this client sends its requests with.

        Override this in a subclass that needs other transport settings than
        ``timeout`` and ``max_retries`` cover. It runs once, at the end of
        ``__init__``, and reads the attributes set before it.

        Returns:
            The SDK client.
        """
        import httpx
        from openai import OpenAI

        return OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=httpx.Timeout(self.timeout, connect=CONNECT_TIMEOUT),
            max_retries=self.max_retries,
        )

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
        reasoning = _message_reasoning(choice.message) if choice else None

        partial = Response(
            text=text,
            stop_reason=finish_reason,
            model=response.model,
            provider=self.provider_type,
            num_output_tokens=num_output_tokens,
            full_response=response,
            reasoning=reasoning,
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
        """Cancel the batches still tracked and delete the files they own.

        A cancellation or deletion that fails is logged at warning level and
        does not stop the remaining clean-up.
        """
        for batch_id in list(self._active_batches):
            try:
                self._client.batches.cancel(batch_id)
            except Exception as exc:
                self._logger.warning(
                    f"Could not cancel batch {batch_id}: {exc}"
                )
            self._delete_batch_files(batch_id)

    def _delete_batch_files(self, batch_id: str) -> None:
        """Delete the files of one batch and stop tracking it.

        Args:
            batch_id: Id of the batch whose files are deleted.
        """
        for file_id in self._active_batches.pop(batch_id, []):
            try:
                self._client.files.delete(file_id)
            except Exception as exc:
                self._logger.warning(
                    f"Could not delete file {file_id} of batch"
                    f" {batch_id}: {exc}"
                )

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
        if options.output_schema is not None:
            body["response_format"] = _json_schema_response_format(
                options.output_schema
            )
        body.update(gen_kwargs or {})
        reject_multiple_responses(body)
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
    ) -> list[Response]:
        """Run the OpenAI Batch API path for ``batch_generate``.

        Uploads a JSONL file, creates a batch job, polls until the job reaches
        a final status, then reads whatever results the job produced. A batch
        that ends as ``expired`` or ``cancelled`` can still carry finished
        requests in its output file, so those are read as well and only the
        ``custom_id``s without a result become errors. The input, output and
        error files are deleted before the method returns, including when the
        results could not be read.

        Args:
            messages: One list of message dicts per request.
            options: Generation options applied to every request in the batch.
            gen_kwargs: Extra kwargs merged into every request body.

        Returns:
            Responses in the same order as the input ``messages``.
        """
        lines = [
            json.dumps(
                self._build_batch_request(idx, msgs, options, gen_kwargs)
            )
            for idx, msgs in enumerate(messages)
        ]
        jsonl_file = io.BytesIO("\n".join(lines).encode())

        uploaded = self._client.files.create(
            file=("batch.jsonl", jsonl_file, "application/jsonl"),
            purpose="batch",
        )
        batch = self._client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_id: str = batch.id
        self._active_batches[batch_id] = [uploaded.id]

        # The batch stays tracked while it runs, so that an interrupted poll
        # leaves destroy() a batch to cancel and an input file to delete.
        terminal_statuses = {"completed", "failed", "expired", "cancelled"}
        while batch.status not in terminal_statuses:
            self._logger.info(
                f"Batch {batch_id} status: {batch.status}. Polling again"
                f" in {self.batch_poll_interval} seconds..."
            )
            time.sleep(self.batch_poll_interval)
            batch = self._client.batches.retrieve(batch_id)

        result_files = [
            file_id
            for file_id in (batch.output_file_id, batch.error_file_id)
            if file_id
        ]
        self._active_batches[batch_id].extend(result_files)

        try:
            try:
                result_map = self._read_batch_results(result_files)
            except Exception as exc:
                return [
                    self._handle_error(
                        exc,
                        context=f"OpenAI batch API result download failed at index {idx}",
                    )
                    for idx in range(len(messages))
                ]

            return [
                self._process_batch_entry(
                    idx, result_map.get(f"request-{idx}"), batch.status
                )
                for idx in range(len(messages))
            ]
        finally:
            self._delete_batch_files(batch_id)

    def _read_batch_results(
        self, file_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Read the result files of a finished batch.

        Args:
            file_ids: Ids of the batch's output and error files, whichever it
                reported.

        Returns:
            One JSONL entry per ``custom_id`` the batch reported on. A batch
            that produced no file gives an empty mapping.
        """
        result_map: dict[str, dict[str, Any]] = {}
        for file_id in file_ids:
            content = self._client.files.content(file_id)
            for raw_line in content.text.splitlines():
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                entry: dict[str, Any] = json.loads(raw_line)
                result_map[entry["custom_id"]] = entry

        return result_map

    def _process_batch_entry(
        self,
        idx: int,
        entry: dict[str, Any] | None,
        status: str,
    ) -> Response:
        """Turn one JSONL entry of a batch result file into a Response.

        Args:
            idx: Zero-based index of the request in the input batch.
            entry: The entry the batch reported for it, or ``None`` when the
                batch reported none.
            status: Final status of the batch, named in the error message of a
                request the batch did not answer.

        Returns:
            The parsed [`Response`][llm_annotator.clients.base.Response], or an
            error ``Response`` when the request has no usable result.
        """
        from openai.types.chat.chat_completion import ChatCompletion

        custom_id = f"request-{idx}"
        if entry is None:
            return self._handle_error(
                ProviderError(
                    f"The batch ended with status '{status}' and holds no"
                    f" result for '{custom_id}'."
                ),
                context=f"OpenAI batch API missing result at index {idx}",
            )

        if entry.get("error") is not None:
            return self._handle_error(
                ProviderError(str(entry["error"])),
                context=f"OpenAI batch API item error at index {idx}",
            )

        item_response = entry.get("response") or {}
        if item_response.get("status_code") != 200:
            return self._handle_error(
                ProviderError(
                    f"Unexpected status code"
                    f" {item_response.get('status_code')} for"
                    f" '{custom_id}'."
                ),
                context=f"OpenAI batch API bad status at index {idx}",
            )

        try:
            completion = ChatCompletion.model_validate(item_response["body"])
            return self._process_response(completion)
        except Exception as exc:
            return self._handle_error(
                exc,
                context=f"OpenAI batch API response processing failed at index {idx}",
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
            A Response object containing the generated response. A failed
            request is an error Response when ``on_error`` is ``"warn"`` or
            ``"ignore"``.

        Raises:
            ProviderError: If the request fails and ``on_error`` is
                ``"raise"``.
            ValueError: If the request asks for more than one response.
        """
        resolved = cast(
            OpenAIRuntimeOptions, options or self._default_options()
        )
        request_payload: dict[str, Any] = resolved.to_payload()
        request_payload.update(
            {
                "model": self.model,
                "messages": messages,
            }
        )
        if resolved.output_schema is not None:
            request_payload["response_format"] = _json_schema_response_format(
                resolved.output_schema
            )
        request_payload.update(gen_kwargs or {})
        reject_multiple_responses(request_payload)

        try:
            response = self._client.chat.completions.create(**request_payload)
        except Exception as exc:
            return self._handle_error(exc, context="OpenAI request failed")

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
        """Generate responses for a batch of inputs.

        The requests go out over a thread pool, or as one OpenAI Batch API job
        when the client was built with ``use_batch_api=True``.

        Args:
            messages: List of message lists, one per request.
            options: Optional generation configuration.
            gen_kwargs: Additional provider-specific generation kwargs that are not covered by the standard options.
                Has precedence over ``options``.

        Returns:
            A list of Response objects in the same order as the input. A
            request that fails is an error Response when ``on_error`` is
            ``"warn"`` or ``"ignore"``.

        Raises:
            ProviderError: If a request fails and ``on_error`` is ``"raise"``.
        """
        if self.use_batch_api:
            resolved = cast(
                OpenAIRuntimeOptions, options or self._default_options()
            )
            return self._execute_batch_api(messages, resolved, gen_kwargs)

        return self._generate_in_threads(
            messages=messages,
            options=options,
            gen_kwargs=gen_kwargs,
            max_workers=self.max_workers,
            context="OpenAI request failed",
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
