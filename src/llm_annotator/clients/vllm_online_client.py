"""Online vLLM provider: a running OpenAI-compatible vLLM server.

The counterpart that loads the weights in-process instead of talking to a
server is
[`vllm_offline_client`][llm_annotator.clients.vllm_offline_client]. Both share
[`VLLMBaseRuntimeOptions`][llm_annotator.clients.vllm_online_client.VLLMBaseRuntimeOptions],
which lives here.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Any, ClassVar

from llm_annotator.clients.base import (
    OnError,
    Provider,
    ProviderRuntimeOptions,
    Response,
)
from llm_annotator.clients.exceptions import ConfigurationError, ProviderError
from llm_annotator.clients.openai_client import OpenAIClient


@dataclass(slots=True, frozen=True)
class VLLMBaseRuntimeOptions(ProviderRuntimeOptions):
    """Shared generation options for both vLLM server and offline clients.

    Every field here means the same thing to both vLLM clients, so a step can
    be moved between ``vllm_online`` and ``vllm_offline`` without its decoding
    quietly changing. Fields that only one of the two accepts live on that
    subclass instead.

    Attributes:
        temperature: Sampling temperature. ``None`` uses the model default;
            ``0.0`` gives greedy, reproducible decoding.
        top_p: Nucleus-sampling probability mass. ``None`` uses the model
            default.
        top_k: Controls the number of top tokens to consider.
            Set to -1 to consider all tokens.
        repetition_penalty: Penalizes new tokens based on whether they appear
            in the prompt and the generated text so far. Values > 1 encourage
            the model to use new tokens; values < 1 encourage repetition.
        presence_penalty: Penalty applied to tokens already present in the
            output.
        frequency_penalty: Penalty applied proportional to token frequency in
            the output.
        stop: Optional list of strings that halt generation when produced.
        seed: Optional fixed random seed for reproducible generation.
        n: Number of independent output sequences to generate per request.
        chat_template_kwargs: Additional kwargs forwarded to the chat template.
            Pass ``{"enable_thinking": True}`` here to enable thinking mode.
        extra_body: Any other parameter the backend accepts, merged into the
            request last. This is the escape hatch for everything the fields
            above do not name, such as ``min_p`` or ``stop_token_ids``.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop: list[str] | None = None
    seed: int | None = None
    n: int | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    extra_body: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        """Build the request payload shared by both vLLM clients.

        ``chat_template_kwargs`` and ``extra_body`` are deliberately excluded:
        the two clients place them differently, so each subclass adds them.

        Returns:
            A dict of the parameters both vLLM backends spell identically.
        """
        payload: dict[str, Any] = {}
        for name in (
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "stop",
            "seed",
            "n",
        ):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload


@dataclass(slots=True, frozen=True)
class VLLMOnlineRuntimeOptions(VLLMBaseRuntimeOptions):
    """Generation options for the vLLM OpenAI-compatible server.

    Extends
    [`VLLMBaseRuntimeOptions`][llm_annotator.clients.vllm_online_client.VLLMBaseRuntimeOptions]
    with server-specific parameters from the `/v1/chat/completions`
    extra-params API.
    See https://docs.vllm.ai/en/latest/serving/openai_compatible_server/#api-reference

    Attributes:
        add_generation_prompt: If ``True``, appends a generation prompt to each
            message. Defaults to ``True``.
        chat_template: Optional chat template string. When omitted the model's
            default template is used.
        mm_processor_kwargs: Arguments forwarded to the model's multi-modal
            processor (e.g. ``{"num_crops": 4}`` for Phi-3-Vision).
    """

    add_generation_prompt: bool = True
    chat_template: str | None = None
    mm_processor_kwargs: dict[str, Any] | None = None

    # Payload keys vLLM accepts but the OpenAI SDK's typed create() does not.
    # On the SDK path they have to travel inside extra_body; the raw batch
    # endpoint takes them at the top level like everything else.
    _VLLM_ONLY_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "top_k",
            "repetition_penalty",
            "add_generation_prompt",
            "chat_template",
            "chat_template_kwargs",
            "mm_processor_kwargs",
        }
    )

    def to_payload(self) -> dict[str, Any]:
        """Build the flat JSON body for vLLM's own endpoints.

        Suitable for a request made directly against the server, where every
        parameter sits at the top level of the body.

        Returns:
            A dict of vLLM server request parameters, including all shared
            base fields.
        """
        payload = VLLMBaseRuntimeOptions.to_payload(self)
        if self.max_completion_tokens is not None:
            payload["max_completion_tokens"] = self.max_completion_tokens
        payload["add_generation_prompt"] = self.add_generation_prompt
        if self.chat_template is not None:
            payload["chat_template"] = self.chat_template
        if self.chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        if self.mm_processor_kwargs is not None:
            payload["mm_processor_kwargs"] = self.mm_processor_kwargs
        if self.extra_body:
            payload.update(self.extra_body)
        return payload

    def split_payload(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split the payload into OpenAI-typed kwargs and an ``extra_body``.

        The OpenAI SDK validates ``chat.completions.create`` against its own
        signature, so vLLM's extensions must be nested rather than passed as
        keyword arguments.

        Returns:
            ``(kwargs, extra_body)``, where ``kwargs`` goes to ``create()`` and
            ``extra_body`` is nested under its ``extra_body=`` parameter.

        Examples:
            >>> opts = VLLMOnlineRuntimeOptions(temperature=0.0, top_k=20)
            >>> standard, extra = opts.split_payload()
            >>> standard
            {'temperature': 0.0}
            >>> sorted(extra)
            ['add_generation_prompt', 'top_k']
        """
        payload = self.to_payload()
        extra_body = {
            key: payload.pop(key)
            for key in list(payload)
            if key in self._VLLM_ONLY_KEYS
        }
        if self.extra_body:
            # A key the user put in `extra_body` belongs there even when it is
            # one the SDK would have accepted.
            for key in self.extra_body:
                if key in payload:
                    extra_body[key] = payload.pop(key)
        return payload, extra_body


class VLLMOnlineClient(OpenAIClient[VLLMOnlineRuntimeOptions]):
    """Client for a running vLLM OpenAI-compatible server."""

    provider_type = Provider.VLLM_ONLINE

    def __init__(
        self,
        model: str | None = None,
        base_url: str = "http://localhost:8000/v1",
        on_error: OnError = "warn",
    ) -> None:
        """Initialize the online vLLM client.

        Args:
            model: Model identifier. When omitted, the server is asked which
                model it serves.
            base_url: Base URL for the vLLM API endpoint.
            on_error: Error behavior when generation fails.
        """
        super().__init__(
            model=model or "",
            api_key="EMPTY",
            base_url=base_url,
            on_error=on_error,
        )

        if model is None:
            models = self._client.models.list()
            self.model = models.data[0].id

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        options: VLLMOnlineRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> Response:
        """Generate a single response from the vLLM server.

        Overridden rather than inherited because vLLM's extensions to the chat
        API (``top_k``, ``chat_template_kwargs``, ...) are not part of the
        OpenAI SDK's typed ``create()`` signature and have to be nested under
        ``extra_body``.
        [`batch_generate`][llm_annotator.clients.vllm_online_client.VLLMOnlineClient.batch_generate]
        needs no such split: it posts the body itself.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            options: Optional generation configuration.
            gen_kwargs: Additional request parameters, merged last so they take
                precedence over ``options``.

        Returns:
            A Response object containing the generated response.
        """
        resolved = options or self._default_options()
        request_payload, extra_body = resolved.split_payload()
        request_payload.update({"model": self.model, "messages": messages})
        if resolved.json_schema is not None:
            request_payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": resolved.json_schema,
                    "strict": True,
                },
            }
        request_payload.update(gen_kwargs or {})
        if extra_body:
            request_payload["extra_body"] = extra_body

        try:
            response = self._client.chat.completions.create(**request_payload)
        except Exception as exc:
            return self._handle_error(exc, context="vLLM request failed")

        try:
            return self._process_response(response=response)
        except Exception as exc:
            return self._handle_error(
                exc, context="vLLM response processing failed"
            )

    def batch_generate(
        self,
        *,
        messages: list[list[dict[str, str]]],
        options: VLLMOnlineRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
        use_batch_api: bool = False,
        poll_interval: float = 10.0,
    ) -> list[Response]:
        """Generate responses for a batch of inputs using vLLM's native batch endpoint.

        Sends all conversations in a single request to ``/v1/chat/completions/batch``.
        The OpenAI Batch API is not supported; passing ``use_batch_api=True`` raises
        a [`ConfigurationError`][llm_annotator.clients.exceptions.ConfigurationError].

        !!! note "No per-sample token counts on this path"

            That endpoint reports one ``usage`` block for the whole batch
            rather than one per choice, so ``num_output_tokens`` is ``None`` on
            every [`Response`][llm_annotator.clients.base.Response] it returns,
            and a step's ``{prefix}num_tokens`` column is ``None`` with it.
            Everything else, including ``reasoning``, is per sample as usual.
            Use [`generate`][llm_annotator.clients.vllm_online_client.VLLMOnlineClient.generate]
            or the offline provider when the token counts matter.

        Args:
            messages: List of message lists, where each list is a conversation.
            options: Optional generation configuration.
            gen_kwargs: Additional provider-specific generation kwargs that are
                not covered by the standard options. Has precedence over
                ``options``.
            use_batch_api: Must be ``False``. The OpenAI Batch API is not
                supported by the vLLM server client.
            poll_interval: Accepted for interface compatibility with
                [`OpenAIClient`][llm_annotator.clients.openai_client.OpenAIClient].
                Ignored.

        Returns:
            A list of Response objects, one per input conversation,
            indexed in the same order as input.

        Raises:
            ConfigurationError: If ``use_batch_api=True``.
            ProviderError: If the batch request fails.
        """
        if use_batch_api:
            raise ConfigurationError(
                "The vLLM server client does not support the OpenAI Batch API."
                " Set use_batch_api=False (the default) to use vLLM's native"
                " batch endpoint instead."
            )
        # avoid unused variable warning for poll_interval, which is ignored
        _ = poll_interval
        import httpx
        from openai.types.chat.chat_completion import ChatCompletion

        options = options or self._default_options()
        try:
            # Construct batch request payload following vLLM batch API format
            request_payload: dict[str, Any] = options.to_payload()
            request_payload["model"] = self.model
            request_payload["messages"] = messages
            if options.json_schema is not None:
                # TODO: test. Maybe we need "structured_outputs"
                # https://docs.vllm.ai/en/latest/serving/openai_compatible_server/#extra-parameters_1
                request_payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": options.json_schema,
                        "strict": True,
                    },
                }
            request_payload.update(gen_kwargs or {})

            # The batch endpoint is at /v1/chat/completions/batch
            batch_url = f"{self._base_url}/chat/completions/batch"
            # Re-use the underlying httpx client
            http_client = self._client._client
            response = http_client.post(batch_url, json=request_payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            # exc's own message drops the response body, which is where
            # vLLM puts the actual reason (e.g. context length, unsupported
            # param), so surface it explicitly instead of just the status.
            error_response = self._handle_error(
                ProviderError(
                    f"vLLM batch endpoint returned"
                    f" {exc.response.status_code}: {exc.response.text}"
                ),
                context="vLLM batch request failed",
            )
            return [error_response for _ in messages]
        except Exception as exc:
            error_response = self._handle_error(
                exc, context="vLLM batch request failed"
            )
            return [error_response for _ in messages]

        # Process batch response: convert each choice to a Response object
        responses: list[Response] = []
        for idx, choice in enumerate(data.get("choices", [])):
            # Use random hash as id and current unix timestamp (int) as created
            # so that we can use self._process_response from super
            dummy_response = ChatCompletion(
                id=f"chatcmpl-{secrets.token_hex(12)}",
                object="chat.completion",
                created=int(time.time()),
                model=self.model,
                choices=[choice],
            )
            try:
                resp = self._process_response(response=dummy_response)
            except Exception as exc:
                resp = self._handle_error(
                    exc,
                    context=f"vLLM batch response processing failed at index {idx}",
                )

            responses.append(resp)

        if len(responses) < len(messages):
            padding = len(messages) - len(responses)
            err = self._handle_error(
                ProviderError(
                    "vLLM batch response returned fewer choices than requested."
                ),
                context="vLLM batch response validation failed",
            )
            responses.extend([err for _ in range(padding)])

        return responses

    def _default_options(self) -> VLLMOnlineRuntimeOptions:
        """Return default runtime options for vLLM requests."""
        return VLLMOnlineRuntimeOptions()


__all__ = [
    "VLLMBaseRuntimeOptions",
    "VLLMOnlineClient",
    "VLLMOnlineRuntimeOptions",
]
