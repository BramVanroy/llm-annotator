"""Online vLLM provider: a running OpenAI-compatible vLLM server.

The counterpart that loads the weights in-process instead of talking to a
server is
[`vllm_offline_client`][llm_annotator.clients.vllm_offline_client]. Both share
[`VLLMBaseRuntimeOptions`][llm_annotator.clients.vllm_online_client.VLLMBaseRuntimeOptions],
which lives here.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from llm_annotator.clients.base import (
    OnError,
    Provider,
    ProviderRuntimeOptions,
    Response,
    reject_multiple_responses,
)
from llm_annotator.clients.openai_client import CONNECT_TIMEOUT, OpenAIClient
from llm_annotator.logging_utils import get_logger


if TYPE_CHECKING:
    from openai import OpenAI


LOGGER = get_logger("clients.vllm_online")

MAX_CONNECTIONS = 8192
"""How many sockets one client may hold open to its server at once.

The OpenAI SDK caps its HTTP client at 1000 connections
(``DEFAULT_CONNECTION_LIMITS`` in ``openai/_constants.py``), and a pooled run
can ask for more: a ``batch_size`` of 1024 times the default
``max_concurrent_batches_per_client`` of 4 is 4096 requests in flight through
one client. Above the cap httpx queues the rest, which would keep part of a
batch from reaching the server that is supposed to schedule it."""


def server_is_healthy(base_url: str, timeout: float) -> bool:
    """Check whether a vLLM server answers its ``/health`` endpoint.

    Args:
        base_url: Base URL of the server, with or without the ``/v1`` suffix.
        timeout: Maximum number of seconds to wait for an answer.

    Returns:
        Whether the server answers with status 200.
    """
    health = f"{base_url.removesuffix('/v1').rstrip('/')}/health"
    try:
        with urllib.request.urlopen(health, timeout=timeout) as response:
            return int(response.status) == 200
    except (urllib.error.URLError, OSError) as exc:
        LOGGER.debug(f"vLLM server at '{base_url}' does not answer: {exc}")
        return False


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
        chat_template_kwargs: Additional kwargs forwarded to the chat template.
            Pass ``{"enable_thinking": True}`` here to enable thinking mode.
        extra_body: Any other parameter the backend accepts, merged into the
            request last. This is the escape hatch for everything the fields
            above do not name, such as ``min_p`` or ``stop_token_ids``. It
            cannot ask for more than one response per sample: the clients read
            the first one and drop the rest, so an ``n`` above 1 is rejected.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop: list[str] | None = None
    seed: int | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    extra_body: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Reject an ``extra_body`` that asks for more than one response.

        Raises:
            ValueError: If ``extra_body`` sets ``n`` to anything but 1.
        """
        reject_multiple_responses({"extra_body": self.extra_body or {}})

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

    # Payload keys vLLM accepts but the OpenAI SDK's typed create() does not,
    # so they have to travel inside extra_body.
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
        """Build the flat JSON body for vLLM's chat-completions route.

        An ``output_schema`` becomes a ``response_format`` of type
        ``json_schema``, which is the form vLLM 0.29 reads on
        ``/v1/chat/completions``: ``structured_outputs_from_response_format``
        (``vllm/entrypoints/generate/base/protocol.py``) turns it into the
        engine's ``StructuredOutputsParams(json=...)``, and it overrides a
        ``structured_outputs`` block in the same body.
        [`split_payload`][llm_annotator.clients.vllm_online_client.VLLMOnlineRuntimeOptions.split_payload]
        divides the result over the SDK's typed arguments and ``extra_body``.

        Returns:
            A dict of vLLM server request parameters, including all shared
            base fields.

        Examples:
            >>> fmt = VLLMOnlineRuntimeOptions(
            ...     output_schema={"type": "object"}
            ... ).to_payload()["response_format"]
            >>> fmt["type"], fmt["json_schema"]["name"]
            ('json_schema', 'response')
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
        if self.output_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": self.output_schema,
                    "strict": True,
                },
            }
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
    """Client for a running vLLM OpenAI-compatible server.

    A batch is one ``/v1/chat/completions`` request per conversation, sent
    concurrently. The server batches the requests it holds continuously, so
    they are scheduled together on the GPU, every request carries its own
    ``usage`` (which fills ``num_output_tokens`` per sample), and a
    conversation that fails is a single error
    [`Response`][llm_annotator.clients.base.Response] rather than a failure of
    the whole batch.
    """

    provider_type = Provider.VLLM_ONLINE

    def __init__(
        self,
        model: str | None = None,
        base_url: str = "http://localhost:8000/v1",
        max_workers: int | None = None,
        timeout: float = 3600.0,
        max_retries: int = 2,
        on_error: OnError = "warn",
    ) -> None:
        """Initialize the online vLLM client.

        Args:
            model: Model identifier. When omitted, the server is asked which
                model it serves.
            base_url: Base URL for the vLLM API endpoint.
            max_workers: Maximum number of requests
                [`batch_generate`][llm_annotator.clients.vllm_online_client.VLLMOnlineClient.batch_generate]
                sends at once. ``None`` sends the whole batch, which is what
                lets the server schedule it as one workload. Lower it only to
                protect a server that is shared with other jobs.
            timeout: Seconds one request may spend reading its answer. It
                covers the wait in the server's queue as well as the
                generation itself, so a value below the time a full batch
                needs turns a healthy run into errors. The default of one hour
                fits a loaded server that holds thousands of prompts. Making
                the connection has its own, short limit, so a host that drops
                packets rather than refusing the connection gives
                [`VLLMQueueAnnotator`][llm_annotator.annotator.VLLMQueueAnnotator]
                the errors it evicts the server on within seconds.
            max_retries: How often the OpenAI SDK retries a request. It
                retries connection errors, request timeouts and the status
                codes 408, 409, 429 and 5xx, with an exponential backoff of
                0.5 to 8 seconds. Two retries cover a server that restarts
                without letting a broken pool member stall a batch for long.
            on_error: Error behavior when generation fails.
        """
        super().__init__(
            model=model or "",
            max_workers=max_workers,
            api_key="EMPTY",
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            on_error=on_error,
        )
        self.base_url = base_url

        if model is None:
            models = self._client.models.list()
            self.model = models.data[0].id

    def _build_sdk_client(self) -> OpenAI:
        """Build an SDK client whose connection pool fits a full batch.

        The SDK caps its HTTP client at 1000 sockets, which is below what a
        pooled run sends at once, so the limits are raised to
        ``MAX_CONNECTIONS``.

        Returns:
            The SDK client.
        """
        import httpx
        from openai import DefaultHttpxClient, OpenAI

        return OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=httpx.Timeout(self.timeout, connect=CONNECT_TIMEOUT),
            max_retries=self.max_retries,
            http_client=DefaultHttpxClient(
                limits=httpx.Limits(
                    max_connections=MAX_CONNECTIONS,
                    max_keepalive_connections=MAX_CONNECTIONS,
                )
            ),
        )

    def is_healthy(self, timeout: float = 5.0) -> bool:
        """Check whether the server answers its ``/health`` endpoint.

        Args:
            timeout: Maximum number of seconds to wait for an answer.

        Returns:
            Whether the server answers with status 200.
        """
        return server_is_healthy(self.base_url, timeout)

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

        Args:
            messages: List of message dicts with "role" and "content" keys.
            options: Optional generation configuration.
            gen_kwargs: Additional request parameters, merged last so they take
                precedence over ``options``.

        Returns:
            A Response object containing the generated response. A failed
            request is an error Response when ``on_error`` is ``"warn"`` or
            ``"ignore"``.

        Raises:
            ProviderError: If the request fails and ``on_error`` is
                ``"raise"``.
            ValueError: If the request asks for more than one response.
        """
        resolved = options or self._default_options()
        request_payload, extra_body = resolved.split_payload()
        request_payload.update({"model": self.model, "messages": messages})
        request_payload.update(gen_kwargs or {})
        if extra_body:
            request_payload["extra_body"] = extra_body
        reject_multiple_responses(request_payload)

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
    ) -> list[Response]:
        """Generate one response per conversation, all in flight at once.

        Each conversation is its own ``/v1/chat/completions`` request. vLLM
        schedules whatever requests it holds as one continuous batch, so the
        GPU sees the same workload as a single combined request would give it,
        while the result stays per sample: response, ``usage`` and error
        belong to one conversation and cannot be mixed up.

        Args:
            messages: List of message lists, where each list is a conversation.
            options: Optional generation configuration.
            gen_kwargs: Additional provider-specific generation kwargs that are
                not covered by the standard options. Has precedence over
                ``options``.

        Returns:
            A list of Response objects, one per input conversation,
            indexed in the same order as input.

        Raises:
            ProviderError: If a request fails and ``on_error`` is ``"raise"``.
            ValueError: If the request asks for more than one response.
        """
        return self._generate_in_threads(
            messages=messages,
            options=options,
            gen_kwargs=gen_kwargs,
            max_workers=self.max_workers or len(messages),
            context="vLLM request failed",
        )

    def _default_options(self) -> VLLMOnlineRuntimeOptions:
        """Return default runtime options for vLLM requests."""
        return VLLMOnlineRuntimeOptions()


__all__ = [
    "VLLMBaseRuntimeOptions",
    "VLLMOnlineClient",
    "VLLMOnlineRuntimeOptions",
    "server_is_healthy",
]
