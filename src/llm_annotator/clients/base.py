"""Abstract interface for LLM provider clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, ClassVar, Generic, Literal, Self, TypeVar

from llm_annotator.clients.exceptions import ProviderError
from llm_annotator.logging_utils import get_logger


# So that sub-classes can extend their runtime options without breaking typing
T_Options = TypeVar("T_Options", bound="ProviderRuntimeOptions")
OnError = Literal["raise", "ignore", "warn"]


class Provider(StrEnum):
    """Canonical name of the inference backend a client talks to.

    Each [`Client`][llm_annotator.clients.base.Client] subclass reports one of
    these through its ``provider_type``. Config files must spell the provider
    exactly as one of these values; anything else is rejected with a
    ``ValueError`` listing the valid options.
    """

    OPENAI = auto()
    CLAUDE = auto()
    VLLM_ONLINE = auto()
    VLLM_OFFLINE = auto()


@dataclass(slots=True, frozen=True)
class ProviderRuntimeOptions:
    """Shared generation options for provider calls; can be subclassed and extended.

    Attributes:
        max_completion_tokens: Optional maximum number of tokens to generate.
            This caps the completion only; it does not include the prompt.
        output_schema: Optional JSON schema for structured output. Every
            client sends it to its provider, which constrains the response to
            JSON that matches the schema. An ``Annotator`` sets it from its
            own ``output_schema`` argument, which is the only place to give
            it there.
    """

    max_completion_tokens: int | None = None
    output_schema: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        """Convert options to a provider-specific API request payload dict.

        Subclasses override this to build the exact kwargs expected by their SDK.
        The default implementation returns an empty dict.

        Returns:
            A dict of provider-specific request parameters.
        """
        return {}


def reject_multiple_responses(payload: dict[str, Any]) -> None:
    """Raise when a request asks the provider for more than one response.

    Every client reads the first response of a request and drops the rest, so
    a higher ``n`` spends tokens on output that is never stored. The check
    covers ``n`` at the top level of the payload and inside a nested
    ``extra_body``, which is where the vLLM server client puts the parameters
    the OpenAI SDK does not accept.

    Args:
        payload: The final request payload, after ``gen_kwargs`` and
            ``extra_body`` were merged in.

    Raises:
        ValueError: If the payload sets ``n`` to anything other than 1.

    Examples:
        >>> reject_multiple_responses({"n": 1, "temperature": 0.0})
        >>> reject_multiple_responses(
        ...     {"extra_body": {"n": 4}}
        ... )  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: 'n' is 4, but one response per sample is read...
    """
    for body in (payload, payload.get("extra_body") or {}):
        count = body.get("n")
        if count is not None and count != 1:
            raise ValueError(
                f"'n' is {count}, but one response per sample is read and the"
                " others are dropped. Remove 'n' or set it to 1."
            )


@dataclass(slots=True, frozen=True)
class Response:
    """Structured response object returned by provider clients."""

    text: str
    stop_reason: str | None = None
    model: str | None = None
    provider: Provider | None = None
    num_output_tokens: int | None = None
    full_response: object | None = None
    error: str | None = None
    error_type: str | None = None
    reasoning: str | None = None
    """The model's reasoning trace, separated from ``text``.

    Set when a thinking model's trace can be told apart from its answer: a
    vLLM server started with ``--reasoning-parser``, an offline vLLM client
    given a ``reasoning_parser`` name, or a Claude request with a thinking
    budget. A reasoning model run without a parser returns its trace inside
    ``text``, tags and all, and this stays ``None``.
    """


class Client(ABC, Generic[T_Options]):
    """Base client interface used by all provider adapters."""

    provider_type: ClassVar[Provider]

    def __init__(
        self,
        model: str,
        max_workers: int | None = None,
        on_error: OnError = "warn",
    ) -> None:
        """Initialize a provider client.

        Args:
            model: Provider-specific model name.
            max_workers: Maximum number of concurrent worker threads for ``batch_generate``. Clients that support native batching may ignore this parameter.
            on_error: Error behavior for provider failures.
                - ``"warn"``: log a warning and return a
                  [`Response`][llm_annotator.clients.base.Response] with
                  ``error`` set (the default).
                - ``"ignore"``: return that error ``Response`` without the
                  warning.
                - ``"raise"``: raise a
                  [`ProviderError`][llm_annotator.clients.exceptions.ProviderError].
        """
        if on_error not in {"raise", "ignore", "warn"}:
            raise ValueError(
                "'on_error' must be one of: 'raise', 'ignore', 'warn'."
            )

        self.model = model
        self.max_workers = max_workers
        self.on_error = on_error
        self._logger = get_logger(f"clients.{self.provider_type.value}")

    def _handle_error(
        self,
        exc: Exception,
        *,
        context: str,
        partial: Response | None = None,
    ) -> Response:
        """Handle provider errors according to ``self.on_error`` policy.

        When ``partial`` is provided (e.g. after a response was already
        partially decoded), its fields are forwarded to the returned
        [`Response`][llm_annotator.clients.base.Response] so callers retain the
        generated text, stop reason, token counts, and the raw provider object.

        Args:
            exc: The exception to handle.
            context: Human-readable description of where the error occurred.
            partial: Optional partial
                [`Response`][llm_annotator.clients.base.Response] built before
                the error was detected. Its ``text``, ``stop_reason``,
                ``num_output_tokens``, ``full_response`` and ``reasoning``
                fields are preserved in the returned error ``Response``.

        Returns:
            An error [`Response`][llm_annotator.clients.base.Response]. Only
            reached when ``self.on_error`` is ``"ignore"`` or ``"warn"``;
            otherwise
            a [`ProviderError`][llm_annotator.clients.exceptions.ProviderError]
            is raised.

        Raises:
            ProviderError: When ``self.on_error`` is ``"raise"``.
        """
        message = f"{context}: {exc}"
        provider_error = (
            exc if isinstance(exc, ProviderError) else ProviderError(message)
        )
        response_error = (
            message if isinstance(exc, ProviderError) else str(provider_error)
        )

        if self.on_error == "raise":
            raise provider_error from exc

        if self.on_error == "warn":
            self._logger.warning(message)

        return Response(
            text=partial.text if partial is not None else "",
            stop_reason=partial.stop_reason if partial is not None else None,
            model=(partial.model if partial is not None else None)
            or self.model,
            provider=(partial.provider if partial is not None else None)
            or self.provider_type,
            num_output_tokens=(
                partial.num_output_tokens if partial is not None else None
            ),
            full_response=partial.full_response
            if partial is not None
            else None,
            error=response_error,
            error_type=type(provider_error).__name__,
            reasoning=partial.reasoning if partial is not None else None,
        )

    def __enter__(self) -> Self:
        """Enter the context manager, returning this client instance."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Exit the context manager cleanup."""
        self.destroy()

    @abstractmethod
    def _process_response(self, response: Any) -> Response:
        """Process raw provider response into a structured Response object."""
        raise NotImplementedError(
            "Subclasses must implement the _process_response method."
        )

    @abstractmethod
    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        options: T_Options | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> Response:
        """Generate a response from the provider.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            options: Provider-specific generation options.
                NOTE: using this over gen_kwargs is preferred and implemented to facilitate sub-classing
                and satisfying typing and code-hinting.
            gen_kwargs: Additional provider-specific generation kwargs that are not covered by the standard options.
                Has precedence over ``options``.

        Returns:
            A Response object containing the generated response.
        """
        raise NotImplementedError(
            "Subclasses must implement the generate method."
        )

    def _generate_in_threads(
        self,
        *,
        messages: list[list[dict[str, str]]],
        options: T_Options | None,
        gen_kwargs: dict[str, Any] | None,
        max_workers: int | None,
        context: str,
    ) -> list[Response]:
        """Run one [`generate`][llm_annotator.clients.base.Client.generate] call per input.

        The worker count is a local value, so a small batch does not lower the
        concurrency of the batches after it.

        Args:
            messages: One conversation per request.
            options: Provider-specific generation options for every request.
            gen_kwargs: Extra generation kwargs for every request.
            max_workers: Threads to dispatch with. ``None``, ``0`` and ``1``
                give one thread, which runs the requests one after another; a
                higher value is capped at the number of requests.
            context: Start of the error context, completed with the index of
                the request that failed.

        Returns:
            One [`Response`][llm_annotator.clients.base.Response] per input
            conversation, in input order. A request that fails is an error
            ``Response`` and leaves the other requests untouched.

        Raises:
            ProviderError: If a request fails and ``on_error`` is ``"raise"``.
        """
        # A pool needs at least one thread, also for an empty batch.
        workers = max(1, min(max_workers or 1, len(messages)))
        responses: list[Response] = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    self.generate,
                    messages=msgs,
                    options=options,
                    gen_kwargs=gen_kwargs,
                )
                for msgs in messages
            ]
            for idx, future in enumerate(futures):
                try:
                    responses.append(future.result())
                except Exception as exc:
                    responses.append(
                        self._handle_error(
                            exc, context=f"{context} at index {idx}"
                        )
                    )
        return responses

    def batch_generate(
        self,
        *,
        messages: list[list[dict[str, str]]],
        options: T_Options | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> list[Response]:
        """Generate responses for a batch of inputs.

        The default implementation dispatches one
        [`generate`][llm_annotator.clients.base.Client.generate] call per input
        over a thread pool of ``max_workers`` threads. Override this method in
        subclasses that support native batching (e.g. vLLM offline and vLLM
        server) for better throughput.

        Args:
            messages: List of message lists, where each message dict has "role" and "content" keys.
            options: Provider-specific generation options.
            gen_kwargs: Additional provider-specific generation kwargs that are not covered by the standard options.
                Has precedence over ``options``.

        Returns:
            One Response per input, in input order. A request that fails is an
            error Response, unless ``on_error`` is ``"raise"``.

        Raises:
            ProviderError: If a request fails and ``on_error`` is ``"raise"``.
        """
        return self._generate_in_threads(
            messages=messages,
            options=options,
            gen_kwargs=gen_kwargs,
            max_workers=self.max_workers,
            context=f"{self.provider_type.value} request failed",
        )

    def warm_up(
        self,
        *,
        system_message: str | None = None,
        prompt_prefix: str | None = None,
        options: T_Options | None = None,
    ) -> None:
        """Prime the client before the main workload (no-op by default).

        Override in clients that benefit from a warm-up pass (e.g.
        [`VLLMOfflineClient`][llm_annotator.clients.vllm_offline_client.VLLMOfflineClient]
        uses this to prime the KV-cache with a shared prefix before the first
        real batch).

        Args:
            system_message: Optional system message shared across all requests.
            prompt_prefix: Optional fixed prefix that starts every user turn.
            options: Optional generation options used to derive the warm-up params.
        """

    def is_healthy(self, timeout: float = 5.0) -> bool:
        """Check whether the backend can take requests (``True`` by default).

        Override in clients whose backend can be probed.
        [`VLLMQueueAnnotator`][llm_annotator.annotator.VLLMQueueAnnotator]
        calls this after a batch failed entirely, and removes the server from
        its pool when the answer is ``False``.

        Args:
            timeout: Maximum number of seconds to wait for an answer.

        Returns:
            Whether the backend answers.
        """
        return True

    def destroy(self) -> None:
        """Clean up any resources used by the client."""

    @abstractmethod
    def _handle_stop_reason(
        self, *, stop_reason: str | None, num_output_tokens: int | None
    ) -> None:
        raise NotImplementedError(
            "Subclasses must implement the _handle_stop_reason method."
        )


__all__ = [
    "Client",
    "OnError",
    "Provider",
    "ProviderRuntimeOptions",
    "Response",
    "reject_multiple_responses",
]
