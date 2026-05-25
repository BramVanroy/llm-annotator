"""Abstract interface for LLM provider clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, ClassVar, Generic, Literal, Self, TypeVar

from llm_annotator.clients.exceptions import ProviderError
from llm_annotator.logging_utils import get_logger


# So that sub-classes can extend their runtime options without breaking typing
T_Options = TypeVar("T_Options", bound="ProviderRuntimeOptions")
OnError = Literal["raise", "ignore", "warn"]


class Provider(StrEnum):
    OPENAI = auto()
    CLAUDE = auto()
    VLLM = auto()
    VLLM_OFFLINE = auto()


@dataclass(slots=True, frozen=True)
class ProviderRuntimeOptions:
    """Shared generation options for provider calls; can be subclassed and extended.

    Attributes:
        max_tokens: Optional maximum output token count.
        json_schema: Optional JSON schema dict for structured output. When provided,
            clients that support guided decoding (e.g. vLLM) will constrain generation
            to valid JSON matching the schema. Other clients will use the schema for
            post-processing / parsing only.
    """

    max_tokens: int | None = None
    json_schema: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        """Convert options to a provider-specific API request payload dict.

        Subclasses override this to build the exact kwargs expected by their SDK.
        The default implementation returns an empty dict.

        Returns:
            A dict of provider-specific request parameters.
        """
        return {}


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


class Client(ABC, Generic[T_Options]):
    """Base client interface used by all provider adapters."""

    provider_type: ClassVar[Provider]

    def __init__(
        self,
        model: str,
        max_workers: int = 4,
        on_error: OnError = "warn",
    ) -> None:
        """Initialize a provider client.

        Args:
            model: Provider-specific model name.
            max_workers: Maximum number of concurrent worker threads for ``batch_generate``. Clients that support native batching may ignore this parameter.
            on_error: Error behavior for provider failures.
                - ``"raise"``: raise a :class:`ProviderError` (default).
                - ``"ignore"``: return a :class:`Response` with ``error`` set.
                - ``"warn"``: log a warning and return an error :class:`Response`.
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
        :class:`Response` so callers retain the generated text, stop
        reason, token counts, and the raw provider object.

        Args:
            exc: The exception to handle.
            context: Human-readable description of where the error occurred.
            partial: Optional partial :class:`Response` built before the
                error was detected. Its ``text``, ``stop_reason``,
                ``num_output_tokens``, and ``full_response`` fields are
                preserved in the returned error :class:`Response`.

        Returns:
            An error :class:`Response`. Only reached when
            ``self.on_error`` is ``"ignore"`` or ``"warn"``; otherwise
            a :class:`~llm_annotator.clients.exceptions.ProviderError`
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

    def batch_generate(
        self,
        *,
        messages: list[list[dict[str, str]]],
        options: T_Options | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> list[Response]:
        """Generate responses for a batch of inputs.

        The default implementation calls :meth:`generate` sequentially. Override
        this method in subclasses that support native batching (e.g. vLLM offline
        and vLLM server) for better throughput.

        Args:
            messages: List of message lists, where each message dict has "role" and "content" keys.
            options: Provider-specific generation options.
            gen_kwargs: Additional provider-specific generation kwargs that are not covered by the standard options.
                Has precedence over ``options``.

        Returns:
            A list of Response objects containing the generated responses.
        """
        return [
            self.generate(
                messages=msgs,
                options=options,
                gen_kwargs=gen_kwargs,
            )
            for msgs in messages
        ]

    def warm_up(
        self,
        *,
        system_message: str | None = None,
        prompt_prefix: str | None = None,
        options: T_Options | None = None,
    ) -> None:
        """Prime the client before the main workload (no-op by default).

        Override in clients that benefit from a warm-up pass (e.g.
        :class:`~llm_annotator.clients.VLLMOfflineClient` uses this to prime
        the KV-cache with a shared prefix before the first real batch).

        Args:
            system_message: Optional system message shared across all requests.
            prompt_prefix: Optional fixed prefix that starts every user turn.
            options: Optional generation options used to derive the warm-up params.
        """

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
]
