"""Abstract interface for LLM provider clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, ClassVar, Generic, Self, TypeVar


T_Response = TypeVar("T_Response")
T_Options = TypeVar("T_Options", bound="ProviderRuntimeOptions")


class Provider(StrEnum):
    OPENAI = auto()
    CLAUDE = auto()
    GEMINI = auto()
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


@dataclass(slots=True, frozen=True)
class Response:
    """Structured response object returned by provider clients."""

    text: str
    stop_reason: str | None = None
    model: str | None = None
    provider: Provider | None = None
    num_output_tokens: int | None = None


class Client(ABC, Generic[T_Response, T_Options]):
    """Base client interface used by all provider adapters."""

    provider_type: ClassVar[Provider]

    def __init__(self, model: str, max_workers: int = 4) -> None:
        """Initialize a provider client.

        Args:
            model: Provider-specific model name.
            max_workers: Maximum number of concurrent worker threads for ``batch_generate``. Clients that support native batching may ignore this parameter.
        """
        self.model = model
        self.max_workers = max_workers

    def __enter__(self) -> Self:
        """Enter the context manager, returning this client instance."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Exit the context manager cleanup."""
        self.destroy()

    @abstractmethod
    def _process_response(self, response: T_Response) -> Response:
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
    ) -> Response:
        """Generate a response from the provider.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            **kwargs: Provider-specific generation options.

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
    ) -> list[Response]:
        """Generate responses for a batch of inputs.

        The default implementation calls :meth:`generate` sequentially. Override
        this method in subclasses that support native batching (e.g. vLLM offline
        and vLLM server) for better throughput.

        Args:
            messages: List of message lists, where each message dict has "role" and "content" keys.
            options: Provider-specific generation options.

        Returns:
            A list of Response objects containing the generated responses.
        """
        return [
            self.generate(messages=msgs, options=options) for msgs in messages
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
