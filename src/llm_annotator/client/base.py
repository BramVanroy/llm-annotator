"""Abstract interface for LLM provider clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar


T_Response = TypeVar("T_Response")


class Provider(StrEnum):
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"


@dataclass(slots=True, frozen=True)
class ProviderRuntimeOptions:
    """Shared generation options for provider calls; can be subclassed and extended.

    Attributes:
        temperature: Sampling temperature.
        max_tokens: Optional maximum output token count.
    """

    max_tokens: int | None = None


@dataclass(slots=True, frozen=True)
class Response:
    """Structured response object returned by provider clients."""

    text: str
    stop_reason: str | None = None
    model: str | None = None
    provider: Provider | None = None
    num_output_tokens: int | None = None


class Client(ABC, Generic[T_Response]):
    """Base client interface used by all provider adapters."""

    def __init__(self, model: str) -> None:
        """Initialize a provider client.

        Args:
            model: Provider-specific model name.
        """
        self.model = model

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
        options: ProviderRuntimeOptions | None = None,
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

    @abstractmethod
    def batch_generate(
        self,
        *,
        messages: list[list[dict[str, str]]],
        options: ProviderRuntimeOptions | None = None,
    ) -> list[Response]:
        """Generate responses for a batch of inputs.

        Args:
            messages: List of message lists, where each message dict has "role" and "content" keys.
            **kwargs: Provider-specific generation options.

        Returns:
            A list of Response objects containing the generated responses.
        """
        raise NotImplementedError(
            "Subclasses must implement the batch_generate method."
        )

    def destroy(self) -> None:
        """Clean up any resources used by the client."""

    @abstractmethod
    def _handle_stop_reason(
        self, *, stop_reason: str | None, num_output_tokens: int | None
    ):
        raise NotImplementedError(
            "Subclasses must implement the _handle_stop_reason method."
        )
