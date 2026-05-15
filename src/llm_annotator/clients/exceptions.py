"""Project-specific exceptions."""


class LLMClientError(Exception):
    """Base exception raised by ``Client`` modules."""


class ConfigurationError(LLMClientError):
    """Raised when runtime configuration is invalid."""


class ProviderError(LLMClientError):
    """Raised when a provider call fails."""


class ParsingError(LLMClientError):
    """Raised when model output cannot be parsed."""
