"""Project-specific exceptions."""


class LLMClientError(Exception):
    """Base exception raised by ``Client`` modules."""


class ProviderError(LLMClientError):
    """Raised when a provider call fails."""


class TooManyConsecutiveFailedBatchesError(LLMClientError):
    """Raised by an ``Annotator`` when too many batches fail in a row.

    Unlike the other exceptions in this module, this is raised by
    orchestration code in ``annotator.py`` rather than by a ``Client``
    itself, as a circuit breaker: it stops a run from burning compute
    against a provider that is silently failing every request (e.g. a
    vLLM server that died mid-run) instead of continuing until the job's
    time limit is hit.
    """


__all__ = [
    "LLMClientError",
    "ProviderError",
    "TooManyConsecutiveFailedBatchesError",
]
