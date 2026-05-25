"""vLLM offline provider implementation."""

from __future__ import annotations

import functools
import gc
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from llm_annotator.clients.base import (
    Client,
    OnError,
    Provider,
    Response,
)
from llm_annotator.clients.exceptions import ProviderError
from llm_annotator.clients.vllm_client import VLLMBaseRuntimeOptions


if TYPE_CHECKING:
    from vllm import LLM, RequestOutput


def _is_oom_error(exc: BaseException) -> bool:
    """Return True if *exc* or any exception in its chain looks like a CUDA OOM."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in {
            "OutOfMemoryError",
            "CudaOutOfMemoryError",
        }:
            return True
        if "out of memory" in str(current).lower():
            return True
        current = current.__cause__ or current.__context__
    return False


def auto_reduce_batch_size(
    method: Callable[..., list[Response]],
) -> Callable[..., list[Response]]:
    """Decorate a ``batch_generate`` method to retry with halved chunk size on OOM.

    Intended for use with :class:`VLLMOfflineClient`. On each call the full
    ``messages`` list is split into chunks and dispatched one at a time. When a
    CUDA out-of-memory error is detected the current chunk size is halved and
    the failing chunk is retried at the new size. This continues until the chunk
    succeeds or the size would fall below the instance's ``_min_batch_size``,
    at which point the error is re-raised.

    The chunk size and minimum are read from the instance's ``_batch_size`` and
    ``_min_batch_size`` attributes on every call, so they can be adjusted after
    construction.

    Args:
        method: Unbound ``batch_generate`` method to wrap.

    Returns:
        The wrapped method with adaptive OOM-recovery logic applied.
    """

    @functools.wraps(method)
    def wrapper(
        self: VLLMOfflineClient,
        *,
        messages: list[list[dict[str, str]]],
        **kwargs: Any,
    ) -> list[Response]:
        batch_size = (
            self._batch_size if self._batch_size is not None else len(messages)
        )
        min_batch_size = max(self._min_batch_size, 1)
        batch_size = max(batch_size, min_batch_size)

        results: list[Response] = []
        i = 0

        while i < len(messages):
            chunk = messages[i : i + batch_size]
            try:
                chunk_results = method(self, messages=chunk, **kwargs)
                results.extend(chunk_results)
                i += len(chunk)
            except Exception as exc:
                if not _is_oom_error(exc):
                    raise
                new_size = batch_size // 2
                if new_size < min_batch_size:
                    raise
                self._logger.warning(
                    "CUDA out-of-memory with batch_size=%d;"
                    " retrying with batch_size=%d.",
                    batch_size,
                    new_size,
                )
                batch_size = new_size

        return results

    return wrapper


@dataclass(slots=True, frozen=True)
class VLLMOfflineRuntimeOptions(VLLMBaseRuntimeOptions):
    """Generation options for the vLLM offline client.

    Extends :class:`VLLMBaseRuntimeOptions` (which provides ``top_k``,
    ``repetition_penalty``, and ``chat_template_kwargs``) with
    ``SamplingParams``-compatible fields for in-process vLLM inference.

    Attributes:
        max_tokens: Maximum number of output tokens. Inherited from
            :class:`~llm_annotator.clients.base.ProviderRuntimeOptions`.
        json_schema: Optional JSON schema dict for structured output via guided
            decoding. Inherited from
            :class:`~llm_annotator.clients.base.ProviderRuntimeOptions`. When
            provided, vLLM constrains generation to valid JSON matching the
            schema.
        top_k: Top-k sampling cutoff. Inherited from
            :class:`VLLMBaseRuntimeOptions`. ``None`` uses the model default.
        repetition_penalty: Multiplicative penalty for token repetition.
            Inherited from :class:`VLLMBaseRuntimeOptions`.
        chat_template_kwargs: Additional kwargs forwarded to the chat template.
            Inherited from :class:`VLLMBaseRuntimeOptions`. Pass
            ``{"enable_thinking": True}`` here to enable thinking mode.
        temperature: Sampling temperature. ``None`` uses the model default.
        top_p: Top-p nucleus sampling probability. ``None`` uses the model
            default.
        stop: Optional list of strings that halt generation when produced.
        presence_penalty: Penalty applied to tokens already present in the
            output. Defaults to ``0.0`` (vLLM default).
        frequency_penalty: Penalty applied proportional to token frequency in
            the output. Defaults to ``0.0`` (vLLM default).
        seed: Optional fixed random seed for reproducible generation.
        n: Number of independent output sequences to generate per request.
            Defaults to ``1``.
        whitespace_pattern: Regex pattern inserted between JSON tokens during
            guided decoding. Only used when ``json_schema`` is set.
    """

    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None
    n: int = 1
    whitespace_pattern: str | None = r"[ ]?"

    def to_payload(self) -> dict[str, Any]:
        """Build a ``SamplingParams``-compatible payload dict.

        The dict can be passed directly to ``vllm.SamplingParams(**payload)``.
        ``chat_template_kwargs`` is intentionally excluded; it must be passed
        separately to ``LLM.chat()``.

        Returns:
            A dict of ``SamplingParams``-compatible keyword arguments.

        Raises:
            ImportError: If vLLM is not installed.
        """
        payload: dict[str, Any] = {}
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.repetition_penalty is not None:
            payload["repetition_penalty"] = self.repetition_penalty

        payload["n"] = self.n
        payload["presence_penalty"] = self.presence_penalty
        payload["frequency_penalty"] = self.frequency_penalty
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.stop is not None:
            payload["stop"] = self.stop
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.json_schema is not None:
            from vllm.sampling_params import StructuredOutputsParams

            payload["structured_outputs"] = StructuredOutputsParams(
                json=self.json_schema,
                whitespace_pattern=self.whitespace_pattern,
            )
        return payload


class VLLMOfflineClient(Client[VLLMOfflineRuntimeOptions]):
    """Offline vLLM client that runs inference in-process.

    Loads the model into GPU memory on construction and uses vLLM's
    ``LLM.chat`` API for batched generation. Supports structured output
    via JSON schema guided decoding, automatic prefix caching, and chunked
    prefill. Use as a context manager to ensure GPU resources are released
    when done.

    ``batch_generate`` automatically splits the message list into chunks of
    ``batch_size`` and retries failing chunks with a halved size on CUDA
    out-of-memory errors (see :func:`auto_reduce_batch_size`). When
    ``batch_size`` is ``None`` (the default) all messages are sent in a
    single vLLM call, mirroring the original behaviour while still
    recovering from OOM when possible.

    Args:
        model: Hugging Face model identifier or local path.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        max_num_seqs: Maximum number of sequences processed in parallel.
        gpu_memory_utilization: Target fraction of GPU memory to use.
        enforce_eager: Disable CUDA graphs and run in eager mode.
        quantization: Quantization method (e.g. ``"fp8"``, ``"awq"``).
        max_model_len: Maximum total sequence length the model supports.
        max_num_batched_tokens: Maximum tokens per forward pass.
        enable_prefix_caching: Enable automatic KV-cache prefix reuse.
        enable_chunked_prefill: Process prefills in chunks to bound memory.
        language_model_only: If True, all non-text modalities are disabled,
            saving some memory.
        speculative_config: Optional dict of vLLM speculative decoding config parameters.
        extra_vllm_kwargs: Additional keyword arguments forwarded to
            ``vllm.LLM``. Explicit constructor arguments take precedence
            over any conflicting keys here.
        batch_size: Starting chunk size for :meth:`batch_generate`. Defaults
            to ``None``, which sends all messages in one call. On OOM the
            chunk size is halved automatically until it succeeds or falls
            below ``min_batch_size``.
        min_batch_size: Smallest permitted chunk size before an OOM error is
            re-raised. Must be >= 1.

    Examples:
        Basic generation:

        >>> client = VLLMOfflineClient(  # doctest: +SKIP
        ...     model="meta-llama/Llama-3.2-3B-Instruct",
        ...     max_model_len=4096,
        ... )
        >>> response = client.generate(  # doctest: +SKIP
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>> client.destroy()  # doctest: +SKIP

        Context manager (recommended):

        >>> with VLLMOfflineClient(  # doctest: +SKIP
        ...     model="meta-llama/Llama-3.2-3B-Instruct",
        ...     max_model_len=4096,
        ... ) as client:
        ...     responses = client.batch_generate(
        ...         messages=[
        ...             [{"role": "user", "content": "Hello!"}],
        ...             [{"role": "user", "content": "What is 2+2?"}],
        ...         ]
        ...     )

        Structured output with JSON schema:

        >>> schema = {  # doctest: +SKIP
        ...     "type": "object",
        ...     "properties": {"label": {"type": "string"}},
        ...     "required": ["label"],
        ... }
        >>> opts = VLLMOfflineRuntimeOptions(
        ...     max_tokens=128, json_schema=schema
        ... )  # doctest: +SKIP
        >>> with VLLMOfflineClient(  # doctest: +SKIP
        ...     model="meta-llama/Llama-3.2-3B-Instruct"
        ... ) as client:
        ...     responses = client.batch_generate(
        ...         messages=[
        ...             [{"role": "user", "content": "Classify: great"}]
        ...         ],
        ...         options=opts,
        ...     )
    """

    provider_type = Provider.VLLM_OFFLINE

    def __init__(
        self,
        model: str,
        *,
        tensor_parallel_size: int = 1,
        max_num_seqs: int = 256,
        gpu_memory_utilization: float = 0.90,
        enforce_eager: bool = False,
        quantization: str | None = None,
        max_model_len: int | None = None,
        max_num_batched_tokens: int | None = None,
        enable_prefix_caching: bool = True,
        enable_chunked_prefill: bool = True,
        language_model_only: bool = True,
        speculative_config: dict[str, Any] | None = None,
        extra_vllm_kwargs: dict[str, Any] | None = None,
        on_error: OnError = "raise",
        batch_size: int | None = None,
        min_batch_size: int = 1,
    ) -> None:
        """Initialize the offline vLLM client and load the model into memory.

        Args:
            model: Hugging Face model identifier or local path.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            max_num_seqs: Maximum number of sequences processed in parallel.
            gpu_memory_utilization: Target fraction of GPU memory to use.
            enforce_eager: Disable CUDA graphs and run in eager mode.
            quantization: Quantization method (e.g. ``"fp8"``, ``"awq"``).
            max_model_len: Maximum total sequence length the model supports.
            max_num_batched_tokens: Maximum tokens per forward pass.
            enable_prefix_caching: Enable automatic KV-cache prefix reuse.
                Particularly beneficial when many prompts share a common prefix
                (e.g. a system message), since the shared prefix is only encoded once.
            enable_chunked_prefill: Process prefills in chunks to reduce
                peak memory usage and improve scheduling efficiency.
            language_model_only: If ``True``, all non-text modalities are
                disabled, saving some memory. Defaults to ``True``.
            speculative_config: Optional dict of vLLM speculative decoding
                config parameters.
            extra_vllm_kwargs: Additional keyword arguments forwarded to
                ``vllm.LLM``. Explicit constructor arguments take precedence
                over any conflicting keys here.
            on_error: Error behavior when generation fails.
            batch_size: Starting chunk size for :meth:`batch_generate`. When
                ``None`` (the default) all messages are sent in one call. On
                OOM the chunk size is halved until the call succeeds or falls
                below ``min_batch_size``.
            min_batch_size: Smallest permitted chunk size before an OOM is
                re-raised. Must be >= 1.

        Raises:
            ImportError: If vLLM is not installed (raised on first use).
        """
        super().__init__(model=model, on_error=on_error)
        self._tensor_parallel_size = tensor_parallel_size
        self._max_num_seqs = max_num_seqs
        self._gpu_memory_utilization = gpu_memory_utilization
        self._enforce_eager = enforce_eager
        self._quantization = quantization
        self._max_model_len = max_model_len
        self._max_num_batched_tokens = max_num_batched_tokens
        self._enable_prefix_caching = enable_prefix_caching
        self._enable_chunked_prefill = enable_chunked_prefill
        self._language_model_only = language_model_only
        self._speculative_config = speculative_config or {}
        self._extra_vllm_kwargs: dict[str, Any] = extra_vllm_kwargs or {}
        self._batch_size = batch_size
        self._min_batch_size = min_batch_size
        self._pipe: LLM | None = None
        self._pipeline_loaded = False

    def _ensure_pipeline_loaded(self) -> None:
        """Load the pipeline on first use if it has not been loaded yet."""
        if not self._pipeline_loaded:
            self._load_pipeline()

    def _load_pipeline(self) -> None:
        """Load the vLLM LLM engine and move weights to GPU.

        Explicit constructor arguments take precedence over any conflicting
        keys in ``extra_vllm_kwargs``. Called lazily on first use so that
        ``enable_prefix_caching`` and ``enable_chunked_prefill`` can be
        adjusted (e.g. by :meth:`warm_up`) before the engine is constructed.

        Raises:
            ImportError: If vLLM is not installed.
        """
        from vllm import LLM

        # Start from caller-supplied extras, then overwrite with explicit args
        # so that explicit args always win on conflict.
        kwargs: dict[str, Any] = self._extra_vllm_kwargs.copy()
        explicit: dict[str, Any] = {
            "model": self.model,
            "tensor_parallel_size": self._tensor_parallel_size,
            "max_num_seqs": self._max_num_seqs,
            "gpu_memory_utilization": self._gpu_memory_utilization,
            "enforce_eager": self._enforce_eager,
            "enable_prefix_caching": self._enable_prefix_caching,
            "enable_chunked_prefill": self._enable_chunked_prefill,
            "language_model_only": self._language_model_only,
        }
        if self._speculative_config:
            explicit["speculative_config"] = self._speculative_config
        if self._quantization is not None:
            explicit["quantization"] = self._quantization
        if self._max_model_len is not None:
            explicit["max_model_len"] = self._max_model_len
        if self._max_num_batched_tokens is not None:
            explicit["max_num_batched_tokens"] = self._max_num_batched_tokens

        kwargs.update(explicit)
        self._pipe = LLM(**kwargs)
        self._pipeline_loaded = True

    def warm_up(
        self,
        *,
        system_message: str | None = None,
        prompt_prefix: str | None = None,
        options: VLLMOfflineRuntimeOptions | None = None,
    ) -> None:
        """Prime the KV-cache with a shared prefix before the main workload.

        When many prompts share a common system message or prompt prefix,
        running a single cheap forward pass first ensures the shared tokens
        are cached before the first real batch, avoiding a cold-start latency
        spike on the initial batch.

        This is a no-op if neither ``system_message`` nor ``prompt_prefix``
        is provided, or if the model has not been loaded yet.

        Args:
            system_message: Optional system message used in every request.
            prompt_prefix: Optional fixed prefix that starts every user turn.
            options: Optional generation options. Only used to derive a base
                ``SamplingParams``; ``max_tokens`` is forced to 1 for the
                warm-up run regardless of the value set here.

        Raises:
            ProviderError: If the warm-up inference call fails.
        """
        if not system_message and not prompt_prefix:
            return

        # Auto-enable KV-cache prefix reuse and chunked prefill before the
        # engine is constructed so that the shared prefix tokens are only
        # encoded once across the full annotation workload.
        if prompt_prefix is not None:
            self._enable_prefix_caching = True
            self._enable_chunked_prefill = True

        self._ensure_pipeline_loaded()

        if self._pipe is None:
            return

        from vllm import SamplingParams

        messages: list[dict[str, str]] = []
        if system_message is not None:
            messages.append({"role": "system", "content": system_message})
        if prompt_prefix is not None:
            messages.append({"role": "user", "content": prompt_prefix})

        chat_template_kwargs = (
            options.chat_template_kwargs.copy()
            if options and options.chat_template_kwargs
            else {}
        )

        resolved_options = options or VLLMOfflineRuntimeOptions()
        payload = resolved_options.to_payload()
        payload["max_tokens"] = 1  # Force minimal output for the warm-up pass
        sampling_params = SamplingParams(**payload)

        try:
            self._pipe.chat(
                cast(Any, [messages]),
                sampling_params,
                chat_template_kwargs=chat_template_kwargs,
                use_tqdm=False,
            )
        except Exception as exc:
            self._handle_error(exc, context="vLLM offline warm-up failed")

    def _process_response(self, response: RequestOutput) -> Response:
        """Convert a single vLLM RequestOutput to a structured Response.

        Args:
            response: Raw RequestOutput from vLLM, containing one or more
                generated sequences.

        Returns:
            A Response built from the first generated sequence.

        Raises:
            ProviderError: If the stop reason indicates an error condition.
        """
        output = response.outputs[0]
        num_output_tokens = len(output.token_ids) if output.token_ids else None
        finish_reason = output.finish_reason

        self._handle_stop_reason(
            stop_reason=finish_reason,
            num_output_tokens=num_output_tokens,
        )

        return Response(
            text=output.text.strip() if output.text else "",
            stop_reason=finish_reason,
            model=self.model,
            provider=self.provider_type,
            num_output_tokens=num_output_tokens,
            full_response=response,
        )

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        options: VLLMOfflineRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> Response:
        """Generate a single response for a conversation.

        Delegates to batch_generate with a single-item batch.

        Args:
            messages: Conversation as a list of role/content dicts.
            options: Optional generation configuration. Pass a
                VLLMRuntimeOptions instance to use vLLM-specific settings.
            gen_kwargs: Additional provider-specific generation kwargs that are
                not covered by ``options``. Has precedence over ``options``.

        Returns:
            A Response object containing the generated text and metadata.

        Raises:
            ProviderError: If the vLLM call fails or the stop reason is
                an error condition.
        """
        return self.batch_generate(
            messages=[messages],
            options=options,
            gen_kwargs=gen_kwargs,
        )[0]

    @auto_reduce_batch_size
    def batch_generate(
        self,
        *,
        messages: list[list[dict[str, str]]],
        options: VLLMOfflineRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> list[Response]:
        """Generate responses for a batch of conversations.

        The full ``messages`` list is automatically split into chunks and each
        chunk is dispatched to vLLM separately. On a CUDA out-of-memory error
        the chunk size is halved and retried. Chunk size and minimum are
        configured via the ``batch_size`` and ``min_batch_size`` constructor
        arguments. Response order matches input order.

        Args:
            messages: List of conversations, where each conversation is a list
                of role/content dicts.
            options: Optional generation configuration. Pass a
                VLLMRuntimeOptions instance to use vLLM-specific settings
                such as temperature, top-p, or a JSON schema.
            gen_kwargs: Additional provider-specific generation kwargs that are
                not covered by ``options``. Has precedence over ``options``.

        Returns:
            A list of Response objects, one per input conversation, in the
            same order as the input.

        Raises:
            ProviderError: If the model is not loaded or the vLLM call fails.
        """
        self._ensure_pipeline_loaded()
        if self._pipe is None:
            error_response = self._handle_error(
                ProviderError(
                    "vLLM model is not loaded. The model may have been destroyed."
                ),
                context="vLLM offline batch generation failed",
            )
            return [error_response for _ in messages]

        chat_template_kwargs = (
            options.chat_template_kwargs.copy()
            if options and options.chat_template_kwargs
            else {}
        )

        from vllm import SamplingParams

        resolved = options or VLLMOfflineRuntimeOptions()
        payload = resolved.to_payload()
        payload.update(gen_kwargs or {})
        try:
            sampling_params = SamplingParams(**payload)
        except TypeError as exc:
            err = self._handle_error(
                exc, context="vLLM offline invalid sampling parameters"
            )
            return [err for _ in messages]

        try:
            outputs = self._pipe.chat(
                cast(Any, messages),
                sampling_params,
                chat_template_kwargs=chat_template_kwargs,
                use_tqdm=False,
            )
        except Exception as exc:
            error_response = self._handle_error(
                exc, context="vLLM offline batch generation failed"
            )
            return [error_response for _ in messages]

        responses: list[Response] = []
        for idx, output in enumerate(outputs):
            try:
                responses.append(self._process_response(output))
            except Exception as exc:
                responses.append(
                    self._handle_error(
                        exc,
                        context=f"vLLM offline response processing failed at index {idx}",
                    )
                )

        if len(responses) < len(messages):
            padding = len(messages) - len(responses)
            err = self._handle_error(
                ProviderError(
                    "vLLM offline returned fewer outputs than requested."
                ),
                context="vLLM offline batch response validation failed",
            )
            responses.extend([err for _ in range(padding)])

        return responses

    def destroy(self) -> None:
        """Free GPU memory and clean up all vLLM resources.

        Safe to call multiple times; subsequent calls after the first are
        no-ops. Also invoked automatically when the client is used as a
        context manager.
        """
        if self._pipe is None:
            return

        try:
            from torch import cuda
            from vllm.distributed import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()

            try:
                self._pipe.llm_engine.model_executor.shutdown()
                del self._pipe.llm_engine.model_executor
            except Exception:
                pass
            try:
                self._pipe.llm_engine.engine_core.shutdown()
                del self._pipe.llm_engine.engine_core
            except Exception:
                pass

            del self._pipe.llm_engine
            del self._pipe
            cuda.empty_cache()
            gc.collect()
        except Exception:
            pass
        finally:
            self._pipe = None

    def _handle_stop_reason(
        self, *, stop_reason: str | None, num_output_tokens: int | None
    ) -> None:
        """Raise ProviderError for non-success vLLM stop reasons.

        Args:
            stop_reason: The finish reason string from vLLM.
            num_output_tokens: Number of output tokens generated.

        Raises:
            ProviderError: If the stop reason indicates truncation or abort.
        """
        if stop_reason in {None, "stop"}:
            return

        token_suffix = (
            ""
            if num_output_tokens is None
            else f" (output tokens: {num_output_tokens:,})"
        )

        if stop_reason == "length":
            raise ProviderError(
                f"vLLM stopped because it hit the configured output token"
                f" limit{token_suffix}."
            )
        if stop_reason == "abort":
            raise ProviderError(
                f"vLLM aborted the request before completing the"
                f" response{token_suffix}."
            )
        raise ProviderError(
            f"vLLM stopped with unexpected reason '{stop_reason}'{token_suffix}."
        )


__all__ = [
    "VLLMOfflineClient",
    "VLLMOfflineRuntimeOptions",
    "auto_reduce_batch_size",
]
