"""vLLM offline provider implementation."""

from __future__ import annotations

import gc
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from llm_annotator.clients.base import (
    Client,
    OnError,
    Provider,
    Response,
    reject_multiple_responses,
)
from llm_annotator.clients.exceptions import ProviderError
from llm_annotator.clients.vllm_online_client import VLLMBaseRuntimeOptions


if TYPE_CHECKING:
    from vllm import LLM, RequestOutput
    from vllm.reasoning import ReasoningParser


@dataclass(slots=True, frozen=True)
class VLLMOfflineRuntimeOptions(VLLMBaseRuntimeOptions):
    """Generation options for the vLLM offline client.

    Extends
    [`VLLMBaseRuntimeOptions`][llm_annotator.clients.vllm_online_client.VLLMBaseRuntimeOptions],
    which carries everything both vLLM clients spell the same way
    (``temperature``, ``top_p``, ``top_k``, ``repetition_penalty``,
    ``presence_penalty``, ``frequency_penalty``, ``stop``, ``seed``,
    ``chat_template_kwargs`` and ``extra_body``), with the
    ``SamplingParams`` fields that only in-process inference offers.

    Attributes:
        max_completion_tokens: Maximum number of output tokens. Inherited from
            [`ProviderRuntimeOptions`][llm_annotator.clients.base.ProviderRuntimeOptions].
            Forwarded to ``SamplingParams`` as ``max_tokens``.
        json_schema: Optional JSON schema dict for structured output via guided
            decoding. Inherited from ``ProviderRuntimeOptions``. When
            provided, vLLM constrains generation to valid JSON matching the
            schema.
        whitespace_pattern: Regex pattern inserted between JSON tokens during
            guided decoding. Only used when ``json_schema`` is set.
    """

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
        payload = VLLMBaseRuntimeOptions.to_payload(self)
        if self.max_completion_tokens is not None:
            payload["max_tokens"] = self.max_completion_tokens
        if self.json_schema is not None:
            from vllm.sampling_params import StructuredOutputsParams

            payload["structured_outputs"] = StructuredOutputsParams(
                json=self.json_schema,
                whitespace_pattern=self.whitespace_pattern,
            )
        if self.extra_body:
            payload.update(self.extra_body)
        return payload


class VLLMOfflineClient(Client[VLLMOfflineRuntimeOptions]):
    """Offline vLLM client that runs inference in-process.

    Loads the model into GPU memory on construction and uses vLLM's
    ``LLM.chat`` API for batched generation. Supports structured output
    via JSON schema guided decoding, automatic prefix caching, and chunked
    prefill. Use as a context manager to ensure GPU resources are released
    when done.

    ``batch_generate`` hands every conversation it is given to one
    ``LLM.chat`` call. How many of them run at the same time is vLLM's own
    decision, governed by ``max_num_seqs`` and ``max_num_batched_tokens``
    against the KV cache that ``gpu_memory_utilization`` sized at start-up.

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
        reasoning_parser: Name of the vLLM reasoning parser that splits a
            thinking model's trace from its answer, e.g. ``"qwen3"``.
        extra_vllm_kwargs: Additional keyword arguments forwarded to
            ``vllm.LLM``. Explicit constructor arguments take precedence
            over any conflicting keys here.

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
        ...     max_completion_tokens=128, json_schema=schema
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
        reasoning_parser: str | None = None,
        extra_vllm_kwargs: dict[str, Any] | None = None,
        on_error: OnError = "warn",
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
            reasoning_parser: Name of the vLLM reasoning parser that splits a
                thinking model's trace from its answer, e.g. ``"qwen3"``.
                ``vllm.LLM`` does no such splitting itself, so without it the
                trace stays inline in the generated text.
            extra_vllm_kwargs: Additional keyword arguments forwarded to
                ``vllm.LLM``. Explicit constructor arguments take precedence
                over any conflicting keys here.
            on_error: Error behavior when generation fails.
                Defaults to ``"warn"``.

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
        self._reasoning_parser_name = reasoning_parser
        self._reasoning_parser: ReasoningParser | None = None
        self._extra_vllm_kwargs: dict[str, Any] = extra_vllm_kwargs or {}
        self._pipe: LLM | None = None
        self._pipeline_loaded = False

    def _ensure_pipeline_loaded(self) -> None:
        """Load the vLLM engine and move the weights to GPU on first use.

        Explicit constructor arguments take precedence over any conflicting
        keys in ``extra_vllm_kwargs``. The engine is built lazily so that
        ``enable_prefix_caching`` and ``enable_chunked_prefill`` can be
        adjusted (e.g. by
        [`warm_up`][llm_annotator.clients.vllm_offline_client.VLLMOfflineClient.warm_up])
        before it is constructed.

        Raises:
            ImportError: If vLLM is not installed.
        """
        if self._pipeline_loaded:
            return

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
                ``SamplingParams``; the token budget is forced to 1 for the
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

    def _split_reasoning(
        self,
        text: str,
        prompt_token_ids: Sequence[int] | None = None,
    ) -> tuple[str | None, str]:
        """Separate a reasoning trace from the answer in one generation.

        The served vLLM entrypoint does this itself when started with
        ``--reasoning-parser``; ``vllm.LLM`` does not, so the same parser is
        applied here to give both providers the same two columns. Without a
        configured parser the text is returned untouched, trace and all.

        The prompt decides whether there is a trace to look for at all. A chat
        template asked for ``enable_thinking: False`` closes the thinking block
        itself, so everything the model then writes is answer. Without this
        check the parser would see no closing tag in the output and, by its own
        rule that an unterminated trace runs to the end, hand back the whole
        answer as reasoning and an empty answer.

        Args:
            text: The raw generated text of one sequence.
            prompt_token_ids: Token ids of the prompt that produced it, used to
                tell whether the template already ended the thinking block.

        Returns:
            The trace (``None`` when there is none) and the answer.
        """
        if not self._reasoning_parser_name:
            return None, text.strip()

        if self._reasoning_parser is None:
            from vllm.reasoning import ReasoningParserManager

            self._ensure_pipeline_loaded()
            parser_cls = ReasoningParserManager.get_reasoning_parser(
                self._reasoning_parser_name
            )
            self._reasoning_parser = parser_cls(
                cast("LLM", self._pipe).get_tokenizer()
            )

        if prompt_token_ids and self._reasoning_parser.is_reasoning_end(
            list(prompt_token_ids)
        ):
            return None, text.strip()

        # `request` is part of the parser interface but unused by the
        # non-streaming path, which reads the text alone.
        reasoning, content = self._reasoning_parser.extract_reasoning(
            text,
            None,  # type: ignore[arg-type]
        )
        return (reasoning or None), (content or "").strip()

    def _process_response(self, response: RequestOutput) -> Response:
        """Convert a single vLLM RequestOutput to a structured Response.

        Args:
            response: Raw RequestOutput from vLLM, containing one or more
                generated sequences.

        Returns:
            A Response built from the first generated sequence.
        """
        output = response.outputs[0]
        num_output_tokens = len(output.token_ids) if output.token_ids else None
        finish_reason = output.finish_reason
        reasoning, text = self._split_reasoning(
            output.text or "", getattr(response, "prompt_token_ids", None)
        )

        partial = Response(
            text=text,
            stop_reason=finish_reason,
            model=self.model,
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
                context="vLLM offline response stop reason",
                partial=partial,
            )

        return partial

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
                ``VLLMOfflineRuntimeOptions`` instance to use vLLM-specific
                settings.
            gen_kwargs: Additional provider-specific generation kwargs that are
                not covered by ``options``. Has precedence over ``options``.

        Returns:
            A Response object containing the generated text and metadata. A
            failed call is an error Response when ``on_error`` is ``"warn"`` or
            ``"ignore"``.

        Raises:
            ProviderError: If the vLLM call fails or the stop reason is an
                error condition, and ``on_error`` is ``"raise"``.
        """
        return self.batch_generate(
            messages=[messages],
            options=options,
            gen_kwargs=gen_kwargs,
        )[0]

    def batch_generate(
        self,
        *,
        messages: list[list[dict[str, str]]],
        options: VLLMOfflineRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> list[Response]:
        """Generate responses for a batch of conversations.

        Every conversation goes to one ``LLM.chat`` call, which returns once
        all of them are generated. vLLM decides how many run at the same time.
        Response order matches input order.

        Args:
            messages: List of conversations, where each conversation is a list
                of role/content dicts.
            options: Optional generation configuration. Pass a
                ``VLLMOfflineRuntimeOptions`` instance to use vLLM-specific
                settings such as temperature, top-p, or a JSON schema.
            gen_kwargs: Additional provider-specific generation kwargs that are
                not covered by ``options``. Has precedence over ``options``.

        Returns:
            A list of Response objects, one per input conversation, in the
            same order as the input. A failed call gives one error Response per
            conversation when ``on_error`` is ``"warn"`` or ``"ignore"``.

        Raises:
            ProviderError: If the model is not loaded or the vLLM call fails,
                and ``on_error`` is ``"raise"``.
            ValueError: If the request asks for more than one response.
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
        reject_multiple_responses(payload)
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

    @contextmanager
    def _cleanup_step(self, name: str) -> Iterator[None]:
        """Run one clean-up step and log a failure instead of raising it.

        Args:
            name: Name of the step, used in the warning.

        Yields:
            Control to the body of the step.
        """
        try:
            yield
        except Exception as exc:
            self._logger.warning(f"vLLM clean-up step '{name}' failed: {exc}")

    def destroy(self) -> None:
        """Free GPU memory and clean up all vLLM resources.

        Every step runs even when an earlier one fails, and a failed step is
        logged at warning level with its name and the exception. Safe to call
        multiple times; subsequent calls after the first are no-ops. Also
        invoked automatically when the client is used as a context manager.
        """
        if self._pipe is None:
            return

        pipe = self._pipe
        self._pipe = None

        with self._cleanup_step("destroy model parallel"):
            from vllm.distributed import destroy_model_parallel

            destroy_model_parallel()

        with self._cleanup_step("destroy distributed environment"):
            from vllm.distributed import destroy_distributed_environment

            destroy_distributed_environment()

        with self._cleanup_step("shut down model executor"):
            pipe.llm_engine.model_executor.shutdown()
            del pipe.llm_engine.model_executor

        with self._cleanup_step("shut down engine core"):
            pipe.llm_engine.engine_core.shutdown()
            del pipe.llm_engine.engine_core

        with self._cleanup_step("release the engine"):
            del pipe.llm_engine

        # The last reference to the engine has to go before the allocator can
        # hand its blocks back.
        del pipe

        with self._cleanup_step("free GPU memory"):
            from torch import cuda

            cuda.empty_cache()
            gc.collect()

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
]
