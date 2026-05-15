"""vLLM offline provider implementation."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llm_annotator.clients.base import (
    Client,
    Provider,
    ProviderRuntimeOptions,
    Response,
)
from llm_annotator.clients.exceptions import ProviderError


if TYPE_CHECKING:
    from vllm import LLM, RequestOutput


@dataclass(slots=True, frozen=True)
class VLLMRuntimeOptions(ProviderRuntimeOptions):
    """vLLM-specific generation options extending ProviderRuntimeOptions.

    Attributes:
        max_tokens: Maximum number of output tokens. Inherited from ProviderRuntimeOptions.
        json_schema: Optional JSON schema dict for structured output via guided
            decoding. Inherited from ProviderRuntimeOptions. When provided, vLLM
            constrains generation to valid JSON matching the schema.
        temperature: Sampling temperature. None uses the model default.
        top_p: Top-p nucleus sampling probability. None uses the model default.
        top_k: Top-k sampling cutoff. None uses the model default.
        stop: Optional list of strings that halt generation when produced.
        presence_penalty: Penalty applied to tokens already present in the output.
        frequency_penalty: Penalty applied proportional to token frequency in output.
        repetition_penalty: Multiplicative penalty for token repetition.
        seed: Optional fixed random seed for reproducible generation.
        n: Number of independent output sequences to generate per request.
        whitespace_pattern: Regex pattern inserted between JSON tokens during
            guided decoding. Only used when json_schema is set.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop: list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    n: int = 1
    whitespace_pattern: str | None = r"[ ]?"

    def to_sampling_params(self) -> Any:
        """Convert these options to a vLLM SamplingParams instance.

        Returns:
            A configured SamplingParams object ready for use with the vLLM LLM.

        Raises:
            ImportError: If vLLM is not installed.
        """
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        structured_outputs = None
        if self.json_schema is not None:
            structured_outputs = StructuredOutputsParams(
                json=self.json_schema,
                whitespace_pattern=self.whitespace_pattern,
            )

        kwargs: dict[str, Any] = {
            "n": self.n,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.stop is not None:
            kwargs["stop"] = self.stop
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if structured_outputs is not None:
            kwargs["structured_outputs"] = structured_outputs

        return SamplingParams(**kwargs)


class VLLMOfflineClient(Client["RequestOutput", VLLMRuntimeOptions]):
    """Offline vLLM client that runs inference in-process.

    Loads the model into GPU memory on construction and uses vLLM's
    ``LLM.chat`` API for batched generation. Supports structured output
    via JSON schema guided decoding, automatic prefix caching, and chunked
    prefill. Use as a context manager to ensure GPU resources are released
    when done.

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
        >>> opts = VLLMRuntimeOptions(max_tokens=128, json_schema=schema)
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
        gpu_memory_utilization: float = 0.95,
        enforce_eager: bool = False,
        quantization: str | None = None,
        max_model_len: int | None = None,
        max_num_batched_tokens: int | None = None,
        enable_prefix_caching: bool = True,
        enable_chunked_prefill: bool = True,
        extra_vllm_kwargs: dict[str, Any] | None = None,
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
            extra_vllm_kwargs: Additional keyword arguments forwarded to
                ``vllm.LLM``. Explicit constructor arguments take precedence
                over any conflicting keys here.

        Raises:
            ImportError: If vLLM is not installed.
        """
        super().__init__(model=model)
        self._tensor_parallel_size = tensor_parallel_size
        self._max_num_seqs = max_num_seqs
        self._gpu_memory_utilization = gpu_memory_utilization
        self._enforce_eager = enforce_eager
        self._quantization = quantization
        self._max_model_len = max_model_len
        self._max_num_batched_tokens = max_num_batched_tokens
        self._enable_prefix_caching = enable_prefix_caching
        self._enable_chunked_prefill = enable_chunked_prefill
        self._extra_vllm_kwargs: dict[str, Any] = extra_vllm_kwargs or {}
        self._pipe: LLM | None = None
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        """Load the vLLM LLM engine and move weights to GPU.

        Explicit constructor arguments take precedence over any conflicting
        keys in ``extra_vllm_kwargs``.

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
        }
        if self._quantization is not None:
            explicit["quantization"] = self._quantization
        if self._max_model_len is not None:
            explicit["max_model_len"] = self._max_model_len
        if self._max_num_batched_tokens is not None:
            explicit["max_num_batched_tokens"] = self._max_num_batched_tokens

        kwargs.update(explicit)
        self._pipe = LLM(**kwargs)

    def warm_up(
        self,
        *,
        system_message: str | None = None,
        prompt_prefix: str | None = None,
        options: VLLMRuntimeOptions | None = None,
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
        if self._pipe is None or (not system_message and not prompt_prefix):
            return

        from vllm import SamplingParams

        messages: list[dict[str, str]] = []
        if system_message is not None:
            messages.append({"role": "system", "content": system_message})
        if prompt_prefix is not None:
            messages.append({"role": "user", "content": prompt_prefix})

        if options is not None:
            sampling_params = options.to_sampling_params()
            # Force minimal output for the cache warm-up pass
            sampling_params.max_tokens = 1
        else:
            sampling_params = SamplingParams(max_tokens=1)

        try:
            self._pipe.chat([messages], sampling_params, use_tqdm=False)
        except Exception as exc:
            raise ProviderError(f"vLLM offline warm-up failed: {exc}") from exc

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
        )

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        options: VLLMRuntimeOptions | None = None,
    ) -> Response:
        """Generate a single response for a conversation.

        Delegates to batch_generate with a single-item batch.

        Args:
            messages: Conversation as a list of role/content dicts.
            options: Optional generation configuration. Pass a
                VLLMRuntimeOptions instance to use vLLM-specific settings.

        Returns:
            A Response object containing the generated text and metadata.

        Raises:
            ProviderError: If the vLLM call fails or the stop reason is
                an error condition.
        """
        return self.batch_generate(messages=[messages], options=options)[0]

    def batch_generate(
        self,
        *,
        messages: list[list[dict[str, str]]],
        options: VLLMRuntimeOptions | None = None,
    ) -> list[Response]:
        """Generate responses for a batch of conversations.

        The batch is dispatched to the vLLM engine in a single call. Response
        order matches input order.

        Args:
            messages: List of conversations, where each conversation is a list
                of role/content dicts.
            options: Optional generation configuration. Pass a
                VLLMRuntimeOptions instance to use vLLM-specific settings
                such as temperature, top-p, or a JSON schema.

        Returns:
            A list of Response objects, one per input conversation, in the
            same order as the input.

        Raises:
            ProviderError: If the model is not loaded or the vLLM call fails.
        """
        if self._pipe is None:
            raise ProviderError(
                "vLLM model is not loaded. The model may have been destroyed."
            )

        if isinstance(options, VLLMRuntimeOptions):
            sampling_params = options.to_sampling_params()
        else:
            from vllm import SamplingParams

            kw: dict[str, Any] = {}
            if options is not None and options.max_tokens is not None:
                kw["max_tokens"] = options.max_tokens
            sampling_params = SamplingParams(**kw)

        try:
            outputs = self._pipe.chat(
                messages, sampling_params, use_tqdm=False
            )
        except Exception as exc:
            raise ProviderError(
                f"vLLM offline batch generation failed: {exc}"
            ) from exc

        return [self._process_response(output) for output in outputs]

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
