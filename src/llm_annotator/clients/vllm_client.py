"""VLLM provider implementation."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Any

from llm_annotator.clients.base import (
    OnError,
    Provider,
    ProviderRuntimeOptions,
    Response,
)
from llm_annotator.clients.exceptions import ConfigurationError, ProviderError
from llm_annotator.clients.openai_client import OpenAIClient


@dataclass(slots=True, frozen=True)
class VLLMBaseRuntimeOptions(ProviderRuntimeOptions):
    """Shared generation options for both vLLM server and offline clients.

    Attributes:
        top_k: Controls the number of top tokens to consider.
            Set to -1 to consider all tokens.
        repetition_penalty: Penalizes new tokens based on whether they appear
            in the prompt and the generated text so far. Values > 1 encourage
            the model to use new tokens; values < 1 encourage repetition.
        chat_template_kwargs: Additional kwargs forwarded to the chat template.
    """

    top_k: int | None = None
    repetition_penalty: float | None = None
    chat_template_kwargs: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        """Build the shared vLLM request payload dict.

        Returns:
            A dict containing the fields common to the vLLM server and offline
            clients.
        """
        payload: dict[str, Any] = {}
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.repetition_penalty is not None:
            payload["repetition_penalty"] = self.repetition_penalty
        return payload


@dataclass(slots=True, frozen=True)
class VLLMRuntimeOptions(VLLMBaseRuntimeOptions):
    """Generation options for the vLLM OpenAI-compatible server.

    Extends :class:`VLLMBaseRuntimeOptions` with server-specific parameters
    from the `/v1/chat/completions` extra-params API.
    See https://docs.vllm.ai/en/latest/serving/openai_compatible_server/#api-reference

    Attributes:
        add_generation_prompt: If ``True``, appends a generation prompt to each
            message. Defaults to ``True``.
        chat_template: Optional chat template string. When omitted the model’s
            default template is used.
        mm_processor_kwargs: Arguments forwarded to the model’s multi-modal
            processor (e.g. ``{"num_crops": 4}`` for Phi-3-Vision).
    """

    add_generation_prompt: bool = True
    chat_template: str | None = None
    mm_processor_kwargs: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        """Build the vLLM server request payload dict.

        Returns:
            A dict of vLLM server-specific request parameters, including all
            shared base fields.
        """
        payload: dict[str, Any] = {}
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.repetition_penalty is not None:
            payload["repetition_penalty"] = self.repetition_penalty
        if self.max_tokens is not None:
            payload["max_completion_tokens"] = self.max_tokens
        payload["add_generation_prompt"] = self.add_generation_prompt
        if self.chat_template is not None:
            payload["chat_template"] = self.chat_template
        if self.chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        if self.mm_processor_kwargs is not None:
            payload["mm_processor_kwargs"] = self.mm_processor_kwargs
        return payload


class VLLMClient(OpenAIClient[VLLMRuntimeOptions]):
    """Client wrapper for VLLM's OpenAI-compatible server/client."""

    provider_type = Provider.VLLM

    def __init__(
        self,
        model: str | None = None,
        base_url: str = "http://localhost:8000/v1",
        on_error: OnError = "raise",
    ) -> None:
        """Initialize the VLLM client.

        Args:
            model: VLLM model identifier.
            base_url: Base URL for the vLLM API endpoint.
            on_error: Error behavior when generation fails.
        """
        super().__init__(
            model=model or "",
            api_key="EMPTY",
            base_url=base_url,
            on_error=on_error,
        )

        if model is None:
            models = self._client.models.list()
            self.model = models.data[0].id

    def batch_generate(
        self,
        *,
        messages: list[list[dict[str, str]]],
        options: VLLMRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
        use_batch_api: bool = False,
        poll_interval: float = 10.0,
    ) -> list[Response]:
        """Generate responses for a batch of inputs using vLLM's native batch endpoint.

        Sends all conversations in a single request to ``/v1/chat/completions/batch``.
        The OpenAI Batch API is not supported; passing ``use_batch_api=True`` raises
        a :class:`ConfigurationError`.

        Args:
            messages: List of message lists, where each list is a conversation.
            options: Optional generation configuration.
            gen_kwargs: Additional provider-specific generation kwargs that are
                not covered by the standard options. Has precedence over
                ``options``.
            use_batch_api: Must be ``False``. The OpenAI Batch API is not
                supported by the vLLM server client.
            poll_interval: Accepted for interface compatibility with
                :class:`OpenAIClient`. Ignored.

        Returns:
            A list of Response objects, one per input conversation,
            indexed in the same order as input.

        Raises:
            ConfigurationError: If ``use_batch_api=True``.
            ProviderError: If the batch request fails.
        """
        if use_batch_api:
            raise ConfigurationError(
                "The vLLM server client does not support the OpenAI Batch API."
                " Set use_batch_api=False (the default) to use vLLM's native"
                " batch endpoint instead."
            )
        _ = poll_interval
        from openai.types.chat.chat_completion import ChatCompletion

        options = options or self._default_options()
        try:
            # Construct batch request payload following vLLM batch API format
            request_payload: dict[str, Any] = options.to_payload()
            request_payload["model"] = self.model
            request_payload["messages"] = messages
            if options.json_schema is not None:
                # TODO: test. Maybe we need "structured_outputs"
                # https://docs.vllm.ai/en/latest/serving/openai_compatible_server/#extra-parameters_1
                request_payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": options.json_schema,
                        "strict": True,
                    },
                }
            request_payload.update(gen_kwargs or {})

            # The batch endpoint is at /v1/chat/completions/batch
            batch_url = f"{self._base_url}/chat/completions/batch"
            # Re-use the underlying httpx client
            http_client = self._client._client
            response = http_client.post(batch_url, json=request_payload)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            error_response = self._handle_error(
                exc, context="vLLM batch request failed"
            )
            return [error_response for _ in messages]

        # Process batch response: convert each choice to a Response object
        responses: list[Response] = []
        for idx, choice in enumerate(data.get("choices", [])):
            # Use random hash as id and current unix timestamp (int) as created
            # so that we can use self._process_response from super
            dummy_response = ChatCompletion(
                id=f"chatcmpl-{secrets.token_hex(12)}",
                object="chat.completion",
                created=int(time.time()),
                model=self.model,
                choices=[choice],
            )
            try:
                resp = self._process_response(response=dummy_response)
            except Exception as exc:
                resp = self._handle_error(
                    exc,
                    context=f"vLLM batch response processing failed at index {idx}",
                )

            responses.append(resp)

        if len(responses) < len(messages):
            padding = len(messages) - len(responses)
            err = self._handle_error(
                ProviderError(
                    "vLLM batch response returned fewer choices than requested."
                ),
                context="vLLM batch response validation failed",
            )
            responses.extend([err for _ in range(padding)])

        return responses

    def _default_options(self) -> VLLMRuntimeOptions:
        """Return default runtime options for vLLM requests."""
        return VLLMRuntimeOptions()


__all__ = ["VLLMBaseRuntimeOptions", "VLLMClient", "VLLMRuntimeOptions"]
