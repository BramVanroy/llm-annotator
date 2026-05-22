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
from llm_annotator.clients.exceptions import ProviderError
from llm_annotator.clients.openai_client import OpenAIClient


@dataclass(slots=True, frozen=True)
class VLLMRuntimeOptions(ProviderRuntimeOptions):
    # https://docs.vllm.ai/en/latest/serving/openai_compatible_server/#api-reference
    top_k: int | None = None
    """Controls the number of top tokens to consider. Set to 0 (or -1) to consider all tokens."""
    repetition_penalty: float | None = None
    """Penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1 encourage the model to use new tokens, while values < 1 encourage the model to repeat tokens."""
    add_generation_prompt: bool = True
    """If True, adds a generation template to each message."""
    chat_template: str | None = None
    """The template to use for structuring the chat. If not provided, the model's default chat template will be used."""
    chat_template_kwargs: dict[str, Any] | None = None
    """Additional kwargs to pass to the chat template."""
    mm_processor_kwargs: dict[str, Any] | None = None
    """Arguments to be forwarded to the model's processor for multi-modal data, e.g., image processor. Overrides for the multi-modal processor obtained from AutoProcessor.from_pretrained. The available overrides depend on the model that is being run. For example, for Phi-3-Vision: {"num_crops": 4}."""


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
    ) -> list[Response]:
        """Generate responses for a batch of inputs using vLLM's batch API.

        Uses the `/v1/chat/completions/batch` endpoint for efficient batch
        processing of multiple conversations.

        Args:
            messages: List of message lists, where each list is a conversation.
            options: Optional generation configuration.
            gen_kwargs: Additional provider-specific generation kwargs that are not covered by the standard options.
                Has precedence over ``options``.

        Returns:
            A list of Response objects, one per input conversation,
            indexed in the same order as input.

        Raises:
            ProviderError: If the batch request fails.
        """
        from openai.types.chat.chat_completion import ChatCompletion

        options = options or self._default_options()
        try:
            # Construct batch request payload following vLLM batch API format
            request_payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": options.max_tokens,
            }
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

            opts = options.dict(exclude_none=True)
            opts.pop("max_tokens", None)
            opts.pop("json_schema", None)
            request_payload.update(opts)
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


__all__ = ["VLLMClient", "VLLMRuntimeOptions"]
