"""VLLM provider implementation."""

from __future__ import annotations

import secrets
import time
from typing import Any

from llm_annotator.clients.base import (
    Provider,
    ProviderRuntimeOptions,
    Response,
)
from llm_annotator.clients.exceptions import ProviderError
from llm_annotator.clients.openai_client import OpenAIClient


# TODO: add custom options for each provider


class VLLMClient(OpenAIClient):
    """Client wrapper for VLLM's OpenAI-compatible server/client."""

    provider_type = Provider.VLLM

    def __init__(
        self,
        model: str | None = None,
        base_url: str = "http://localhost:8000/v1",
    ) -> None:
        """Initialize the VLLM client.

        Args:
            model: VLLM model identifier.
            base_url: Base URL for the vLLM API endpoint.
        """
        super().__init__(model=model or "", api_key="EMPTY", base_url=base_url)

        if model is None:
            models = self._client.models.list()
            self.model = models.data[0].id

    def batch_generate(
        self,
        *,
        messages: list[list[dict[str, str]]],
        options: ProviderRuntimeOptions | None = None,
    ) -> list[Response]:
        """Generate responses for a batch of inputs using vLLM's batch API.

        Uses the `/v1/chat/completions/batch` endpoint for efficient batch
        processing of multiple conversations.

        Args:
            messages: List of message lists, where each list is a conversation.
            options: Optional generation configuration.

        Returns:
            A list of Response objects, one per input conversation,
            indexed in the same order as input.

        Raises:
            ProviderError: If the batch request fails.
        """
        from openai.types.chat.chat_completion import ChatCompletion

        options = options or ProviderRuntimeOptions()
        try:
            # Construct batch request payload following vLLM batch API format
            request_payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": options.max_tokens,
            }
            if options.json_schema is not None:
                request_payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": options.json_schema,
                        "strict": True,
                    },
                }
            # The batch endpoint is at /v1/chat/completions/batch
            batch_url = f"{self._base_url}/chat/completions/batch"
            # Re-use the underlying httpx client
            http_client = self._client._client
            response = http_client.post(batch_url, json=request_payload)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise ProviderError(f"vLLM batch request failed: {exc}") from exc

        # Process batch response: convert each choice to a Response object
        responses: list[Response] = []
        for choice in data.get("choices", []):
            # Use random hash as id and current unix timestamp (int) as created
            dummy_response = ChatCompletion(
                id=f"chatcmpl-{secrets.token_hex(12)}",
                created=int(time.time()),
                model=self.model,
                choices=[choice],
            )
            resp = self._process_response(response=dummy_response)

            responses.append(resp)

        return responses
