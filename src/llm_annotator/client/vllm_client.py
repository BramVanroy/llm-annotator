"""OpenAI provider implementation."""

from __future__ import annotations

from llm_annotator.client.base import Provider
from llm_annotator.client.openai_client import OpenAIClient


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
        super().__init__(model=model, api_key="EMPTY", base_url=base_url)

        if model is None:
            models = self._client.models.list()
            self.model = models.data[0].id
