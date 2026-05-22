from __future__ import annotations

import json
from collections.abc import Generator

import pytest

from llm_annotator.clients.base import Provider
from llm_annotator.clients.vllm_offline_client import (
    VLLMOfflineClient,
    VLLMRuntimeOptions,
)


pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="session")
def smollm_model_id() -> str:
    return "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="session")
def vllm_offline_smollm_client(
    smollm_model_id: str,
) -> Generator[VLLMOfflineClient, None, None]:
    """Create one offline vLLM client for the full test session.

    Reusing a single client keeps these integration tests fast by avoiding
    repeated model load and tokenizer initialization costs.
    """
    try:
        import vllm  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"vLLM is not available: {exc}")

    extra_vllm_kwargs: dict[str, str] = {}
    try:
        import torch

        if not torch.cuda.is_available():
            extra_vllm_kwargs["device"] = "cpu"
    except Exception:
        extra_vllm_kwargs["device"] = "cpu"

    try:
        client = VLLMOfflineClient(
            model=smollm_model_id,
            max_model_len=512,
            max_num_seqs=8,
            enforce_eager=True,
            extra_vllm_kwargs=extra_vllm_kwargs,
        )
        client.warm_up(
            system_message="You are a concise assistant.",
            prompt_prefix="Answer briefly.",
            options=VLLMRuntimeOptions(max_tokens=8),
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not initialize vLLM offline test client: {exc}")

    yield client

    client.destroy()


def test_generate_with_smollm(
    vllm_offline_smollm_client: VLLMOfflineClient,
    smollm_model_id: str,
) -> None:
    # Verifies single-request generation returns non-empty output and metadata.
    response = vllm_offline_smollm_client.generate(
        messages=[
            {
                "role": "user",
                "content": "Reply with only one word: hello",
            }
        ],
        options=VLLMRuntimeOptions(max_tokens=10, temperature=0.0, seed=0),
    )

    assert response.error is None
    assert response.provider == Provider.VLLM_OFFLINE
    assert response.model == smollm_model_id
    assert response.text.strip()


def test_batch_generate_with_smollm(
    vllm_offline_smollm_client: VLLMOfflineClient,
) -> None:
    # Verifies batch generation returns one successful response per input prompt.
    responses = vllm_offline_smollm_client.batch_generate(
        messages=[
            [{"role": "user", "content": "Reply with one short greeting."}],
            [{"role": "user", "content": "Reply with one short farewell."}],
        ],
        options=VLLMRuntimeOptions(max_tokens=12, temperature=0.0, seed=1),
    )

    assert len(responses) == 2
    assert all(response.error is None for response in responses)
    assert all(response.text.strip() for response in responses)


def test_guided_json_generation_with_smollm(
    vllm_offline_smollm_client: VLLMOfflineClient,
) -> None:
    # Verifies guided decoding with JSON schema yields parseable structured output.
    schema = {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative"],
            }
        },
        "required": ["sentiment"],
    }

    response = vllm_offline_smollm_client.generate(
        messages=[
            {
                "role": "user",
                "content": "Classify the sentiment as positive or negative: I love this movie.",
            }
        ],
        options=VLLMRuntimeOptions(
            max_tokens=32,
            temperature=0.0,
            seed=7,
            json_schema=schema,
        ),
    )

    assert response.error is None
    payload = json.loads(response.text)
    assert payload["sentiment"] in {"positive", "negative"}
