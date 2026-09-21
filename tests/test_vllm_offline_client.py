from __future__ import annotations

import json
from collections.abc import Generator

import pytest

from llm_annotator.clients.base import Provider
from llm_annotator.clients.vllm_offline_client import (
    VLLMOfflineClient,
    VLLMOfflineRuntimeOptions,
)


pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="session")
def smollm_model_id() -> str:
    return "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def vllm_offline_smollm_client(
    smollm_model_id: str,
) -> Generator[VLLMOfflineClient, None, None]:
    """Create one offline vLLM client for the tests of this module.

    One client per module keeps the tests fast, and the module scope makes
    the engine release the GPU before the next slow module starts its own.
    """
    try:
        import vllm  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"vLLM is not available: {exc}")

    try:
        import torch

        has_gpu = torch.cuda.is_available()
    except Exception:
        has_gpu = False
    extra_vllm_kwargs: dict[str, str] = {} if has_gpu else {"device": "cpu"}

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
            options=VLLMOfflineRuntimeOptions(max_completion_tokens=8),
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        # With a GPU the engine has to start, so a failure is a finding and
        # not a reason to skip.
        if has_gpu:
            raise
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
        options=VLLMOfflineRuntimeOptions(
            max_completion_tokens=64, temperature=0.0, seed=0
        ),
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
        # A response that hits the token limit is reported as an error, and
        # a 135M model does not end a farewell on its own. The stop strings
        # end each answer at its first sentence.
        options=VLLMOfflineRuntimeOptions(
            max_completion_tokens=64,
            temperature=0.0,
            seed=1,
            stop=[".", "!", "?", "\n"],
        ),
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
        options=VLLMOfflineRuntimeOptions(
            max_completion_tokens=32,
            temperature=0.0,
            seed=7,
            json_schema=schema,
        ),
    )

    assert response.error is None
    payload = json.loads(response.text)
    assert payload["sentiment"] in {"positive", "negative"}
