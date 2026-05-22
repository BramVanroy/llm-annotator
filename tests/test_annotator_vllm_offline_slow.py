from __future__ import annotations

from collections.abc import Generator

import pytest

from llm_annotator.annotator import Annotator
from llm_annotator.clients.vllm_offline_client import (
    VLLMOfflineClient,
    VLLMRuntimeOptions,
)


pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="session")
def smollm_model_id() -> str:
    return "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="session")
def offline_vllm_client(
    smollm_model_id: str,
) -> Generator[VLLMOfflineClient, None, None]:
    """Create one offline client for all slow Annotator integration tests."""
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
            max_model_len=768,
            max_num_seqs=8,
            enforce_eager=True,
            extra_vllm_kwargs=extra_vllm_kwargs,
        )
        client.warm_up(
            system_message="You are a concise sentiment annotation assistant.",
            prompt_prefix="Read the review and classify sentiment.",
            options=VLLMRuntimeOptions(max_tokens=16, temperature=0.0),
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not initialize vLLM offline client: {exc}")

    yield client
    client.destroy()


def test_annotate_imdb_smoke_with_offline_vllm(
    tmp_path,
    offline_vllm_client: VLLMOfflineClient,
) -> None:
    # Verifies Annotator can process a small IMDb slice end-to-end with offline vLLM.
    annotator = Annotator(
        client=offline_vllm_client, batch_size=2, num_proc=None
    )

    ds = annotator.annotate_dataset(
        output_dir=tmp_path / "imdb-smoke",
        prompt_template=(
            "Classify the sentiment of this movie review as positive or negative: {text}"
        ),
        dataset_name="stanfordnlp/imdb",
        dataset_split="test",
        max_num_samples=6,
        keep_columns=["text", "label"],
        options=VLLMRuntimeOptions(max_tokens=24, temperature=0.0, seed=5),
        sort_by_length="shortest_first",
    )

    assert len(ds) == 6
    assert "text" in ds.column_names
    assert "label" in ds.column_names
    assert "response" in ds.column_names
    assert "finish_reason" in ds.column_names


def test_annotate_imdb_with_schema_and_task_prefix(
    tmp_path,
    offline_vllm_client: VLLMOfflineClient,
) -> None:
    # Verifies Annotator schema parsing path with prefixed output keys on IMDb samples.
    annotator = Annotator(
        client=offline_vllm_client, batch_size=2, num_proc=None
    )
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

    ds = annotator.annotate_dataset(
        output_dir=tmp_path / "imdb-schema",
        prompt_template=(
            "Return JSON with key sentiment (positive or negative) for this review: {text}"
        ),
        dataset_name="stanfordnlp/imdb",
        dataset_split="test",
        max_num_samples=5,
        keep_columns=["text", "label"],
        options=VLLMRuntimeOptions(max_tokens=32, temperature=0.0, seed=11),
        output_schema=schema,
        task_prefix="sent_",
        num_retries_invalid=0,
    )

    assert len(ds) == 5
    assert "sent_response" in ds.column_names
    assert "sent_valid_fields" in ds.column_names
    assert "sent_error" in ds.column_names
