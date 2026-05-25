from __future__ import annotations

import json
import types
from typing import Any, cast

import pytest

from llm_annotator.clients.openai_client import (
    OpenAIClient,
    OpenAIRuntimeOptions,
)


pytestmark = pytest.mark.usefixtures("block_network")


def test_openai_generate_builds_payload_and_parses_response(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies request payload shaping and response parsing for OpenAI generate.
    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(model="gpt-test")

    response = client.generate(
        messages=[{"role": "user", "content": "Hello"}],
        options=OpenAIRuntimeOptions(
            max_tokens=12,
            temperature=0.2,
            json_schema={"type": "object"},
        ),
        gen_kwargs={"temperature": 0.9},
    )

    request = fake_openai_module["last_create_kwargs"]
    assert isinstance(request, dict)
    assert request["model"] == "gpt-test"
    assert request["max_completion_tokens"] == 12
    assert request["temperature"] == 0.9
    assert request["response_format"]["type"] == "json_schema"
    assert response.text == "hello"
    assert response.model == "fake-model"
    assert response.num_output_tokens == 7


def test_openai_generate_request_error_returns_error_response(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies provider request errors are converted to error responses.
    fake_openai_module["create_raises"] = RuntimeError("api down")
    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(
        model="gpt-test", on_error="ignore"
    )

    response = client.generate(messages=[{"role": "user", "content": "hi"}])

    assert response.error is not None
    assert response.error_type == "ProviderError"


def test_openai_batch_generate_preserves_input_order(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies OpenAI batch_generate returns one response per input in order.
    _ = fake_openai_module
    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(model="gpt-test")

    responses = client.batch_generate(
        messages=[
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ],
        options=OpenAIRuntimeOptions(max_tokens=8),
    )

    assert len(responses) == 2
    assert all(r.text == "hello" for r in responses)


@pytest.mark.parametrize(
    "stop_reason",
    [
        None,
        "stop",
        "length",
        "content_filter",
        "tool_calls",
        "function_call",
        "weird",
    ],
)
def test_openai_handle_stop_reason_branches(stop_reason: str | None) -> None:
    # Verifies OpenAI stop-reason handler accepts success and raises for failures.
    client: OpenAIClient[OpenAIRuntimeOptions] = object.__new__(OpenAIClient)
    client.model = "x"
    client.max_workers = 1
    client.on_error = "raise"
    client._logger = cast(
        Any,
        types.SimpleNamespace(
            warning=lambda _msg: None, debug=lambda _msg: None
        ),
    )

    if stop_reason == "stop":
        client._handle_stop_reason(
            stop_reason=stop_reason, num_output_tokens=1
        )
        return

    with pytest.raises(Exception):
        client._handle_stop_reason(
            stop_reason=stop_reason, num_output_tokens=1
        )


# ---------------------------------------------------------------------------
# Batch API tests
# ---------------------------------------------------------------------------


def _make_batch_output(*custom_ids_and_content: tuple[str, str]) -> str:
    """Build a JSONL batch-output string for the given (custom_id, text) pairs."""
    lines = []
    for custom_id, content in custom_ids_and_content:
        lines.append(
            json.dumps(
                {
                    "id": f"resp-{custom_id}",
                    "custom_id": custom_id,
                    "response": {
                        "status_code": 200,
                        "body": {
                            "model": "fake-model",
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "message": {
                                        "role": "assistant",
                                        "content": content,
                                    },
                                }
                            ],
                            "usage": {
                                "completion_tokens": 7,
                                "prompt_tokens": 10,
                                "total_tokens": 17,
                            },
                        },
                    },
                    "error": None,
                }
            )
        )
    return "\n".join(lines)


def test_batch_api_happy_path(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies file upload, batch creation, polling, and output parsing.
    fake_openai_module["batch_output_content"] = _make_batch_output(
        ("request-0", "hello")
    )
    # First retrieve returns in_progress, second returns completed.
    fake_openai_module["batch_retrieve_responses"] = [
        types.SimpleNamespace(
            id="batch-fake",
            status="in_progress",
            output_file_id="file-output-fake",
        ),
        types.SimpleNamespace(
            id="batch-fake",
            status="completed",
            output_file_id="file-output-fake",
        ),
    ]

    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(model="gpt-test")
    responses = client.batch_generate(
        messages=[[{"role": "user", "content": "ping"}]],
        options=OpenAIRuntimeOptions(max_tokens=16),
        use_batch_api=True,
        poll_interval=0.0,
    )

    assert len(responses) == 1
    assert responses[0].text == "hello"
    assert responses[0].model == "fake-model"
    assert responses[0].num_output_tokens == 7

    # Verify file was uploaded and batch was created with correct args.
    assert len(fake_openai_module["uploaded_files"]) == 1
    assert fake_openai_module["uploaded_files"][0]["purpose"] == "batch"
    assert len(fake_openai_module["created_batches"]) == 1
    batch_call = fake_openai_module["created_batches"][0]
    assert batch_call["endpoint"] == "/v1/chat/completions"
    assert batch_call["completion_window"] == "24h"
    assert batch_call["input_file_id"] == "file-fake"


def test_batch_api_preserves_order(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies responses are returned in the same order as input messages.
    fake_openai_module["batch_output_content"] = _make_batch_output(
        ("request-1", "second"),
        ("request-0", "first"),  # deliberately out of order in output
    )

    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(model="gpt-test")
    responses = client.batch_generate(
        messages=[
            [{"role": "user", "content": "msg0"}],
            [{"role": "user", "content": "msg1"}],
        ],
        use_batch_api=True,
        poll_interval=0.0,
    )

    assert len(responses) == 2
    assert responses[0].text == "first"
    assert responses[1].text == "second"


def test_batch_api_item_error_returns_error_response(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies that a per-item error in the output produces an error Response.
    fake_openai_module["batch_output_content"] = json.dumps(
        {
            "id": "resp-r0",
            "custom_id": "request-0",
            "response": None,
            "error": {"code": "server_error", "message": "something broke"},
        }
    )

    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(
        model="gpt-test", on_error="ignore"
    )
    responses = client.batch_generate(
        messages=[[{"role": "user", "content": "hi"}]],
        use_batch_api=True,
        poll_interval=0.0,
    )

    assert len(responses) == 1
    assert responses[0].error is not None
    assert responses[0].error_type == "ProviderError"


def test_batch_api_batch_failure_returns_all_error_responses(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies that a batch-level failure produces error Responses for all items.
    fake_openai_module["batch_retrieve_responses"] = [
        types.SimpleNamespace(
            id="batch-fake",
            status="failed",
            output_file_id=None,
        ),
    ]

    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(
        model="gpt-test", on_error="ignore"
    )
    responses = client.batch_generate(
        messages=[
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
        ],
        use_batch_api=True,
        poll_interval=0.0,
    )

    assert len(responses) == 2
    assert all(r.error is not None for r in responses)
    assert all(r.error_type == "ProviderError" for r in responses)


def test_batch_api_build_request_includes_json_schema(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies that json_schema in options is forwarded to the batch request body.
    uploaded_jsonl: list[str] = []

    original_create = fake_openai_module

    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(model="gpt-test")

    # We don't need to run the full batch; just inspect the JSONL via
    # _build_batch_request directly.
    req = client._build_batch_request(
        0,
        [{"role": "user", "content": "x"}],
        OpenAIRuntimeOptions(
            max_tokens=8, json_schema={"type": "object", "properties": {}}
        ),
        None,
    )
    _ = uploaded_jsonl
    _ = original_create
    assert req["custom_id"] == "request-0"
    assert req["method"] == "POST"
    assert req["url"] == "/v1/chat/completions"
    body = req["body"]
    assert body["model"] == "gpt-test"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True


def test_destroy_cancels_active_batches(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies destroy() calls cancel for each tracked batch id.
    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(model="gpt-test")
    client._active_batch_ids = ["batch-1", "batch-2"]

    client.destroy()

    assert set(fake_openai_module["cancelled_batches"]) == {
        "batch-1",
        "batch-2",
    }
    assert client._active_batch_ids == []


def test_destroy_swallows_cancel_errors(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies destroy() does not propagate exceptions from cancellation.
    _ = fake_openai_module

    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(model="gpt-test")

    def _raising_cancel(batch_id: str) -> None:
        raise RuntimeError("cancel failed")

    client._client.batches.cancel = _raising_cancel  # type: ignore[assignment]
    client._active_batch_ids = ["batch-x"]

    client.destroy()  # Must not raise.
    assert client._active_batch_ids == []


def test_batch_api_batch_id_removed_after_completion(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies the batch id is removed from _active_batch_ids after the job completes.
    _ = fake_openai_module
    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(model="gpt-test")

    client.batch_generate(
        messages=[[{"role": "user", "content": "hi"}]],
        use_batch_api=True,
        poll_interval=0.0,
    )

    assert client._active_batch_ids == []
