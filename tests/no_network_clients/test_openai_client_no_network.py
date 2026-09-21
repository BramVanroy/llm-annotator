from __future__ import annotations

import json
import types
from typing import Any, cast

import pytest

from llm_annotator.clients.exceptions import ProviderError
from llm_annotator.clients.openai_client import (
    CONNECT_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
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
            max_completion_tokens=12,
            temperature=0.2,
            output_schema={"type": "object"},
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


def test_openai_runtime_options_to_payload() -> None:
    # Verifies optional OpenAI runtime fields are emitted in the payload.
    payload = OpenAIRuntimeOptions(
        max_completion_tokens=12,
        frequency_penalty=0.5,
        reasoning_effort="high",
        temperature=0.2,
        top_p=0.8,
        presence_penalty=1.5,
    ).to_payload()

    assert payload == {
        "max_completion_tokens": 12,
        "frequency_penalty": 0.5,
        "reasoning_effort": "high",
        "temperature": 0.2,
        "top_p": 0.8,
        "presence_penalty": 1.5,
    }


def test_openai_process_response_stop_reason_error(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies stop-reason failures are converted to error responses.
    _ = fake_openai_module
    from openai.types.chat.chat_completion import ChatCompletion

    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(
        model="gpt-test", on_error="ignore"
    )
    completion = ChatCompletion.model_validate(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "role": "assistant",
                        "content": " hello ",
                    },
                }
            ],
            "usage": {"completion_tokens": 4},
            "model": "fake-model",
        }
    )

    response = client._process_response(completion)

    assert response.text == "hello"
    assert response.error is not None
    assert response.stop_reason == "length"


def _completion(message: dict[str, Any]) -> Any:
    """Build a real ChatCompletion around one assistant message."""
    from openai.types.chat.chat_completion import ChatCompletion

    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-fake",
            "created": 0,
            "object": "chat.completion",
            "model": "fake-model",
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": message}
            ],
            "usage": {
                "completion_tokens": 4,
                "prompt_tokens": 1,
                "total_tokens": 5,
            },
        }
    )


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_openai_process_response_reads_reasoning(
    fake_openai_module: dict[str, Any],
    field: str,
) -> None:
    # Verifies the trace is picked up under either name an OpenAI-compatible
    # server may use: vLLM emits `reasoning`, others `reasoning_content`.
    _ = fake_openai_module
    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(model="gpt-test")

    response = client._process_response(
        _completion(
            {
                "role": "assistant",
                "content": " the answer ",
                field: " first I think ",
            }
        )
    )

    assert response.reasoning == "first I think"
    assert response.text == "the answer"


def test_openai_process_response_without_reasoning_content(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies a plain completion leaves Response.reasoning unset.
    _ = fake_openai_module
    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(model="gpt-test")

    response = client._process_response(
        _completion({"role": "assistant", "content": "the answer"})
    )

    assert response.reasoning is None


def test_openai_generate_request_error_follows_on_error(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies a failed request is an error Response unless on_error is raise.
    fake_openai_module["create_raises"] = RuntimeError("api down")
    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(
        model="gpt-test", on_error="ignore"
    )

    response = client.generate(messages=[{"role": "user", "content": "hi"}])

    assert response.text == ""
    assert response.error is not None
    assert "api down" in response.error
    assert response.error_type == "ProviderError"


def test_openai_generate_request_error_raises(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies on_error="raise" turns a failed request into a ProviderError.
    fake_openai_module["create_raises"] = RuntimeError("api down")
    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(
        model="gpt-test", on_error="raise"
    )

    with pytest.raises(ProviderError, match="api down"):
        client.generate(messages=[{"role": "user", "content": "hi"}])


def test_openai_generate_rejects_multiple_responses(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies a request for more than one response per sample is rejected.
    _ = fake_openai_module
    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(
        model="gpt-test", on_error="ignore"
    )

    with pytest.raises(ValueError, match="one response per sample"):
        client.generate(
            messages=[{"role": "user", "content": "hi"}],
            gen_kwargs={"n": 4},
        )


def test_openai_batch_generate_keeps_max_workers(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies a batch smaller than max_workers leaves the client's own value.
    _ = fake_openai_module
    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(
        model="gpt-test", max_workers=16
    )

    client.batch_generate(messages=[[{"role": "user", "content": "one"}]])

    assert client.max_workers == 16


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
        options=OpenAIRuntimeOptions(max_completion_tokens=8),
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
            error_file_id=None,
        ),
        types.SimpleNamespace(
            id="batch-fake",
            status="completed",
            output_file_id="file-output-fake",
            error_file_id=None,
        ),
    ]

    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(
        model="gpt-test", use_batch_api=True, batch_poll_interval=0.0
    )
    responses = client.batch_generate(
        messages=[[{"role": "user", "content": "ping"}]],
        options=OpenAIRuntimeOptions(max_completion_tokens=16),
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


def test_batch_api_missing_result_and_blank_lines(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies missing batch results become error responses and blank lines are ignored.
    fake_openai_module["batch_output_content"] = "\n" + _make_batch_output(
        ("request-1", "second")
    )

    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(
        model="gpt-test",
        on_error="ignore",
        use_batch_api=True,
        batch_poll_interval=0.0,
    )
    responses = client.batch_generate(
        messages=[
            [{"role": "user", "content": "msg0"}],
            [{"role": "user", "content": "msg1"}],
        ],
    )

    assert len(responses) == 2
    assert responses[0].error is not None
    assert responses[1].text == "second"


def test_batch_api_bad_status_and_processing_error(
    fake_openai_module: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Verifies bad status codes and response parsing failures are surfaced as errors.
    fake_openai_module["batch_output_content"] = "\n".join(
        [
            json.dumps(
                {
                    "id": "resp-r0",
                    "custom_id": "request-0",
                    "response": {
                        "status_code": 500,
                        "body": {},
                    },
                    "error": None,
                }
            ),
            json.dumps(
                {
                    "id": "resp-r1",
                    "custom_id": "request-1",
                    "response": {
                        "status_code": 200,
                        "body": {
                            "model": "fake-model",
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "message": {
                                        "role": "assistant",
                                        "content": "hello",
                                    },
                                }
                            ],
                            "usage": {"completion_tokens": 3},
                        },
                    },
                    "error": None,
                }
            ),
        ]
    )

    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(
        model="gpt-test",
        on_error="ignore",
        use_batch_api=True,
        batch_poll_interval=0.0,
    )
    monkeypatch.setattr(
        client,
        "_process_response",
        lambda response: (_ for _ in ()).throw(RuntimeError("bad completion")),
    )

    responses = client.batch_generate(
        messages=[
            [{"role": "user", "content": "hi"}],
            [{"role": "user", "content": "there"}],
        ],
    )

    assert len(responses) == 2
    assert responses[0].error is not None
    assert responses[1].error is not None


def test_batch_api_preserves_order(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies responses are returned in the same order as input messages.
    fake_openai_module["batch_output_content"] = _make_batch_output(
        ("request-1", "second"),
        ("request-0", "first"),  # deliberately out of order in output
    )

    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(
        model="gpt-test", use_batch_api=True, batch_poll_interval=0.0
    )
    responses = client.batch_generate(
        messages=[
            [{"role": "user", "content": "msg0"}],
            [{"role": "user", "content": "msg1"}],
        ],
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
        model="gpt-test",
        on_error="ignore",
        use_batch_api=True,
        batch_poll_interval=0.0,
    )
    responses = client.batch_generate(
        messages=[[{"role": "user", "content": "hi"}]],
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
            error_file_id=None,
        ),
    ]

    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(
        model="gpt-test",
        on_error="ignore",
        use_batch_api=True,
        batch_poll_interval=0.0,
    )
    responses = client.batch_generate(
        messages=[
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
        ],
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
            max_completion_tokens=8,
            output_schema={"type": "object", "properties": {}},
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


def test_destroy_cancels_active_batches_and_deletes_files(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies destroy() cancels each tracked batch and deletes its files.
    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(model="gpt-test")
    client._active_batches = {
        "batch-1": ["file-in-1"],
        "batch-2": ["file-in-2", "file-out-2"],
    }

    client.destroy()

    assert set(fake_openai_module["cancelled_batches"]) == {
        "batch-1",
        "batch-2",
    }
    assert set(fake_openai_module["deleted_files"]) == {
        "file-in-1",
        "file-in-2",
        "file-out-2",
    }
    assert client._active_batches == {}


def test_destroy_logs_cancel_and_delete_errors(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies a failed cancellation or deletion does not stop the clean-up.
    fake_openai_module["delete_raises"] = RuntimeError("delete failed")

    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(model="gpt-test")

    def _raising_cancel(batch_id: str) -> None:
        raise RuntimeError("cancel failed")

    client._client.batches.cancel = _raising_cancel  # type: ignore[assignment]
    client._active_batches = {"batch-x": ["file-x"]}

    client.destroy()  # Must not raise.

    assert fake_openai_module["deleted_files"] == ["file-x"]
    assert client._active_batches == {}


def test_batch_api_deletes_files_and_stops_tracking(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies the input and output files are deleted once results are read.
    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(
        model="gpt-test", use_batch_api=True, batch_poll_interval=0.0
    )

    client.batch_generate(
        messages=[[{"role": "user", "content": "hi"}]],
    )

    assert client._active_batches == {}
    assert fake_openai_module["deleted_files"] == [
        "file-fake",
        "file-output-fake",
    ]


def test_batch_api_deletes_files_when_reading_fails(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies a failed download still deletes the files of the batch.
    def _raising_content(file_id: str) -> object:
        raise RuntimeError("download failed")

    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(
        model="gpt-test",
        on_error="ignore",
        use_batch_api=True,
        batch_poll_interval=0.0,
    )
    client._client.files.content = _raising_content  # type: ignore[assignment]

    responses = client.batch_generate(
        messages=[[{"role": "user", "content": "hi"}]],
    )

    assert responses[0].error is not None
    assert "download failed" in responses[0].error
    assert client._active_batches == {}
    assert fake_openai_module["deleted_files"] == [
        "file-fake",
        "file-output-fake",
    ]


def test_batch_api_expired_keeps_finished_requests(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies an expired batch keeps the requests its output file holds and
    # marks only the missing ones as errors, naming the status.
    fake_openai_module["batch_retrieve_responses"] = [
        types.SimpleNamespace(
            id="batch-fake",
            status="expired",
            output_file_id="file-output-fake",
            error_file_id=None,
        ),
    ]
    fake_openai_module["batch_output_content"] = _make_batch_output(
        ("request-0", "first")
    )

    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(
        model="gpt-test",
        on_error="ignore",
        use_batch_api=True,
        batch_poll_interval=0.0,
    )
    responses = client.batch_generate(
        messages=[
            [{"role": "user", "content": "msg0"}],
            [{"role": "user", "content": "msg1"}],
        ],
    )

    assert responses[0].text == "first"
    assert responses[0].error is None
    assert responses[1].error is not None
    assert "expired" in responses[1].error


def test_batch_api_reads_the_error_file(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies entries of the error file are read alongside the output file.
    fake_openai_module["batch_retrieve_responses"] = [
        types.SimpleNamespace(
            id="batch-fake",
            status="completed",
            output_file_id="file-output-fake",
            error_file_id="file-error-fake",
        ),
    ]
    fake_openai_module["file_contents"] = {
        "file-output-fake": _make_batch_output(("request-1", "second")),
        "file-error-fake": json.dumps(
            {
                "id": "resp-r0",
                "custom_id": "request-0",
                "response": None,
                "error": {"code": "invalid_request", "message": "too long"},
            }
        ),
    }

    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(
        model="gpt-test",
        on_error="ignore",
        use_batch_api=True,
        batch_poll_interval=0.0,
    )
    responses = client.batch_generate(
        messages=[
            [{"role": "user", "content": "msg0"}],
            [{"role": "user", "content": "msg1"}],
        ],
    )

    assert responses[0].error is not None
    assert "too long" in responses[0].error
    assert responses[1].text == "second"
    assert fake_openai_module["deleted_files"] == [
        "file-fake",
        "file-output-fake",
        "file-error-fake",
    ]


def test_batch_api_keeps_tracking_an_interrupted_batch(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies an interrupted poll leaves the batch and its input file for
    # destroy() instead of deleting the file of a running job.
    def _raising_retrieve(batch_id: str) -> object:
        raise KeyboardInterrupt

    fake_openai_module["batch_initial_status"] = "in_progress"
    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(
        model="gpt-test", use_batch_api=True, batch_poll_interval=0.0
    )
    client._client.batches.retrieve = _raising_retrieve  # type: ignore[assignment]

    with pytest.raises(KeyboardInterrupt):
        client.batch_generate(
            messages=[[{"role": "user", "content": "hi"}]],
        )

    assert client._active_batches == {"batch-fake": ["file-fake"]}
    assert fake_openai_module["deleted_files"] == []

    client.destroy()

    assert fake_openai_module["cancelled_batches"] == ["batch-fake"]
    assert fake_openai_module["deleted_files"] == ["file-fake"]


def test_openai_client_passes_the_sdk_defaults(
    fake_openai_module: dict[str, Any],
) -> None:
    """The SDK client gets the OpenAI SDK's own timeout and retry count.

    The timeout is an ``httpx.Timeout`` rather than a plain float, so the
    connect limit stays short while the read limit is the full timeout.
    """
    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(model="gpt-test")

    assert client.timeout == DEFAULT_TIMEOUT
    assert client.max_retries == DEFAULT_MAX_RETRIES
    assert client.use_batch_api is False
    assert client.batch_poll_interval == 10.0
    inits = cast(list[Any], fake_openai_module["openai_init_kwargs"])
    assert len(inits) == 1
    assert inits[0]["max_retries"] == DEFAULT_MAX_RETRIES
    assert inits[0]["timeout"].read == DEFAULT_TIMEOUT
    assert inits[0]["timeout"].connect == CONNECT_TIMEOUT


def test_openai_client_takes_its_own_timeout_and_retries(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies the constructor arguments reach the SDK client.
    OpenAIClient(model="gpt-test", timeout=30.0, max_retries=5)

    sdk_kwargs = cast(list[Any], fake_openai_module["openai_init_kwargs"])[-1]
    assert sdk_kwargs["timeout"].read == 30.0
    assert sdk_kwargs["max_retries"] == 5


def test_openai_batch_generate_uses_the_thread_pool_by_default(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies a client built without use_batch_api sends no batch job.
    client: OpenAIClient[OpenAIRuntimeOptions] = OpenAIClient(model="gpt-test")

    client.batch_generate(messages=[[{"role": "user", "content": "hi"}]])

    assert fake_openai_module["created_batches"] == []
    assert fake_openai_module["uploaded_files"] == []
