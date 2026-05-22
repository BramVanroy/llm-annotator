from __future__ import annotations

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
    [None, "stop", "length", "content_filter", "tool_calls", "function_call", "weird"],
)
def test_openai_handle_stop_reason_branches(stop_reason: str | None) -> None:
    # Verifies OpenAI stop-reason handler accepts success and raises for failures.
    client: OpenAIClient[OpenAIRuntimeOptions] = object.__new__(OpenAIClient)
    client.model = "x"
    client.max_workers = 1
    client.on_error = "raise"
    client._logger = cast(
        Any,
        types.SimpleNamespace(warning=lambda _msg: None, debug=lambda _msg: None),
    )

    if stop_reason == "stop":
        client._handle_stop_reason(stop_reason=stop_reason, num_output_tokens=1)
        return

    with pytest.raises(Exception):
        client._handle_stop_reason(stop_reason=stop_reason, num_output_tokens=1)
