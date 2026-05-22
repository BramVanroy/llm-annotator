from __future__ import annotations

import types
from typing import Any, cast

import pytest

from llm_annotator.clients.claude_client import (
    ClaudeClient,
    ClaudeRuntimeOptions,
    _extract_system_instruction,
)


pytestmark = pytest.mark.usefixtures("block_network")


def test_extract_system_instruction_happy_path() -> None:
    # Verifies Claude helper separates a leading system message.
    messages, system = _extract_system_instruction(
        [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "question"},
        ]
    )

    assert system == "rules"
    assert messages == [{"role": "user", "content": "question"}]


def test_claude_extract_system_instruction_errors() -> None:
    # Verifies Claude helper rejects multiple or misplaced system messages.
    with pytest.raises(Exception):
        _extract_system_instruction(
            [
                {"role": "system", "content": "a"},
                {"role": "system", "content": "b"},
            ]
        )

    with pytest.raises(ValueError):
        _extract_system_instruction(
            [
                {"role": "user", "content": "u"},
                {"role": "system", "content": "s"},
            ]
        )


def test_claude_generate_builds_payload_and_parses_response(
    fake_anthropic_module: dict[str, Any],
) -> None:
    # Verifies Claude payload construction and text block joining.
    client = ClaudeClient(model="claude-test")

    response = client.generate(
        messages=[
            {"role": "system", "content": "Use JSON"},
            {"role": "user", "content": "Summarize"},
        ],
        options=ClaudeRuntimeOptions(
            max_tokens=11,
            json_schema={"type": "object"},
            effort="low",
            thinking_type="adaptive",
            thinking_display="summarized",
        ),
        gen_kwargs={"max_tokens": 99},
    )

    request = fake_anthropic_module["last_create_kwargs"]
    assert isinstance(request, dict)
    assert request["model"] == "claude-test"
    assert request["system"] == "Use JSON"
    assert request["max_tokens"] == 99
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["thinking"]["type"] == "adaptive"
    assert response.text == "first line\nsecond line"


def test_claude_generate_request_error_returns_error_response(
    fake_anthropic_module: dict[str, Any],
) -> None:
    # Verifies Claude request failures return structured error responses.
    fake_anthropic_module["create_raises"] = RuntimeError("api down")
    client = ClaudeClient(model="claude-test", on_error="ignore")

    response = client.generate(messages=[{"role": "user", "content": "hi"}])

    assert response.error is not None
    assert response.error_type == "ProviderError"


@pytest.mark.parametrize("stop_reason", [None, "end_turn", "stop_sequence"])
def test_claude_handle_stop_reason_success_branches(
    stop_reason: str | None,
) -> None:
    # Verifies Claude stop-reason handler accepts all success reasons.
    client = object.__new__(ClaudeClient)
    client.model = "c"
    client.max_workers = 1
    client.on_error = "raise"
    client._logger = cast(
        Any,
        types.SimpleNamespace(
            warning=lambda _msg: None, debug=lambda _msg: None
        ),
    )

    client._handle_stop_reason(stop_reason=stop_reason, num_output_tokens=3)


@pytest.mark.parametrize(
    "stop_reason",
    [
        "max_tokens",
        "tool_use",
        "pause_turn",
        "refusal",
        "model_context_window_exceeded",
        "weird",
    ],
)
def test_claude_handle_stop_reason_error_branches(stop_reason: str) -> None:
    # Verifies Claude stop-reason handler raises for each non-success reason.
    client = object.__new__(ClaudeClient)
    client.model = "c"
    client.max_workers = 1
    client.on_error = "raise"
    client._logger = cast(
        Any,
        types.SimpleNamespace(
            warning=lambda _msg: None, debug=lambda _msg: None
        ),
    )

    with pytest.raises(Exception):
        client._handle_stop_reason(
            stop_reason=stop_reason, num_output_tokens=3
        )


def test_claude_destroy_cancels_running_batches() -> None:
    # Verifies Claude destroy cancels all tracked running batch ids.
    cancelled: list[str] = []
    client = object.__new__(ClaudeClient)
    client._client = cast(
        Any,
        types.SimpleNamespace(
            messages=types.SimpleNamespace(
                batches=types.SimpleNamespace(
                    cancel=lambda batch_id: cancelled.append(batch_id)
                )
            )
        ),
    )
    client._running_batch_ids = {"a", "b"}
    client.destroy()
    assert sorted(cancelled) == ["a", "b"]
