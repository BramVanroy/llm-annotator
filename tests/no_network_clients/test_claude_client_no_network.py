from __future__ import annotations

import types
from typing import Any, cast

import pytest

from llm_annotator.clients.base import Response
from llm_annotator.clients.claude_client import (
    CONNECT_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    ClaudeClient,
    ClaudeRuntimeOptions,
    _extract_system_instruction,
)
from llm_annotator.clients.exceptions import ProviderError


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
    with pytest.raises(ValueError, match="single system message"):
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
            max_completion_tokens=11,
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


def test_claude_runtime_options_to_payload() -> None:
    # Verifies optional Claude runtime fields are emitted in the payload.
    payload = ClaudeRuntimeOptions(
        max_completion_tokens=11,
        effort="high",
        thinking_type="enabled",
        thinking_budget=2048,
        thinking_display="full",
    ).to_payload()

    assert payload == {
        "max_tokens": 11,
        "output_config": {"effort": "high"},
        "thinking": {
            "type": "enabled",
            "budget_tokens": 2048,
            "display": "full",
        },
    }


def test_claude_process_response_handles_mixed_blocks_and_error(
    fake_anthropic_module: dict[str, Any],
) -> None:
    # Verifies non-text blocks are skipped and stop-reason failures become error responses.
    _ = fake_anthropic_module
    client = object.__new__(ClaudeClient)
    client.model = "claude-test"
    client.max_workers = 1
    client.on_error = "ignore"
    client._logger = cast(
        Any,
        types.SimpleNamespace(
            warning=lambda _msg: None, debug=lambda _msg: None
        ),
    )

    response = types.SimpleNamespace(
        usage=types.SimpleNamespace(output_tokens=9),
        stop_reason="max_tokens",
        model="claude-fake",
        content=[
            types.SimpleNamespace(type="text", text="first"),
            types.SimpleNamespace(type="tool_use", text="ignore"),
            types.SimpleNamespace(type="text", text="second"),
        ],
    )

    parsed = client._process_response(response)  # type: ignore[arg-type]

    assert parsed.text == "first\nsecond"
    assert parsed.error is not None
    assert parsed.stop_reason == "max_tokens"


def test_claude_process_response_collects_thinking_blocks(
    fake_anthropic_module: dict[str, Any],
) -> None:
    # Verifies thinking blocks are separated from the text of the answer.
    _ = fake_anthropic_module
    client = object.__new__(ClaudeClient)
    client.model = "claude-test"
    client.max_workers = 1
    client.on_error = "ignore"
    client._logger = cast(
        Any,
        types.SimpleNamespace(
            warning=lambda _msg: None, debug=lambda _msg: None
        ),
    )

    response = types.SimpleNamespace(
        usage=types.SimpleNamespace(output_tokens=9),
        stop_reason="end_turn",
        model="claude-fake",
        content=[
            types.SimpleNamespace(type="thinking", thinking="step one"),
            types.SimpleNamespace(type="thinking", thinking="step two"),
            types.SimpleNamespace(type="text", text="the answer"),
        ],
    )

    parsed = client._process_response(response)  # type: ignore[arg-type]

    assert parsed.reasoning == "step one\nstep two"
    assert parsed.text == "the answer"


def test_claude_generate_request_error_follows_on_error(
    fake_anthropic_module: dict[str, Any],
) -> None:
    # Verifies a failed request is an error Response unless on_error is raise.
    fake_anthropic_module["create_raises"] = RuntimeError("api down")
    client = ClaudeClient(model="claude-test", on_error="ignore")

    response = client.generate(messages=[{"role": "user", "content": "hi"}])

    assert response.text == ""
    assert response.error is not None
    assert "api down" in response.error
    assert response.error_type == "ProviderError"


def test_claude_generate_request_error_raises(
    fake_anthropic_module: dict[str, Any],
) -> None:
    # Verifies on_error="raise" turns a failed request into a ProviderError.
    fake_anthropic_module["create_raises"] = RuntimeError("api down")
    client = ClaudeClient(model="claude-test", on_error="raise")

    with pytest.raises(ProviderError, match="api down"):
        client.generate(messages=[{"role": "user", "content": "hi"}])


def test_claude_extract_system_instruction_drops_empty_system() -> None:
    # Verifies an empty system message is removed from the message list.
    messages, system = _extract_system_instruction(
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": "question"},
        ]
    )

    assert system == ""
    assert messages == [{"role": "user", "content": "question"}]


def test_claude_generate_omits_empty_system(
    fake_anthropic_module: dict[str, Any],
) -> None:
    # Verifies an empty system message reaches neither 'messages' nor 'system'.
    client = ClaudeClient(model="claude-test")

    client.generate(
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": "Summarize"},
        ]
    )

    request = fake_anthropic_module["last_create_kwargs"]
    assert "system" not in request
    assert request["messages"] == [{"role": "user", "content": "Summarize"}]


@pytest.mark.parametrize("max_workers", [0, None])
def test_claude_batch_generate_runs_sequentially(
    fake_anthropic_module: dict[str, Any],
    max_workers: int | None,
) -> None:
    # Verifies a worker count of 0 or None answers every input sequentially.
    _ = fake_anthropic_module
    client = ClaudeClient(model="claude-test", max_workers=max_workers)

    responses = client.batch_generate(
        messages=[
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ]
    )

    assert len(responses) == 2
    assert all(r.text == "first line\nsecond line" for r in responses)


def test_claude_batch_generate_handles_worker_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Verifies exceptions raised inside worker tasks are converted to error responses.
    client = object.__new__(ClaudeClient)
    client.model = "claude-test"
    client.max_workers = 2
    client.on_error = "ignore"
    client._logger = cast(
        Any,
        types.SimpleNamespace(
            warning=lambda _msg: None, debug=lambda _msg: None
        ),
    )

    def _generate(
        self: ClaudeClient,
        *,
        messages: list[dict[str, str]],
        options: ClaudeRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> Response:
        _ = options
        _ = gen_kwargs
        if messages[0]["content"] == "bad":
            raise RuntimeError("boom")
        return Response(
            text=messages[0]["content"],
            provider=self.provider_type,
            model=self.model,
        )

    monkeypatch.setattr(ClaudeClient, "generate", _generate)

    responses = client.batch_generate(
        messages=[
            [{"role": "user", "content": "good"}],
            [{"role": "user", "content": "bad"}],
        ]
    )

    assert responses[0].text == "good"
    assert responses[1].error is not None


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


def test_claude_client_passes_the_sdk_defaults(
    fake_anthropic_module: dict[str, Any],
) -> None:
    """The SDK client gets the Anthropic SDK's own timeout and retry count.

    The timeout is an ``httpx.Timeout`` rather than a plain float, so the
    connect limit stays short while the read limit is the full timeout.
    """
    client = ClaudeClient(model="claude-test")

    assert client.timeout == DEFAULT_TIMEOUT
    assert client.max_retries == DEFAULT_MAX_RETRIES
    inits = cast(list[Any], fake_anthropic_module["anthropic_init_kwargs"])
    assert len(inits) == 1
    assert inits[0]["max_retries"] == DEFAULT_MAX_RETRIES
    assert inits[0]["timeout"].read == DEFAULT_TIMEOUT
    assert inits[0]["timeout"].connect == CONNECT_TIMEOUT


def test_claude_client_takes_its_own_timeout_and_retries(
    fake_anthropic_module: dict[str, Any],
) -> None:
    # Verifies the constructor arguments reach the SDK client.
    ClaudeClient(model="claude-test", timeout=45.0, max_retries=7)

    sdk_kwargs = cast(
        list[Any], fake_anthropic_module["anthropic_init_kwargs"]
    )[-1]
    assert sdk_kwargs["timeout"].read == 45.0
    assert sdk_kwargs["max_retries"] == 7
