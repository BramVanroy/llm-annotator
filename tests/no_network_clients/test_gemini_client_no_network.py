from __future__ import annotations

import types
from typing import Any, cast

import pytest

from llm_annotator.clients.base import ProviderRuntimeOptions
from llm_annotator.clients.gemini_client import (
    GeminiClient,
    _messages_to_prompt,
)


pytestmark = pytest.mark.usefixtures("block_network")


def test_messages_to_prompt_happy_path() -> None:
    # Verifies Gemini helper extracts system instruction and user prompt.
    system, prompt = _messages_to_prompt(
        [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "question"},
        ]
    )

    assert system == "rules"
    assert prompt == "question"


def test_gemini_messages_to_prompt_error_branches() -> None:
    # Verifies Gemini helper enforces one user message and valid roles.
    with pytest.raises(ValueError):
        _messages_to_prompt([{"role": "assistant", "content": "x"}])

    with pytest.raises(ValueError):
        _messages_to_prompt(
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u1"},
                {"role": "user", "content": "u2"},
            ]
        )

    with pytest.raises(Exception):
        _messages_to_prompt(
            [
                {"role": "system", "content": "s1"},
                {"role": "system", "content": "s2"},
                {"role": "user", "content": "u"},
            ]
        )


def test_gemini_generate_builds_payload_and_parses_response(
    fake_google_genai_module: dict[str, Any],
) -> None:
    # Verifies Gemini request config mapping and response parsing.
    client = GeminiClient(model="gemini-test")

    response = client.generate(
        messages=[
            {"role": "system", "content": "Answer in JSON"},
            {"role": "user", "content": "Provide a label"},
        ],
        options=ProviderRuntimeOptions(
            max_tokens=10,
            json_schema={"type": "object"},
        ),
    )

    request = fake_google_genai_module["last_generate_kwargs"]
    assert isinstance(request, dict)
    assert request["model"] == "gemini-test"
    assert request["contents"] == "Provide a label"
    config = request["config"]
    assert getattr(config, "max_output_tokens") == 10
    assert getattr(config, "system_instruction") == "Answer in JSON"
    assert getattr(config, "response_mime_type") == "application/json"
    assert response.text == '{"label": "ok"}'
    assert response.model == "gemini-fake"


def test_gemini_generate_request_error_returns_error_response(
    fake_google_genai_module: dict[str, Any],
) -> None:
    # Verifies Gemini request failures return structured error responses.
    fake_google_genai_module["generate_raises"] = RuntimeError("api down")
    client = GeminiClient(model="gemini-test", on_error="ignore")

    response = client.generate(messages=[{"role": "user", "content": "hi"}])

    assert response.error is not None
    assert response.error_type == "ProviderError"


@pytest.mark.parametrize("stop_reason", [None, "STOP"])
def test_gemini_handle_stop_reason_success_branches(
    stop_reason: str | None,
) -> None:
    # Verifies Gemini stop-reason handler accepts success reasons.
    client = object.__new__(GeminiClient)
    client.model = "g"
    client.max_workers = 1
    client.on_error = "raise"
    client._logger = cast(
        Any,
        types.SimpleNamespace(
            warning=lambda _msg: None, debug=lambda _msg: None
        ),
    )

    client._handle_stop_reason(stop_reason=stop_reason, num_output_tokens=2)


@pytest.mark.parametrize(
    "stop_reason",
    [
        "MAX_TOKENS",
        "SAFETY",
        "MALFORMED_FUNCTION_CALL",
        "LANGUAGE",
        "NO_IMAGE",
        "OTHER",
    ],
)
def test_gemini_handle_stop_reason_error_branches(stop_reason: str) -> None:
    # Verifies Gemini stop-reason handler maps all error classes to exceptions.
    client = object.__new__(GeminiClient)
    client.model = "g"
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
            stop_reason=stop_reason, num_output_tokens=2
        )
