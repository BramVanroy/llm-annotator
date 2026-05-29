from __future__ import annotations

import types
from typing import Any, Literal, cast

import pytest

from llm_annotator.clients.base import (
    Client,
    Provider,
    ProviderRuntimeOptions,
    Response,
)
from llm_annotator.clients.exceptions import ProviderError


class DummyClient(Client[ProviderRuntimeOptions]):
    provider_type = Provider.OPENAI

    def __init__(
        self,
        model: str = "demo",
        on_error: Literal["raise", "ignore", "warn"] = "warn",
    ) -> None:
        super().__init__(model=model, on_error=on_error)
        self.destroy_called = 0
        self.generate_calls: list[tuple[list[dict[str, str]], Any, Any]] = []

    def _process_response(self, response: Any) -> Response:
        return Response(text=str(response), provider=self.provider_type)

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        options: ProviderRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> Response:
        self.generate_calls.append((messages, options, gen_kwargs))
        return Response(
            text=messages[-1]["content"], provider=self.provider_type
        )

    def _handle_stop_reason(
        self, *, stop_reason: str | None, num_output_tokens: int | None
    ) -> None:
        _ = stop_reason
        _ = num_output_tokens

    def destroy(self) -> None:
        self.destroy_called += 1


def test_provider_runtime_options_default_payload() -> None:
    # Verifies the base runtime options serialize to an empty payload.
    assert ProviderRuntimeOptions().to_payload() == {}


def test_client_handle_error_warns_and_preserves_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Verifies partial responses are preserved when errors are downgraded.
    client = DummyClient(on_error="warn")
    warnings: list[str] = []
    client._logger = cast(Any, types.SimpleNamespace(warning=warnings.append))

    partial = Response(
        text="partial",
        stop_reason="stop",
        model="alt-model",
        provider=Provider.CLAUDE,
        num_output_tokens=3,
        full_response={"raw": True},
    )

    response = client._handle_error(
        ValueError("boom"), context="demo context", partial=partial
    )

    assert warnings == ["demo context: boom"]
    assert response.text == "partial"
    assert response.stop_reason == "stop"
    assert response.model == "alt-model"
    assert response.provider == Provider.CLAUDE
    assert response.num_output_tokens == 3
    assert response.full_response == {"raw": True}
    assert response.error == "demo context: boom"
    assert response.error_type == "ProviderError"


def test_client_handle_error_raise_and_context_manager() -> None:
    # Verifies raise mode propagates ProviderError and context manager cleanup.
    client = DummyClient(on_error="raise")

    with pytest.raises(ProviderError, match="demo context: boom"):
        client._handle_error(ValueError("boom"), context="demo context")

    with client as entered:
        assert entered is client

    assert client.destroy_called == 1


def test_client_batch_generate_defaults_to_generate() -> None:
    # Verifies the default batch implementation delegates to generate for each input.
    client = DummyClient()
    responses = client.batch_generate(
        messages=[
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ],
        options=ProviderRuntimeOptions(max_tokens=2),
        gen_kwargs={"temperature": 0.1},
    )

    assert [response.text for response in responses] == ["first", "second"]
    assert len(client.generate_calls) == 2
    assert client.generate_calls[0][1] == ProviderRuntimeOptions(max_tokens=2)
    assert client.generate_calls[0][2] == {"temperature": 0.1}
