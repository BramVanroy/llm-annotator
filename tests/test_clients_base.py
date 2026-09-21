from __future__ import annotations

import types
from typing import Any, Literal, cast

import pytest

from llm_annotator.clients.base import (
    Client,
    Provider,
    ProviderRuntimeOptions,
    Response,
    reject_multiple_responses,
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
        reasoning="half a thought",
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
    assert response.reasoning == "half a thought"
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
        options=ProviderRuntimeOptions(max_completion_tokens=2),
        gen_kwargs={"temperature": 0.1},
    )

    assert [response.text for response in responses] == ["first", "second"]
    assert len(client.generate_calls) == 2
    assert client.generate_calls[0][1] == ProviderRuntimeOptions(
        max_completion_tokens=2
    )
    assert client.generate_calls[0][2] == {"temperature": 0.1}


class FlakyClient(DummyClient):
    """Dummy client whose ``generate`` fails for one marked conversation."""

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        options: ProviderRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> Response:
        if messages[-1]["content"] == "bad":
            raise RuntimeError("boom")
        return DummyClient.generate(
            self, messages=messages, options=options, gen_kwargs=gen_kwargs
        )


@pytest.mark.parametrize("max_workers", [None, 0, 1, 4])
def test_generate_in_threads_keeps_order_and_isolates_failures(
    max_workers: int | None,
) -> None:
    # Verifies one Response per input, in input order, at every worker count.
    client = FlakyClient(on_error="ignore")

    responses = client._generate_in_threads(
        messages=[
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "bad"}],
            [{"role": "user", "content": "third"}],
        ],
        options=None,
        gen_kwargs=None,
        max_workers=max_workers,
        context="demo request failed",
    )

    assert [response.text for response in responses] == ["first", "", "third"]
    assert responses[0].error is None
    assert responses[1].error == "demo request failed at index 1: boom"
    assert responses[2].error is None


def test_generate_in_threads_raises_when_on_error_is_raise() -> None:
    # Verifies raise mode surfaces a worker failure as a ProviderError.
    client = FlakyClient(on_error="raise")

    with pytest.raises(ProviderError, match="at index 0: boom"):
        client._generate_in_threads(
            messages=[[{"role": "user", "content": "bad"}]],
            options=None,
            gen_kwargs=None,
            max_workers=4,
            context="demo request failed",
        )


def test_reject_multiple_responses_accepts_single_response() -> None:
    # Verifies a payload without 'n', or with n=1, passes the check.
    reject_multiple_responses({})
    reject_multiple_responses({"n": 1})
    reject_multiple_responses({"extra_body": {"n": 1}})


@pytest.mark.parametrize(
    "payload", [{"n": 2}, {"extra_body": {"n": 2}}, {"n": 0}]
)
def test_reject_multiple_responses_rejects_other_counts(
    payload: dict[str, Any],
) -> None:
    # Verifies any 'n' other than 1 is rejected, at either level.
    with pytest.raises(ValueError, match="one response per sample"):
        reject_multiple_responses(payload)
