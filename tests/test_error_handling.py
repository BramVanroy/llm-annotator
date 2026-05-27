from __future__ import annotations

import pytest

from llm_annotator.annotator import Annotator
from llm_annotator.clients.base import (
    Client,
    Provider,
    ProviderRuntimeOptions,
    Response,
)
from llm_annotator.logging_utils import configure_logging, get_logger


class DummyClient(Client[ProviderRuntimeOptions]):
    provider_type = Provider.OPENAI

    def _process_response(self, response: str) -> Response:
        return Response(
            text=response,
            provider=self.provider_type,
            model=self.model,
        )

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        options: ProviderRuntimeOptions | None = None,
        gen_kwargs: dict | None = None,
    ) -> Response:
        content = messages[0]["content"]
        if content == "fail":
            return self._handle_error(
                RuntimeError("boom"),
                context="dummy generation failed",
            )
        return Response(
            text=content,
            provider=self.provider_type,
            model=self.model,
        )

    def _handle_stop_reason(
        self, *, stop_reason: str | None, num_output_tokens: int | None
    ) -> None:
        return None


def test_default_on_error_ignores() -> None:
    # Verifies default on_error policy raises ProviderError on failures.
    client = DummyClient(model="dummy")

    # Will not trigger an error since on_error defaults to 'ignore'.
    client.generate(messages=[{"role": "user", "content": "fail"}])


def test_on_error_ignore_returns_error_response() -> None:
    # Verifies on_error='ignore' returns an error Response instead of raising.
    client = DummyClient(model="dummy", on_error="ignore")

    response = client.generate(messages=[{"role": "user", "content": "fail"}])

    assert response.error is not None
    assert response.error_type == "ProviderError"
    assert response.text == ""


def test_on_error_warn_logs_and_returns_error_response(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Verifies on_error='warn' logs warning text and returns an error Response.
    configure_logging(enabled=True, level="WARNING", style="plain")
    _ = get_logger("clients.openai")
    client = DummyClient(model="dummy", on_error="warn")
    response = client.generate(messages=[{"role": "user", "content": "fail"}])
    captured = capsys.readouterr()

    assert response.error is not None
    assert "dummy generation failed" in captured.err


def test_annotator_process_batch_keeps_error_rows() -> None:
    # Verifies Annotator keeps failed rows in batch output with error metadata.
    client = DummyClient(model="dummy", on_error="ignore")
    annotator = Annotator(client=client)

    batch = {
        "messages": [
            [{"role": "user", "content": "ok"}],
            [{"role": "user", "content": "fail"}],
        ]
    }

    results = annotator._process_batch(batch=batch, options=None)

    assert len(results) == 2
    assert results[0]["error"] is None
    assert results[1]["error"] is not None
    assert results[1]["response"] == ""
