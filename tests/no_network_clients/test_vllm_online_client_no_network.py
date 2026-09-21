from __future__ import annotations

import logging
import threading
import urllib.error
from typing import Any, cast

import pytest

from llm_annotator.clients.base import Response
from llm_annotator.clients.openai_client import CONNECT_TIMEOUT
from llm_annotator.clients.vllm_online_client import (
    MAX_CONNECTIONS,
    VLLMOnlineClient,
    VLLMOnlineRuntimeOptions,
    server_is_healthy,
)


pytestmark = pytest.mark.usefixtures("block_network")


def conversations(*contents: str) -> list[list[dict[str, str]]]:
    """Build one single-turn conversation per given user message."""
    return [[{"role": "user", "content": text}] for text in contents]


def test_vllm_online_client_uses_listed_model_when_none_given(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies VLLM client auto-selects first served model when model is None.
    fake_openai_module["model_list"] = ["served-vllm-model"]
    client = VLLMOnlineClient(model=None)

    assert client.model == "served-vllm-model"


def test_vllm_online_client_sets_base_url(
    fake_openai_module: dict[str, Any],
) -> None:
    _ = fake_openai_module
    client = VLLMOnlineClient(
        model="served-vllm-model", base_url="http://worker:8000/v1"
    )

    assert client.base_url == "http://worker:8000/v1"


def test_vllm_online_client_configures_the_sdk_client(
    fake_openai_module: dict[str, Any],
) -> None:
    """The SDK gets the long timeout and the raised connection limit.

    The SDK's own defaults (600 s, 1000 sockets) are cut for a hosted API and
    are both too small for a pool that holds thousands of prompts per server.
    """
    client = VLLMOnlineClient(
        model="served-vllm-model", timeout=120.0, max_retries=1
    )

    assert client.timeout == 120.0
    assert client.max_retries == 1
    sdk_kwargs = cast(list[Any], fake_openai_module["openai_init_kwargs"])[-1]
    assert sdk_kwargs["max_retries"] == 1
    assert sdk_kwargs["api_key"] == "EMPTY"
    # A plain float would make httpx wait `timeout` on the connection too,
    # so a black-holed server would stall the batch instead of erroring.
    assert sdk_kwargs["timeout"].read == 120.0
    assert sdk_kwargs["timeout"].connect == CONNECT_TIMEOUT
    limits = sdk_kwargs["http_client"].kwargs["limits"]
    assert limits.max_connections == MAX_CONNECTIONS
    assert limits.max_keepalive_connections == MAX_CONNECTIONS


def test_vllm_online_client_defaults_fit_a_loaded_server(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies the defaults documented in the constructor docstring.
    _ = fake_openai_module
    client = VLLMOnlineClient(model="served-vllm-model")

    assert client.timeout == 3600.0
    assert client.max_retries == 2
    assert client.max_workers is None


def test_server_is_healthy_logs_the_probe_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        _ = args
        _ = kwargs
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with caplog.at_level(
        logging.DEBUG, logger="llm_annotator.clients.vllm_online"
    ):
        assert not server_is_healthy("http://a:8000/v1", 1)

    assert any(
        "connection refused" in record.message for record in caplog.records
    )


def test_vllm_online_client_is_healthy_delegates_to_the_probe(
    monkeypatch: pytest.MonkeyPatch,
    fake_openai_module: dict[str, Any],
) -> None:
    _ = fake_openai_module
    seen: dict[str, Any] = {}

    def fake_probe(base_url: str, timeout: float) -> bool:
        seen["base_url"] = base_url
        seen["timeout"] = timeout
        return True

    monkeypatch.setattr(
        "llm_annotator.clients.vllm_online_client.server_is_healthy",
        fake_probe,
    )
    client = VLLMOnlineClient(
        model="served-vllm-model", base_url="http://worker:8000/v1"
    )

    assert client.is_healthy(timeout=2) is True
    assert seen == {"base_url": "http://worker:8000/v1", "timeout": 2}


def test_vllm_online_runtime_options_to_payload() -> None:
    # Verifies the shared and server-specific runtime options serialize correctly.
    base_payload = VLLMOnlineRuntimeOptions(
        max_completion_tokens=8,
        top_k=4,
        repetition_penalty=1.1,
        add_generation_prompt=False,
        chat_template="tmpl",
        chat_template_kwargs={"foo": "bar"},
        mm_processor_kwargs={"num_crops": 4},
    ).to_payload()

    assert base_payload["top_k"] == 4
    assert base_payload["repetition_penalty"] == 1.1
    assert base_payload["max_completion_tokens"] == 8
    assert base_payload["add_generation_prompt"] is False
    assert base_payload["chat_template"] == "tmpl"
    assert base_payload["chat_template_kwargs"] == {"foo": "bar"}
    assert base_payload["mm_processor_kwargs"] == {"num_crops": 4}
    assert "response_format" not in base_payload


def test_vllm_online_runtime_options_render_the_json_schema() -> None:
    """vLLM 0.29 reads a schema from ``response_format``, not elsewhere.

    ``structured_outputs_from_response_format`` in
    ``vllm/entrypoints/generate/base/protocol.py`` turns exactly this shape
    into the engine's ``StructuredOutputsParams(json=...)``.
    """
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    payload = VLLMOnlineRuntimeOptions(output_schema=schema).to_payload()

    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": schema,
            "strict": True,
        },
    }


def test_vllm_online_generate_sends_the_json_schema(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies response_format survives the split into typed SDK kwargs.
    client = VLLMOnlineClient(model="served-vllm-model")
    client.generate(
        messages=[{"role": "user", "content": "one"}],
        options=VLLMOnlineRuntimeOptions(output_schema={"type": "object"}),
    )

    kwargs = cast(dict[str, Any], fake_openai_module["last_create_kwargs"])
    assert kwargs["response_format"]["json_schema"]["schema"] == {
        "type": "object"
    }
    assert "response_format" not in kwargs.get("extra_body", {})


def test_vllm_online_batch_generate_answers_each_conversation_on_its_own(
    fake_openai_module: dict[str, Any],
) -> None:
    """One request per conversation, each with its own text and usage.

    The batch route reported a single ``usage`` for the whole batch, which
    left ``num_output_tokens`` empty; one request per conversation fills it.
    """
    fake_openai_module["create_responses"] = {
        "one": {"content": "A", "completion_tokens": 3},
        "two": {"content": "B", "completion_tokens": 5},
    }
    client = VLLMOnlineClient(model="served-vllm-model")

    responses = client.batch_generate(
        messages=conversations("one", "two"),
        options=VLLMOnlineRuntimeOptions(max_completion_tokens=9, top_k=20),
    )

    assert [response.text for response in responses] == ["A", "B"]
    assert [response.num_output_tokens for response in responses] == [3, 5]
    calls = cast(list[Any], fake_openai_module["create_calls"])
    assert len(calls) == 2
    assert all(call["extra_body"]["top_k"] == 20 for call in calls)
    assert all(call["max_completion_tokens"] == 9 for call in calls)


def test_vllm_online_batch_generate_sends_the_whole_batch_at_once(
    fake_openai_module: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every conversation of a batch reaches the server together.

    The server does the scheduling, so holding requests back would starve it.
    The barrier only releases once all four requests are in flight, and a
    smaller pool would leave it waiting until it breaks.
    """
    _ = fake_openai_module
    client = VLLMOnlineClient(model="served-vllm-model")
    barrier = threading.Barrier(4, timeout=30)

    def blocking_generate(
        *,
        messages: list[dict[str, str]],
        options: Any = None,
        gen_kwargs: Any = None,
    ) -> Response:
        _ = options
        _ = gen_kwargs
        barrier.wait()
        return Response(text=messages[-1]["content"])

    monkeypatch.setattr(client, "generate", blocking_generate)

    responses = client.batch_generate(
        messages=conversations("0", "1", "2", "3")
    )

    assert [response.text for response in responses] == ["0", "1", "2", "3"]


def test_vllm_online_max_workers_caps_the_concurrency(
    fake_openai_module: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Verifies a user-set cap is honoured for a server shared with other jobs.
    _ = fake_openai_module
    client = VLLMOnlineClient(model="served-vllm-model", max_workers=1)
    threads: list[str] = []

    def recording_generate(
        *,
        messages: list[dict[str, str]],
        options: Any = None,
        gen_kwargs: Any = None,
    ) -> Response:
        _ = options
        _ = gen_kwargs
        threads.append(threading.current_thread().name)
        return Response(text=messages[-1]["content"])

    monkeypatch.setattr(client, "generate", recording_generate)

    responses = client.batch_generate(
        messages=conversations("0", "1", "2", "3")
    )

    assert [response.text for response in responses] == ["0", "1", "2", "3"]
    assert len(set(threads)) == 1


def test_vllm_online_batch_generate_isolates_a_failed_conversation(
    fake_openai_module: dict[str, Any],
) -> None:
    """A request that fails costs its own sample, not the whole batch.

    vLLM's batch route failed the entire request instead
    (``_raise_if_error`` in
    ``vllm/entrypoints/openai/chat_completion/batch_serving.py``).
    """
    fake_openai_module["create_responses"] = {
        "one": {"content": "A"},
        "two": {"raises": RuntimeError("context length exceeded")},
        "three": {"content": "C"},
    }
    client = VLLMOnlineClient(model="served-vllm-model", on_error="ignore")

    responses = client.batch_generate(
        messages=conversations("one", "two", "three")
    )

    assert [response.text for response in responses] == ["A", "", "C"]
    assert responses[0].error is None
    assert responses[2].error is None
    assert "context length exceeded" in cast(str, responses[1].error)
    assert responses[1].error_type == "ProviderError"


def test_vllm_online_batch_generate_errors_every_sample_when_the_server_is_gone(
    fake_openai_module: dict[str, Any],
) -> None:
    """A dead server errors every sample, which is what eviction looks for.

    [`VLLMQueueAnnotator`][llm_annotator.annotator.VLLMQueueAnnotator] probes
    ``/health`` only once every sample of a batch carries an error, so a
    connection failure has to reach all of them.
    """
    fake_openai_module["create_raises"] = ConnectionError("connection refused")
    client = VLLMOnlineClient(model="served-vllm-model", on_error="ignore")

    responses = client.batch_generate(messages=conversations("one", "two"))

    assert len(responses) == 2
    assert all(response.error is not None for response in responses)
    assert all(
        response.error_type == "ProviderError" for response in responses
    )


def test_vllm_online_batch_generate_reads_reasoning(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies the trace survives the batch path, which names the field
    # `reasoning`, not `reasoning_content`. A server started with
    # --reasoning-parser returns it for every request.
    fake_openai_module["create_responses"] = {
        "one": {
            "content": "Antwerpen",
            "reasoning": "The article names Antwerpen.",
        },
        "two": {"content": "Gent"},
    }
    client = VLLMOnlineClient(model="served-vllm-model")

    responses = client.batch_generate(messages=conversations("one", "two"))

    assert responses[0].reasoning == "The article names Antwerpen."
    assert responses[0].text == "Antwerpen"
    # A server without a reasoning parser returns no such field at all.
    assert responses[1].reasoning is None
    assert responses[1].text == "Gent"


def test_vllm_online_generate_nests_vllm_extensions_in_extra_body(
    fake_openai_module: dict[str, Any],
) -> None:
    """The typed SDK signature only accepts OpenAI's own parameters.

    vLLM's extensions have to travel inside ``extra_body`` or ``create()``
    rejects them, which is why this client does not inherit OpenAI's
    ``generate``.
    """
    client = VLLMOnlineClient(model="m")
    client.generate(
        messages=[{"role": "user", "content": "hi"}],
        options=VLLMOnlineRuntimeOptions(
            temperature=0.0,
            top_k=4,
            chat_template_kwargs={"enable_thinking": False},
            extra_body={"min_p": 0.1},
        ),
    )

    kwargs = cast(dict[str, Any], fake_openai_module["last_create_kwargs"])
    assert kwargs["temperature"] == 0.0
    extra_body = kwargs["extra_body"]
    assert extra_body["top_k"] == 4
    assert extra_body["chat_template_kwargs"] == {"enable_thinking": False}
    assert extra_body["min_p"] == 0.1
    # Nothing vLLM-only may leak into the typed keyword arguments.
    assert not {"top_k", "chat_template_kwargs", "min_p"} & set(kwargs)


def test_vllm_online_extra_body_and_gen_kwargs_reach_the_request(
    fake_openai_module: dict[str, Any],
) -> None:
    """Both escape hatches land in the body the server actually receives."""
    client = VLLMOnlineClient(model="m")
    client.batch_generate(
        messages=conversations("hi"),
        options=VLLMOnlineRuntimeOptions(
            temperature=0.7, extra_body={"min_p": 0.1}
        ),
        gen_kwargs={"temperature": 0.0, "priority": 1},
    )

    kwargs = cast(dict[str, Any], fake_openai_module["last_create_kwargs"])
    assert kwargs["extra_body"]["min_p"] == 0.1
    assert kwargs["priority"] == 1
    # gen_kwargs is documented as taking precedence over options.
    assert kwargs["temperature"] == 0.0


def test_vllm_online_client_builds_one_sdk_client(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies the constructor builds exactly one SDK client, the pooled one.
    VLLMOnlineClient(model="served-vllm-model")

    inits = cast(list[Any], fake_openai_module["openai_init_kwargs"])
    assert len(inits) == 1
    assert inits[0]["http_client"] is not None
