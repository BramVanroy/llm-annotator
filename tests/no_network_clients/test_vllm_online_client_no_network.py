from __future__ import annotations

from typing import Any, cast

import pytest

from llm_annotator.clients.exceptions import ConfigurationError
from llm_annotator.clients.vllm_online_client import (
    VLLMOnlineClient,
    VLLMOnlineRuntimeOptions,
)


pytestmark = pytest.mark.usefixtures("block_network")


def test_vllm_online_client_uses_listed_model_when_none_given(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies VLLM client auto-selects first served model when model is None.
    fake_openai_module["model_list"] = ["served-vllm-model"]
    client = VLLMOnlineClient(model=None)

    assert client.model == "served-vllm-model"


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
        json_schema={"type": "object"},
    ).to_payload()

    assert base_payload["top_k"] == 4
    assert base_payload["repetition_penalty"] == 1.1
    assert base_payload["max_completion_tokens"] == 8
    assert base_payload["add_generation_prompt"] is False
    assert base_payload["chat_template"] == "tmpl"
    assert base_payload["chat_template_kwargs"] == {"foo": "bar"}
    assert base_payload["mm_processor_kwargs"] == {"num_crops": 4}


def test_vllm_online_batch_generate_rejects_openai_batch_api(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies the OpenAI Batch API guard raises a configuration error.
    _ = fake_openai_module
    client = VLLMOnlineClient(model="served-vllm-model")

    with pytest.raises(ConfigurationError, match="does not support"):
        client.batch_generate(
            messages=[[{"role": "user", "content": "one"}]],
            use_batch_api=True,
        )


def test_vllm_online_batch_generate_uses_batch_endpoint(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies vLLM batch API endpoint and response mapping.
    fake_openai_module["post_json"] = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "A"},
            },
            {
                "index": 1,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "B"},
            },
        ]
    }
    client = VLLMOnlineClient(model="served-vllm-model")

    responses = client.batch_generate(
        messages=[
            [{"role": "user", "content": "one"}],
            [{"role": "user", "content": "two"}],
        ],
        options=VLLMOnlineRuntimeOptions(max_completion_tokens=9, top_k=20),
    )

    assert fake_openai_module["last_post_url"] == (
        "http://localhost:8000/v1/chat/completions/batch"
    )
    post_payload = fake_openai_module["last_post_json"]
    assert isinstance(post_payload, dict)
    assert post_payload["top_k"] == 20
    assert len(responses) == 2
    assert responses[0].text == "A"
    assert responses[1].text == "B"


def test_vllm_online_batch_generate_includes_json_schema(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies json_schema is forwarded into the vLLM batch request body.
    fake_openai_module["post_json"] = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "A"},
            }
        ]
    }
    client = VLLMOnlineClient(model="served-vllm-model")

    responses = client.batch_generate(
        messages=[[{"role": "user", "content": "one"}]],
        options=VLLMOnlineRuntimeOptions(json_schema={"type": "object"}),
    )

    post_payload = fake_openai_module["last_post_json"]
    assert isinstance(post_payload, dict)
    assert post_payload["response_format"]["type"] == "json_schema"
    assert post_payload["response_format"]["json_schema"]["strict"] is True
    assert len(responses) == 1
    assert responses[0].text == "A"


def test_vllm_online_batch_generate_pads_missing_choices(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies vLLM batch responses are padded with errors when choices are missing.
    fake_openai_module["post_json"] = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "A"},
            }
        ]
    }
    client = VLLMOnlineClient(model="served-vllm-model", on_error="ignore")

    responses = client.batch_generate(
        messages=[
            [{"role": "user", "content": "one"}],
            [{"role": "user", "content": "two"}],
        ],
        options=VLLMOnlineRuntimeOptions(max_completion_tokens=9),
    )

    assert len(responses) == 2
    assert responses[0].text == "A"
    assert responses[1].error is not None


def test_vllm_online_batch_generate_http_error_returns_error_responses(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies vLLM batch HTTP errors are mapped to one error response per input.
    class FailingHTTPResponse:
        def raise_for_status(self) -> None:
            raise RuntimeError("http error")

        def json(self) -> dict[str, object]:
            return {"choices": []}

    class FailingHTTPClient:
        def post(
            self, url: str, json: dict[str, object]
        ) -> FailingHTTPResponse:
            _ = url
            _ = json
            return FailingHTTPResponse()

    _ = fake_openai_module
    client = VLLMOnlineClient(model="served-vllm-model", on_error="ignore")
    cast(Any, client._client)._client = FailingHTTPClient()

    responses = client.batch_generate(
        messages=[
            [{"role": "user", "content": "one"}],
            [{"role": "user", "content": "two"}],
        ],
        options=VLLMOnlineRuntimeOptions(max_completion_tokens=4),
    )
    assert len(responses) == 2
    assert all(r.error is not None for r in responses)


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
    fake_openai_module["post_json"] = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }
        ]
    }
    client = VLLMOnlineClient(model="m")
    client.batch_generate(
        messages=[[{"role": "user", "content": "hi"}]],
        options=VLLMOnlineRuntimeOptions(
            temperature=0.7, extra_body={"min_p": 0.1}
        ),
        gen_kwargs={"temperature": 0.0, "priority": 1},
    )

    payload = cast(dict[str, Any], fake_openai_module["last_post_json"])
    assert payload["min_p"] == 0.1
    assert payload["priority"] == 1
    # gen_kwargs is documented as taking precedence over options.
    assert payload["temperature"] == 0.0


def test_vllm_online_batch_generate_reads_reasoning(
    fake_openai_module: dict[str, Any],
) -> None:
    # Verifies the trace survives the batch endpoint, which is the path the
    # annotator actually uses and which names the field `reasoning`, not
    # `reasoning_content`. A server started with --reasoning-parser returns
    # it there for every choice.
    fake_openai_module["post_json"] = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Antwerpen",
                    "reasoning": "The article names Antwerpen.",
                },
            },
            {
                "index": 1,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Gent"},
            },
        ]
    }
    client = VLLMOnlineClient(model="served-vllm-model")

    responses = client.batch_generate(
        messages=[
            [{"role": "user", "content": "one"}],
            [{"role": "user", "content": "two"}],
        ],
    )

    assert responses[0].reasoning == "The article names Antwerpen."
    assert responses[0].text == "Antwerpen"
    # A server without a reasoning parser returns no such field at all.
    assert responses[1].reasoning is None
    assert responses[1].text == "Gent"
