from __future__ import annotations

import gc
import types
from typing import Any

import pytest

from llm_annotator.clients.base import Response
from llm_annotator.clients.exceptions import ProviderError
from llm_annotator.clients.vllm_offline_client import (
    VLLMOfflineClient,
    VLLMRuntimeOptions,
)


@pytest.fixture
def fake_vllm_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {
        "llm_kwargs": None,
        "chat_calls": [],
    }

    class FakeSamplingParams:
        def __init__(self, **kwargs: object) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    class FakeStructuredOutputsParams:
        def __init__(self, **kwargs: object) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    class FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            state["llm_kwargs"] = kwargs
            self.llm_engine = types.SimpleNamespace(
                model_executor=types.SimpleNamespace(shutdown=lambda: None),
                engine_core=types.SimpleNamespace(shutdown=lambda: None),
            )

        def chat(
            self,
            messages: list[list[dict[str, str]]],
            sampling_params: object,
            use_tqdm: bool = False,
        ) -> list[object]:
            state["chat_calls"].append(
                {
                    "messages": messages,
                    "sampling_params": sampling_params,
                    "use_tqdm": use_tqdm,
                }
            )
            outputs: list[object] = []
            for idx, _ in enumerate(messages):
                outputs.append(
                    types.SimpleNamespace(
                        outputs=[
                            types.SimpleNamespace(
                                text=f" out-{idx} ",
                                token_ids=[1, 2],
                                finish_reason="stop",
                            )
                        ]
                    )
                )
            return outputs

    fake_vllm = types.ModuleType("vllm")
    fake_sampling_mod = types.ModuleType("vllm.sampling_params")
    fake_dist_mod = types.ModuleType("vllm.distributed")

    fake_vllm.LLM = FakeLLM  # type: ignore[attr-defined]
    fake_vllm.SamplingParams = FakeSamplingParams  # type: ignore[attr-defined]
    fake_sampling_mod.StructuredOutputsParams = FakeStructuredOutputsParams  # type: ignore[attr-defined]
    fake_dist_mod.destroy_distributed_environment = lambda: None  # type: ignore[attr-defined]
    fake_dist_mod.destroy_model_parallel = lambda: None  # type: ignore[attr-defined]

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)  # type: ignore[attr-defined]

    monkeypatch.setitem(__import__("sys").modules, "vllm", fake_vllm)
    monkeypatch.setitem(
        __import__("sys").modules,
        "vllm.sampling_params",
        fake_sampling_mod,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "vllm.distributed",
        fake_dist_mod,
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    return state


def test_runtime_options_to_sampling_params(fake_vllm_runtime: dict[str, Any]) -> None:
    # Verifies runtime options are translated to SamplingParams fields.
    opts = VLLMRuntimeOptions(
        max_tokens=10,
        temperature=0.1,
        top_p=0.9,
        top_k=20,
        stop=["END"],
        seed=7,
    )
    params = opts.to_sampling_params()
    assert params.max_tokens == 10
    assert params.temperature == 0.1
    assert params.top_p == 0.9
    assert params.top_k == 20
    assert params.stop == ["END"]
    assert params.seed == 7


def test_runtime_options_with_json_schema(
    fake_vllm_runtime: dict[str, Any],
) -> None:
    # Verifies structured output params are attached when json_schema is provided.
    opts = VLLMRuntimeOptions(json_schema={"type": "object"})
    params = opts.to_sampling_params()
    assert hasattr(params, "structured_outputs")


def test_load_pipeline_explicit_args_override_extras(
    fake_vllm_runtime: dict[str, Any],
) -> None:
    # Verifies explicit constructor args override conflicting extra kwargs.
    client = VLLMOfflineClient(
        model="m",
        max_model_len=512,
        extra_vllm_kwargs={"max_model_len": 128, "foo": "bar"},
    )
    kwargs = fake_vllm_runtime["llm_kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["max_model_len"] == 512
    assert kwargs["foo"] == "bar"
    client.destroy()


def test_warm_up_no_op_without_prefix(
    fake_vllm_runtime: dict[str, Any],
) -> None:
    # Verifies warm_up is a no-op when no warm-up prefix/context is supplied.
    client = VLLMOfflineClient(model="m")
    client.warm_up(system_message=None, prompt_prefix=None)
    assert fake_vllm_runtime["chat_calls"] == []
    client.destroy()


def test_warm_up_executes_with_forced_max_tokens(
    fake_vllm_runtime: dict[str, Any],
) -> None:
    # Verifies warm_up performs one chat call and forces max_tokens=1.
    client = VLLMOfflineClient(model="m")
    client.warm_up(
        system_message="sys",
        prompt_prefix="prefix",
        options=VLLMRuntimeOptions(max_tokens=99),
    )
    calls = fake_vllm_runtime["chat_calls"]
    assert isinstance(calls, list)
    last = calls[-1]
    assert last["messages"][0][0]["role"] == "system"
    assert last["messages"][0][1]["role"] == "user"
    assert last["sampling_params"].max_tokens == 1
    client.destroy()


def test_generate_delegates_to_batch_generate(
    fake_vllm_runtime: dict[str, Any],
) -> None:
    # Verifies generate delegates to batch_generate and unwraps first response.
    client = VLLMOfflineClient(model="m")
    response = client.generate(messages=[{"role": "user", "content": "x"}])
    assert isinstance(response, Response)
    assert response.text == "out-0"
    client.destroy()


def test_batch_generate_pipe_none_returns_error_response(
    fake_vllm_runtime: dict[str, Any],
) -> None:
    # Verifies missing pipeline returns one error response per input message.
    client = VLLMOfflineClient(model="m", on_error="ignore")
    client._pipe = None
    responses = client.batch_generate(
        messages=[[{"role": "user", "content": "x"}], [{"role": "user", "content": "y"}]]
    )
    assert len(responses) == 2
    assert all(r.error is not None for r in responses)


def test_batch_generate_rejects_unknown_gen_kwarg(
    fake_vllm_runtime: dict[str, Any],
) -> None:
    # Verifies unsupported sampling override keys are surfaced as errors.
    client = VLLMOfflineClient(model="m", on_error="ignore")
    responses = client.batch_generate(
        messages=[[{"role": "user", "content": "x"}]],
        gen_kwargs={"unknown": 1},
    )
    assert responses[0].error is not None
    client.destroy()


def test_batch_generate_pad_when_fewer_outputs(
    fake_vllm_runtime: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Verifies response list is padded with errors when model returns too few outputs.
    client = VLLMOfflineClient(model="m", on_error="ignore")

    def _chat_short(
        _messages: list[list[dict[str, str]]], _sampling: object, use_tqdm: bool = False
    ) -> list[object]:
        _ = use_tqdm
        return [
            types.SimpleNamespace(
                outputs=[
                    types.SimpleNamespace(
                        text="ok",
                        token_ids=[1],
                        finish_reason="stop",
                    )
                ]
            )
        ]

    monkeypatch.setattr(client._pipe, "chat", _chat_short)
    responses = client.batch_generate(
        messages=[[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    )
    assert len(responses) == 2
    assert responses[0].error is None
    assert responses[1].error is not None
    client.destroy()


def test_process_response_and_stop_reason_error(
    fake_vllm_runtime: dict[str, Any],
) -> None:
    # Verifies output text/token parsing and stop-reason error handling branches.
    client = VLLMOfflineClient(model="m")
    output = types.SimpleNamespace(
        outputs=[
            types.SimpleNamespace(
                text="  hi  ", token_ids=[1, 2, 3], finish_reason="stop"
            )
        ]
    )
    resp = client._process_response(output)  # type: ignore[arg-type]
    assert resp.text == "hi"
    assert resp.num_output_tokens == 3

    with pytest.raises(ProviderError, match="hit the configured output token"):
        client._handle_stop_reason(stop_reason="length", num_output_tokens=4)
    with pytest.raises(ProviderError, match="aborted"):
        client._handle_stop_reason(stop_reason="abort", num_output_tokens=4)
    with pytest.raises(ProviderError, match="unexpected reason"):
        client._handle_stop_reason(stop_reason="weird", num_output_tokens=None)
    client.destroy()


def test_destroy_is_idempotent(
    fake_vllm_runtime: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Verifies destroy can be called repeatedly without side effects.
    collected = {"called": 0}

    def _collect() -> int:
        collected["called"] += 1
        return 0

    monkeypatch.setattr(gc, "collect", _collect)

    client = VLLMOfflineClient(model="m")
    client.destroy()
    client.destroy()
    assert client._pipe is None
    assert collected["called"] >= 1
