from __future__ import annotations

import gc
import sys
import types
from typing import Any

import pytest

from llm_annotator.clients.base import Response
from llm_annotator.clients.exceptions import ProviderError
from llm_annotator.clients.vllm_offline_client import (
    VLLMOfflineClient,
    VLLMOfflineRuntimeOptions,
)


@pytest.fixture
def fake_vllm_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {
        "llm_kwargs": None,
        "chat_calls": [],
    }

    _VALID_SAMPLING_KEYS = frozenset(
        {
            "n",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "structured_outputs",
        }
    )

    class FakeSamplingParams:
        def __init__(self, **kwargs: object) -> None:
            unknown = set(kwargs) - _VALID_SAMPLING_KEYS
            if unknown:
                raise TypeError(
                    f"FakeSamplingParams got unexpected kwargs: {unknown}"
                )
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

        def get_tokenizer(self) -> str:
            return "tok"

        def chat(
            self,
            messages: list[list[dict[str, str]]],
            sampling_params: object,
            chat_template_kwargs: dict[str, object] | None = None,
            use_tqdm: bool = False,
        ) -> list[object]:
            state["chat_calls"].append(
                {
                    "messages": messages,
                    "sampling_params": sampling_params,
                    "chat_template_kwargs": chat_template_kwargs,
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

    state["distributed"] = fake_dist_mod
    return state


def test_runtime_options_to_payload(
    fake_vllm_runtime: dict[str, Any],
) -> None:
    # Verifies runtime options are translated to SamplingParams-compatible dict fields.
    opts = VLLMOfflineRuntimeOptions(
        max_completion_tokens=10,
        temperature=0.1,
        top_p=0.9,
        top_k=20,
        stop=["END"],
        seed=7,
    )
    payload = opts.to_payload()
    assert payload["max_tokens"] == 10
    assert payload["temperature"] == 0.1
    assert payload["top_p"] == 0.9
    assert payload["top_k"] == 20
    assert payload["stop"] == ["END"]
    assert payload["seed"] == 7


def test_runtime_options_with_json_schema(
    fake_vllm_runtime: dict[str, Any],
) -> None:
    # Verifies structured output params are attached when json_schema is provided.
    opts = VLLMOfflineRuntimeOptions(json_schema={"type": "object"})
    payload = opts.to_payload()
    assert "structured_outputs" in payload


def test_ensure_pipeline_loaded_explicit_args_override_extras(
    fake_vllm_runtime: dict[str, Any],
) -> None:
    # Verifies explicit constructor args override conflicting extra kwargs.
    client = VLLMOfflineClient(
        model="m",
        max_model_len=512,
        extra_vllm_kwargs={"max_model_len": 128, "foo": "bar"},
    )
    client._ensure_pipeline_loaded()
    kwargs = fake_vllm_runtime["llm_kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["max_model_len"] == 512
    assert kwargs["foo"] == "bar"
    client.destroy()


def test_split_reasoning_without_a_parser_leaves_the_text_alone(
    fake_vllm_runtime: dict[str, Any],
) -> None:
    # Verifies the trace stays inline when no reasoning parser is configured,
    # which is what vllm.LLM returns on its own.
    _ = fake_vllm_runtime
    client = VLLMOfflineClient(model="m")

    assert client._split_reasoning(" <think>hm</think> answer ") == (
        None,
        "<think>hm</think> answer",
    )


def test_split_reasoning_uses_the_configured_parser(
    fake_vllm_runtime: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Verifies a configured parser is built once, from the engine's tokenizer,
    # and splits the trace off the answer.
    _ = fake_vllm_runtime
    built: list[Any] = []

    class FakeParser:
        def __init__(self, tokenizer: Any) -> None:
            built.append(tokenizer)

        def extract_reasoning(
            self, text: str, request: Any
        ) -> tuple[str | None, str | None]:
            reasoning, _, content = text.partition("</think>")
            return reasoning.replace("<think>", ""), content

        def is_reasoning_end(self, token_ids: list[int]) -> bool:
            # Stands in for a template that closed the block in the prompt.
            return token_ids == [99]

    # The fake vllm of `fake_vllm_runtime` is a module, not a package, so the
    # submodule the client imports has to be registered by hand.
    monkeypatch.setitem(
        sys.modules,
        "vllm.reasoning",
        types.SimpleNamespace(
            ReasoningParserManager=types.SimpleNamespace(
                get_reasoning_parser=lambda name: FakeParser
            )
        ),
    )
    client = VLLMOfflineClient(model="m", reasoning_parser="qwen3")

    assert client._split_reasoning("<think>hm</think> answer") == (
        "hm",
        "answer",
    )
    # Second call reuses the parser instead of rebuilding it per sample.
    client._split_reasoning("<think>again</think> more")
    assert built == ["tok"]


def test_split_reasoning_skips_parsing_when_the_prompt_ended_thinking(
    fake_vllm_runtime: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Verifies a step run with thinking off keeps its answer. The parser sees
    # no closing tag in such output and would otherwise call the whole answer
    # reasoning, leaving the response empty.
    _ = fake_vllm_runtime

    class FakeParser:
        def __init__(self, tokenizer: Any) -> None:
            pass

        def extract_reasoning(
            self, text: str, request: Any
        ) -> tuple[str | None, str | None]:
            return text, None

        def is_reasoning_end(self, token_ids: list[int]) -> bool:
            return token_ids == [99]

    monkeypatch.setitem(
        sys.modules,
        "vllm.reasoning",
        types.SimpleNamespace(
            ReasoningParserManager=types.SimpleNamespace(
                get_reasoning_parser=lambda name: FakeParser
            )
        ),
    )
    client = VLLMOfflineClient(model="m", reasoning_parser="qwen3")

    # Prompt closed the thinking block: the answer survives untouched.
    assert client._split_reasoning('{"rating": 4}', [99]) == (
        None,
        '{"rating": 4}',
    )
    # Prompt left it open: the parser decides, as before.
    assert client._split_reasoning("still thinking", [1]) == (
        "still thinking",
        "",
    )


def test_reasoning_parser_is_not_passed_to_the_engine(
    fake_vllm_runtime: dict[str, Any],
) -> None:
    # Verifies the parser name stays with the client: vllm.LLM does not take
    # it, so forwarding it would break engine construction.
    client = VLLMOfflineClient(model="m", reasoning_parser="qwen3")
    client._ensure_pipeline_loaded()

    assert "reasoning_parser" not in fake_vllm_runtime["llm_kwargs"]
    client.destroy()


def test_warm_up_no_op_without_prefix(
    fake_vllm_runtime: dict[str, Any],
) -> None:
    # Verifies warm_up is a no-op when no warm-up prefix/context is supplied.
    client = VLLMOfflineClient(model="m")
    client.warm_up(system_message=None, prompt_prefix=None)
    assert fake_vllm_runtime["chat_calls"] == []
    # Pipeline should not have been loaded (lazy loading: no-op warm_up skips load).
    assert fake_vllm_runtime["llm_kwargs"] is None
    client.destroy()


def test_warm_up_enables_prefix_caching_when_prefix_given(
    fake_vllm_runtime: dict[str, Any],
) -> None:
    # Verifies that prefix caching and chunked prefill are enabled automatically
    # before the engine is loaded when a prompt prefix is supplied.
    client = VLLMOfflineClient(
        model="m",
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    assert not client._enable_prefix_caching
    assert not client._enable_chunked_prefill
    client.warm_up(prompt_prefix="Classify the following:")
    assert client._enable_prefix_caching
    assert client._enable_chunked_prefill
    assert fake_vllm_runtime["llm_kwargs"]["enable_prefix_caching"] is True
    assert fake_vllm_runtime["llm_kwargs"]["enable_chunked_prefill"] is True
    client.destroy()


def test_warm_up_executes_with_forced_max_tokens(
    fake_vllm_runtime: dict[str, Any],
) -> None:
    # Verifies warm_up performs one chat call and forces max_tokens=1.
    client = VLLMOfflineClient(model="m")
    client.warm_up(
        system_message="sys",
        prompt_prefix="prefix",
        options=VLLMOfflineRuntimeOptions(max_completion_tokens=99),
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
    # Simulate a pipeline that was loaded and then destroyed.
    client._pipeline_loaded = True
    client._pipe = None
    responses = client.batch_generate(
        messages=[
            [{"role": "user", "content": "x"}],
            [{"role": "user", "content": "y"}],
        ]
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
        _messages: list[list[dict[str, str]]],
        _sampling: object,
        chat_template_kwargs: dict[str, object] | None = None,
        use_tqdm: bool = False,
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

    client._ensure_pipeline_loaded()
    monkeypatch.setattr(client._pipe, "chat", _chat_short)
    responses = client.batch_generate(
        messages=[
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
        ]
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


def test_process_response_warns_by_default_on_length_stop(
    fake_vllm_runtime: dict[str, Any],
) -> None:
    # Verifies the offline client default policy downgrades truncation to a response error.
    client = VLLMOfflineClient(model="m")
    output = types.SimpleNamespace(
        outputs=[
            types.SimpleNamespace(
                text=" truncated ",
                token_ids=[1, 2, 3],
                finish_reason="length",
            )
        ]
    )

    response = client._process_response(output)  # type: ignore[arg-type]

    assert response.text == "truncated"
    assert response.stop_reason == "length"
    assert response.error is not None
    assert "hit the configured output token limit" in response.error
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
    client._ensure_pipeline_loaded()
    client.destroy()
    # Idempotent: second call should work without error
    client.destroy()
    assert client._pipe is None
    assert collected["called"] >= 1


def test_destroy_logs_failed_steps_and_continues(
    fake_vllm_runtime: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Verifies a failing clean-up step is logged by name and the rest still run.
    collected = {"called": 0}

    def _collect() -> int:
        collected["called"] += 1
        return 0

    monkeypatch.setattr(gc, "collect", _collect)

    def _raise() -> None:
        raise RuntimeError("teardown boom")

    fake_vllm_runtime["distributed"].destroy_model_parallel = _raise

    client = VLLMOfflineClient(model="m")
    client._ensure_pipeline_loaded()
    monkeypatch.setattr(
        client._pipe.llm_engine.engine_core,  # type: ignore[union-attr]
        "shutdown",
        _raise,
    )

    with caplog.at_level("WARNING"):
        client.destroy()

    warnings = [record.getMessage() for record in caplog.records]
    assert any(
        "destroy model parallel" in message and "teardown boom" in message
        for message in warnings
    )
    assert any("shut down engine core" in message for message in warnings)
    assert client._pipe is None
    assert collected["called"] == 1


def test_batch_generate_rejects_multiple_responses(
    fake_vllm_runtime: dict[str, Any],
) -> None:
    # Verifies a request for more than one response per sample is rejected.
    client = VLLMOfflineClient(model="m", on_error="ignore")

    with pytest.raises(ValueError, match="one response per sample"):
        client.batch_generate(
            messages=[[{"role": "user", "content": "x"}]],
            gen_kwargs={"n": 2},
        )

    assert fake_vllm_runtime["chat_calls"] == []
    client.destroy()


def test_runtime_options_reject_multiple_responses_in_extra_body() -> None:
    # Verifies 'n' inside extra_body is rejected when the options are built.
    with pytest.raises(ValueError, match="one response per sample"):
        VLLMOfflineRuntimeOptions(extra_body={"n": 3})
