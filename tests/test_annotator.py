from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset

from llm_annotator.annotator import Annotator
from llm_annotator.clients.base import (
    Client,
    Provider,
    ProviderRuntimeOptions,
    Response,
)
from llm_annotator.clients.claude_client import ClaudeClient
from llm_annotator.clients.gemini_client import GeminiClient
from llm_annotator.clients.openai_client import OpenAIClient
from llm_annotator.clients.vllm_client import VLLMClient
from llm_annotator.clients.vllm_offline_client import VLLMOfflineClient


class DummyClient(Client[ProviderRuntimeOptions]):
    provider_type = Provider.OPENAI

    def __init__(self, model: str = "dummy", on_error: str = "raise") -> None:
        super().__init__(model=model, on_error=on_error)  # type: ignore[arg-type]
        self.destroy_called = 0

    def _process_response(self, response: str) -> Response:
        return Response(
            text=response, provider=self.provider_type, model=self.model
        )

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        options: ProviderRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> Response:
        _ = gen_kwargs
        text = messages[-1]["content"]
        if options and options.json_schema is not None:
            text = json.dumps({"label": "ok", "echo": text})
        return Response(
            text=text,
            stop_reason="stop",
            provider=self.provider_type,
            model=self.model,
            num_output_tokens=3,
        )

    def batch_generate(
        self,
        *,
        messages: list[list[dict[str, str]]],
        options: ProviderRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> list[Response]:
        _ = gen_kwargs
        return [
            self.generate(messages=msg, options=options) for msg in messages
        ]

    def _handle_stop_reason(
        self, *, stop_reason: str | None, num_output_tokens: int | None
    ) -> None:
        _ = stop_reason
        _ = num_output_tokens

    def destroy(self) -> None:
        self.destroy_called += 1


@pytest.fixture
def dummy_annotator() -> Annotator:
    return Annotator(client=DummyClient(), batch_size=2, verbose=True)


def test_get_skip_idxs_with_filters(
    tmp_path: Path, dummy_annotator: Annotator
) -> None:
    # Verifies skip-index discovery respects dataset_split and dataset_config filters.
    p = tmp_path / "out"
    p.mkdir()
    (p / "out.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "idx": 1,
                        "dataset_split": "train",
                        "dataset_config": "en",
                    }
                ),
                json.dumps(
                    {"idx": 2, "dataset_split": "test", "dataset_config": "en"}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    only_train = dummy_annotator._get_skip_idxs(
        pdout=p,
        idx_column="idx",
        dataset_split="train",
        dataset_config="en",
    )
    assert only_train == {1}


def test_load_dataset_validation_errors(
    tmp_path: Path, dummy_annotator: Annotator
) -> None:
    # Verifies core _load_dataset argument validation branches.
    ds = Dataset.from_dict({"text": ["a"]})

    with pytest.raises(ValueError, match="Provide only one"):
        dummy_annotator._load_dataset(
            prompt_template="{text}",
            pdout=tmp_path,
            idx_column="idx",
            dataset=ds,
            dataset_name="x",
        )

    with pytest.raises(ValueError, match="must be provided"):
        dummy_annotator._load_dataset(
            prompt_template="{text}",
            pdout=tmp_path,
            idx_column="idx",
        )

    with pytest.raises(ValueError, match="positive integer"):
        dummy_annotator._load_dataset(
            prompt_template="{text}",
            pdout=tmp_path,
            idx_column="idx",
            dataset=ds,
            max_num_samples=0,
        )


def test_create_messages_with_and_without_system(
    dummy_annotator: Annotator,
) -> None:
    # Verifies message construction branches for system and non-system prompts.
    sample = {"text": "hello"}
    with_system = dummy_annotator._create_messages(
        sample,
        idx=4,
        prompt_fields=("text",),
        prompt_template="Say {text}",
        idx_column="idx",
        task_prefix="",
        system_message="sys",
    )
    no_system = dummy_annotator._create_messages(
        sample,
        idx=5,
        prompt_fields=("text",),
        prompt_template="Say {text}",
        idx_column="idx",
        task_prefix="",
        system_message=None,
    )

    assert with_system["prompted"][0]["role"] == "system"
    assert with_system["prompted"][1]["role"] == "user"
    assert no_system["prompted"][0]["role"] == "user"


def test_process_output_branches(dummy_annotator: Annotator) -> None:
    # Verifies output parsing branches: plain text, valid JSON, invalid JSON, and error payloads.
    schema = {
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
    }

    no_schema = dummy_annotator._process_output(response=Response(text="x"))
    assert no_schema["response"] == "x"

    ok_schema = dummy_annotator._process_output(
        response=Response(text='{"label":"good"}'),
        output_schema=schema,
    )
    assert ok_schema["valid_fields"] is True

    bad_json = dummy_annotator._process_output(
        response=Response(text="not-json"),
        output_schema=schema,
    )
    assert bad_json["valid_fields"] is False

    err_schema = dummy_annotator._process_output(
        response=Response(text="", error="boom", error_type="ProviderError"),
        output_schema=schema,
    )
    assert err_schema["valid_fields"] is False
    assert err_schema["label"] is None


def test_process_batch_validate_and_postprocess(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Verifies _process_batch integrates schema parsing, custom postprocess, and validate hooks.
    annotator = Annotator(client=DummyClient(on_error="ignore"), verbose=True)
    schema = {
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
    }

    batch = {
        "prompted": [
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
        ]
    }

    def _post(x: dict[str, Any]) -> dict[str, Any]:
        x["post"] = True
        return x

    res = annotator._process_batch(
        batch=batch,
        options=ProviderRuntimeOptions(json_schema=schema),
        validate_fn=lambda x: x.get("label") == "ok",
        postprocess_fn=_post,
    )
    assert len(res) == 2
    assert all(item["post"] is True for item in res)
    assert all(item["valid"] is True for item in res)

    _ = capsys.readouterr()


def test_annotate_dataset_retries_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Verifies annotate_dataset retry loop re-processes invalid outputs.
    annotator = Annotator(
        client=DummyClient(on_error="ignore"), batch_size=2, verbose=False
    )
    ds = Dataset.from_dict({"text": ["a", "b"]})

    calls = {"n": 0}

    def _fake_process_batch(
        self: Annotator, **kwargs: Any
    ) -> list[dict[str, Any]]:
        _ = self
        calls["n"] += 1
        batch = kwargs["batch"]
        size = len(batch["idx"])
        if calls["n"] == 1:
            return [
                {
                    "response": "bad",
                    "finish_reason": "stop",
                    "num_tokens": 1,
                    "error": None,
                    "error_type": None,
                    "valid": False,
                }
                for _ in range(size)
            ]
        return [
            {
                "response": "good",
                "finish_reason": "stop",
                "num_tokens": 1,
                "error": None,
                "error_type": None,
                "valid": True,
            }
            for _ in range(size)
        ]

    monkeypatch.setattr(Annotator, "_process_batch", _fake_process_batch)

    out = annotator.annotate_dataset(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
        validate_fn=lambda x: x["response"] == "good",
        num_retries_invalid=2,
    )

    assert len(out) == 2
    assert calls["n"] >= 2


def test_annotate_dataset_guard_rails(tmp_path: Path) -> None:
    # Verifies annotate_dataset rejects unsupported runtime combinations and upload config.
    client = DummyClient()
    annotator = Annotator(client=client, num_proc=2)
    client._pipe = object()  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="cannot be pickled"):
        annotator.annotate_dataset(
            output_dir=tmp_path / "x",
            prompt_template="{text}",
            dataset=Dataset.from_dict({"text": ["a"]}),
        )

    annotator = Annotator(client=DummyClient())
    with pytest.raises(ValueError, match="new_hub_id must be provided"):
        annotator.annotate_dataset(
            output_dir=tmp_path / "x2",
            prompt_template="{text}",
            dataset=Dataset.from_dict({"text": ["a"]}),
            upload_every_n_samples=10,
        )


def test_generate_dataset_forwards_to_annotate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Verifies generate_dataset builds prompt dataset and forwards expected args.
    annotator = Annotator(client=DummyClient())
    captured: dict[str, Any] = {}

    def _fake_annotate_dataset(
        self: Annotator, *args: Any, **kwargs: Any
    ) -> Dataset:
        _ = self
        _ = args
        captured.update(kwargs)
        return Dataset.from_dict({"response": ["ok"]})

    monkeypatch.setattr(Annotator, "annotate_dataset", _fake_annotate_dataset)
    result = annotator.generate_dataset(
        output_dir=tmp_path / "g",
        prompts="hello",
        max_num_samples=3,
    )

    assert len(result) == 1
    assert captured["prompt_template"] == "{prompt}"
    assert len(captured["dataset"]) == 3


def test_post_annotate_and_pfout_name(
    tmp_path: Path, dummy_annotator: Annotator
) -> None:
    # Verifies _post_annotate output assembly/sorting and output filename generation.
    p = tmp_path / "out"
    p.mkdir()
    (p / "a.jsonl").write_text(
        '{"idx": 1, "response": "x"}\n', encoding="utf-8"
    )
    (p / "b.jsonl").write_text(
        '{"idx": 0, "response": "y"}\n', encoding="utf-8"
    )

    done = dummy_annotator._post_annotate(
        pdout=p, idx_column="idx", keep_idx_column=False
    )
    assert done.column_names == ["response"]

    single = dummy_annotator.get_pfout_name(
        pdout=p, max_samples_per_output_file=0, processed_n_samples=0
    )
    chunked = dummy_annotator.get_pfout_name(
        pdout=p, max_samples_per_output_file=10, processed_n_samples=25
    )
    assert single.name == "out.jsonl"
    assert chunked.name == "out_2.jsonl"


def test_push_dir_to_hub_calls_hf_helpers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Verifies push_dir_to_hub validation and helper call ordering.
    annotator = Annotator(client=DummyClient(), verbose=True)
    called: list[str] = []

    monkeypatch.setattr(
        "llm_annotator.annotator.create_repo",
        lambda *args, **kwargs: called.append("repo"),
    )
    monkeypatch.setattr(
        "llm_annotator.annotator.create_branch",
        lambda *args, **kwargs: called.append("branch"),
    )
    monkeypatch.setattr(
        "llm_annotator.annotator.upload_large_folder",
        lambda *args, **kwargs: called.append("upload"),
    )

    with pytest.raises(ValueError, match="must be set"):
        annotator.push_dir_to_hub(tmp_path, new_hub_id=None)

    annotator.push_dir_to_hub(tmp_path, new_hub_id="me/test")
    assert called == ["repo", "branch", "upload"]


def test_destroy_on_error_calls_client_destroy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Verifies destroy_on_error wrapper triggers client cleanup on pipeline exceptions.
    client = DummyClient()
    annotator = Annotator(client=client)

    def _boom(self: Annotator, **_kwargs: Any) -> tuple[Dataset, int]:
        _ = self
        raise RuntimeError("boom")

    monkeypatch.setattr(Annotator, "_load_dataset", _boom)

    with pytest.raises(RuntimeError, match="boom"):
        annotator.annotate_dataset(
            output_dir=tmp_path / "x",
            prompt_template="{text}",
            dataset=Dataset.from_dict({"text": ["a"]}),
        )

    assert client.destroy_called == 1


@pytest.mark.parametrize(
    "client_cls",
    [OpenAIClient, ClaudeClient, GeminiClient, VLLMClient, VLLMOfflineClient],
)
def test_annotator_smoke_with_all_client_types(client_cls: type[Any]) -> None:
    # Verifies Annotator batch processing works across all client classes when calls are mocked.
    client = object.__new__(client_cls)
    client.model = "fake-model"
    client.max_workers = 1
    client.on_error = "ignore"
    client.batch_generate = types.MethodType(
        lambda self, *, messages, options=None, gen_kwargs=None: [
            Response(text="ok", provider=self.provider_type, model=self.model)
            for _ in messages
        ],
        client,
    )
    client.warm_up = types.MethodType(lambda self, **kwargs: None, client)
    client.destroy = types.MethodType(lambda self: None, client)

    annotator = Annotator(client=client)
    out = annotator._process_batch(
        batch={
            "prompted": [
                [{"role": "user", "content": "a"}],
                [{"role": "user", "content": "b"}],
            ]
        },
        options=None,
    )
    assert len(out) == 2
    assert all(item["response"] == "ok" for item in out)


def test_load_dataset_with_dataset_name_split_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Verifies _load_dataset defaults to the only split when dataset_name is provided.
    annotator = Annotator(client=DummyClient(), verbose=False)

    monkeypatch.setattr(
        "llm_annotator.annotator.get_dataset_split_names",
        lambda dataset_name, config_name=None: ["train"],
    )
    monkeypatch.setattr(
        "llm_annotator.annotator.load_dataset",
        lambda *args, **kwargs: Dataset.from_dict({"text": ["a", "b"]}),
    )

    loaded, skipped = annotator._load_dataset(
        prompt_template="Q: {text}",
        pdout=tmp_path / "out",
        idx_column="idx",
        dataset_name="dummy/name",
        prompt_fields=("text",),
        cache_input_dataset=False,
        use_cached_input_dataset=False,
    )

    assert len(loaded) == 2
    assert skipped == 0
    assert "prompted" in loaded.column_names


def test_load_dataset_split_validation_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Verifies _load_dataset raises on ambiguous or unknown dataset split names.
    annotator = Annotator(client=DummyClient(), verbose=False)

    monkeypatch.setattr(
        "llm_annotator.annotator.get_dataset_split_names",
        lambda dataset_name, config_name=None: ["train", "test"],
    )

    with pytest.raises(ValueError, match="multiple splits"):
        annotator._load_dataset(
            prompt_template="Q: {text}",
            pdout=tmp_path / "o1",
            idx_column="idx",
            dataset_name="dummy/name",
            prompt_fields=("text",),
            cache_input_dataset=False,
            use_cached_input_dataset=False,
        )

    with pytest.raises(ValueError, match="does not have a split"):
        annotator._load_dataset(
            prompt_template="Q: {text}",
            pdout=tmp_path / "o2",
            idx_column="idx",
            dataset_name="dummy/name",
            dataset_split="validation",
            prompt_fields=("text",),
            cache_input_dataset=False,
            use_cached_input_dataset=False,
        )


def test_annotate_dataset_output_schema_validation(tmp_path: Path) -> None:
    # Verifies output_schema normalization and conflict checks with options.json_schema.
    annotator = Annotator(client=DummyClient(), verbose=False)
    ds = Dataset.from_dict({"text": ["a"]})

    with pytest.raises(TypeError, match="decode to a dictionary"):
        annotator.annotate_dataset(
            output_dir=tmp_path / "a",
            prompt_template="Q: {text}",
            dataset=ds,
            output_schema="[]",
        )

    with pytest.raises(ValueError, match="Provide 'output_schema' OR set"):
        annotator.annotate_dataset(
            output_dir=tmp_path / "b",
            prompt_template="Q: {text}",
            dataset=ds,
            options=ProviderRuntimeOptions(json_schema={"type": "object"}),
            output_schema={"type": "object"},
        )


def test_annotate_dataset_keep_columns_type_error(tmp_path: Path) -> None:
    # Verifies keep_columns validation rejects unsupported non-iterable objects.
    annotator = Annotator(client=DummyClient(), verbose=False)

    with pytest.raises(TypeError, match="keep_columns must be"):
        annotator.annotate_dataset(
            output_dir=tmp_path / "x",
            prompt_template="Q: {text}",
            dataset=Dataset.from_dict({"text": ["a"]}),
            keep_columns=1,
        )
