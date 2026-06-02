from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any, cast

import pytest
from datasets import Dataset

from llm_annotator.annotator import (
    Annotator,
    _create_messages,
    destroy_on_error,
)
from llm_annotator.clients.base import (
    Client,
    Provider,
    ProviderRuntimeOptions,
    Response,
)
from llm_annotator.clients.claude_client import ClaudeClient
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
            idx_column="idx",
            dataset=ds,
            dataset_name="x",
        )

    with pytest.raises(ValueError, match="must be provided"):
        dummy_annotator._load_dataset(
            prompt_template="{text}",
            idx_column="idx",
        )

    with pytest.raises(ValueError, match="positive integer"):
        dummy_annotator._load_dataset(
            prompt_template="{text}",
            idx_column="idx",
            dataset=ds,
            max_num_samples=0,
        )


def test_create_messages_with_and_without_system() -> None:
    # Verifies message construction branches for system and non-system prompts.
    sample = {"text": "hello"}
    with_system = _create_messages(
        sample,
        prompt_fields=("text",),
        prompt_template="Say {text}",
        task_prefix="",
        system_message="sys",
    )
    no_system = _create_messages(
        sample,
        prompt_fields=("text",),
        prompt_template="Say {text}",
        task_prefix="",
        system_message=None,
    )

    assert with_system["messages"][0]["role"] == "system"
    assert with_system["messages"][1]["role"] == "user"
    assert no_system["messages"][0]["role"] == "user"


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
        "messages": [
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


def test_run_annotation_retries_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Verifies run_annotation retry loop re-processes invalid outputs.
    annotator = Annotator(
        client=DummyClient(on_error="ignore"), batch_size=2, verbose=False
    )
    prepared_ds = Dataset.from_dict(
        {
            "idx": [0, 1],
            "text": ["a", "b"],
            "messages": [
                [{"role": "user", "content": "Q: a"}],
                [{"role": "user", "content": "Q: b"}],
            ],
        }
    )

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

    out = annotator.run_annotation(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        prepared_dataset=prepared_ds,
        num_retries_invalid=2,
    )

    assert len(out) == 2
    assert calls["n"] >= 2


def test_run_annotation_guard_rails(tmp_path: Path) -> None:
    annotator = Annotator(client=DummyClient())
    prepared_ds = Dataset.from_dict(
        {
            "idx": [0],
            "messages": [[{"role": "user", "content": "Q: a"}]],
        }
    )
    with pytest.raises(ValueError, match="new_hub_id must be provided"):
        # Because upload_every_n_samples is set, new_hub_id must be provided.
        annotator.run_annotation(
            output_dir=tmp_path / "x2",
            prompt_template="{text}",
            prepared_dataset=prepared_ds,
            upload_every_n_samples=10,
        )


def test_run_annotation_resume_from_hub_id_validation(
    tmp_path: Path,
) -> None:
    # Verifies resume_from_hub_id format and overwrite guard rails.
    annotator = Annotator(client=DummyClient())
    prepared_ds = Dataset.from_dict(
        {
            "idx": [0],
            "messages": [[{"role": "user", "content": "Q: a"}]],
        }
    )

    with pytest.raises(ValueError, match="must be a Hugging Face dataset ID"):
        annotator.run_annotation(
            output_dir=tmp_path / "bad-id",
            prompt_template="{text}",
            prepared_dataset=prepared_ds,
            resume_from_hub_id="invalid-id",
        )

    with pytest.raises(ValueError, match="overwrite=True"):
        annotator.run_annotation(
            output_dir=tmp_path / "overwrite",
            prompt_template="{text}",
            prepared_dataset=prepared_ds,
            resume_from_hub_id="owner/name",
            overwrite=True,
        )


def test_prepare_data_uses_local_cache(tmp_path: Path) -> None:
    # Verifies prepare_data loads from the on-disk cached_input_dataset on repeat runs.
    annotator = Annotator(client=DummyClient())
    ds = Dataset.from_dict({"text": ["a", "b"]})

    first_ds, first_path, _ = annotator.prepare_data(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
    )
    assert first_path is not None

    cached_ds, cached_path, _ = annotator.prepare_data(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
    )

    assert cached_path is not None
    assert first_path == cached_path
    assert len(first_ds) == len(cached_ds)


def test_run_annotation_skips_existing_indices_from_output(
    tmp_path: Path,
) -> None:
    # Verifies existing jsonl output rows are respected by skip-index resume logic.
    annotator = Annotator(client=DummyClient())
    prepared_ds = Dataset.from_dict(
        {
            "idx": [0, 1],
            "text": ["a", "b"],
            "messages": [
                [{"role": "user", "content": "Q: a"}],
                [{"role": "user", "content": "Q: b"}],
            ],
        }
    )

    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    (tmp_path / "out" / "restored.jsonl").write_text(
        '{"idx": 0, "response": "from_out"}\n', encoding="utf-8"
    )

    done = annotator.run_annotation(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        prepared_dataset=prepared_ds,
        keep_idx_column=True,
    )

    assert len(done) == 2
    assert set(done["idx"]) == {0, 1}


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
    # Verifies destroy_on_error wrapper triggers client cleanup on run_annotation exceptions.
    client = DummyClient()
    annotator = Annotator(client=client)
    prepared_ds = Dataset.from_dict(
        {
            "idx": [0],
            "messages": [[{"role": "user", "content": "Q: a"}]],
        }
    )

    def _boom(self: Annotator, **_kwargs: Any) -> list[dict[str, Any]]:
        _ = self
        raise RuntimeError("boom")

    monkeypatch.setattr(Annotator, "_process_batch", _boom)

    with pytest.raises(RuntimeError, match="boom"):
        annotator.run_annotation(
            output_dir=tmp_path / "x",
            prompt_template="Q: {text}",
            prepared_dataset=prepared_ds,
        )

    assert client.destroy_called == 1


@pytest.mark.parametrize(
    "client_cls",
    [OpenAIClient, ClaudeClient, VLLMClient, VLLMOfflineClient],
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
            "messages": [
                [{"role": "user", "content": "a"}],
                [{"role": "user", "content": "b"}],
            ]
        },
        options=None,
    )
    assert len(out) == 2
    assert all(item["response"] == "ok" for item in out)


def test_load_dataset_with_dataset_name_split_selection(
    monkeypatch: pytest.MonkeyPatch,
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

    loaded = annotator._load_dataset(
        prompt_template="Q: {text}",
        idx_column="idx",
        dataset_name="dummy/name",
        prompt_fields=("text",),
    )

    assert len(loaded) == 2
    assert "messages" in loaded.column_names


def test_load_dataset_split_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
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
            idx_column="idx",
            dataset_name="dummy/name",
            prompt_fields=("text",),
        )

    with pytest.raises(ValueError, match="does not have a split"):
        annotator._load_dataset(
            prompt_template="Q: {text}",
            idx_column="idx",
            dataset_name="dummy/name",
            dataset_split="validation",
            prompt_fields=("text",),
        )


def test_run_annotation_output_schema_validation(tmp_path: Path) -> None:
    # Verifies output_schema normalization and conflict checks with options.json_schema.
    annotator = Annotator(client=DummyClient(), verbose=False)
    ds = Dataset.from_dict({"text": ["a"]})
    prepared_ds, _, _ = annotator.prepare_data(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
    )

    with pytest.raises(TypeError, match="decode to a dictionary"):
        annotator.run_annotation(
            output_dir=tmp_path / "a",
            prompt_template="Q: {text}",
            prepared_dataset=prepared_ds,
            output_schema="[]",
        )

    with pytest.raises(ValueError, match="Provide 'output_schema' OR set"):
        annotator.run_annotation(
            output_dir=tmp_path / "b",
            prompt_template="Q: {text}",
            prepared_dataset=prepared_ds,
            options=ProviderRuntimeOptions(json_schema={"type": "object"}),
            output_schema={"type": "object"},
        )


def test_run_annotation_keep_columns_type_error(tmp_path: Path) -> None:
    # Verifies keep_columns validation rejects unsupported non-iterable objects.
    annotator = Annotator(client=DummyClient(), verbose=False)
    prepared_ds = Dataset.from_dict(
        {
            "idx": [0],
            "messages": [[{"role": "user", "content": "Q: a"}]],
        }
    )

    with pytest.raises(TypeError, match="keep_columns must be"):
        annotator.run_annotation(
            output_dir=tmp_path / "x",
            prompt_template="Q: {text}",
            prepared_dataset=prepared_ds,
            keep_columns=1,
        )


def test_run_annotation_writes_version_file(tmp_path: Path) -> None:
    # Verifies _version.json is created in output_dir at runtime.
    annotator = Annotator(client=DummyClient(), batch_size=4, num_proc=2)
    prepared_ds = Dataset.from_dict(
        {
            "idx": [0, 1],
            "messages": [
                [{"role": "user", "content": "Q: a"}],
                [{"role": "user", "content": "Q: b"}],
            ],
        }
    )
    annotator.run_annotation(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        prepared_dataset=prepared_ds,
    )

    version_path = tmp_path / "out" / "_version.json"
    assert version_path.exists()
    version_data = json.loads(version_path.read_text(encoding="utf-8"))
    assert "python" in version_data
    assert "llm_annotator" in version_data


def test_run_annotation_keep_columns_and_validation_fields(
    tmp_path: Path,
) -> None:
    # Verifies run_annotation preserves requested columns and writes validation metadata.
    def my_validator(sample: dict) -> bool:
        return bool(sample.get("response"))

    options = ProviderRuntimeOptions(max_tokens=64)
    annotator = Annotator(client=DummyClient())
    prepared_ds = Dataset.from_dict(
        {
            "idx": [0],
            "text": ["a"],
            "messages": [[{"role": "user", "content": "Q: a"}]],
        }
    )
    done = annotator.run_annotation(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        prepared_dataset=prepared_ds,
        keep_columns=["text"],
        options=options,
        validate_fn=my_validator,
        num_retries_invalid=0,
        system_message="You are helpful.",
        keep_idx_column=True,
    )

    assert done.column_names == [
        "idx",
        "text",
        "response",
        "finish_reason",
        "num_tokens",
        "error",
        "error_type",
        "valid",
    ]
    assert done["valid"] == [True]


def test_destroy_on_error_appends_cleanup_failure_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Verifies cleanup failures add a note to the original exception.
    annotator = Annotator(client=DummyClient())

    class NoteError(RuntimeError):
        pass

    @destroy_on_error
    def _boom(self: Annotator) -> None:
        err = NoteError("boom")
        err.add_note("original note")
        raise err

    def _failing_destroy(self: Annotator) -> None:
        _ = self
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(Annotator, "destroy", _failing_destroy)

    with pytest.raises(NoteError) as excinfo:
        _boom(annotator)

    assert any(
        note.startswith("Cleanup failed: RuntimeError('cleanup failed')")
        for note in excinfo.value.__notes__
    )


def test_annotator_context_manager_calls_destroy() -> None:
    # Verifies __enter__ returns self and __exit__ always destroys the client.
    client = DummyClient()
    annotator = Annotator(client=client)

    with annotator as entered:
        assert entered is annotator

    assert client.destroy_called == 1


def test_load_dataset_handles_loaded_vllm_pipeline_and_sorting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Verifies the offline-vLLM multiprocessing guard and preprocessing branches.
    class FakeVLLMOfflineClient:
        def __init__(self) -> None:
            self._pipeline_loaded = True

    monkeypatch.setattr(
        "llm_annotator.annotator.VLLMOfflineClient",
        FakeVLLMOfflineClient,
    )

    annotator = Annotator(
        client=cast(Client[Any], FakeVLLMOfflineClient()),
        num_proc=2,
        verbose=True,
    )
    dataset = Dataset.from_dict({"text": ["bbb", "a", "cc"]})

    def _preprocess(*, dataset: Dataset) -> Dataset:
        return dataset.add_column("extra", list(range(len(dataset))))

    loaded = annotator._load_dataset(
        prompt_template="Q: {text}",
        idx_column="idx",
        dataset=dataset,
        prompt_fields=("text",),
        preprocess_fn=_preprocess,
        shuffle_seed=1,
        max_num_samples=2,
        sort_by_length="shortest_first",
        system_message="sys",
        task_prefix="pre_",
    )

    assert annotator.num_proc is None
    assert "pre_messages" in loaded.column_names
    assert "pre_messages_chars" not in loaded.column_names


def test_prepare_data_uses_prepared_hub_and_force_rebuild(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Verifies prepared Hub restore and force-rebuild behavior.
    annotator = Annotator(client=DummyClient())
    dataset = Dataset.from_dict({"text": ["a", "b"]})

    monkeypatch.setattr(
        "llm_annotator.annotator.load_dataset",
        lambda *args, **kwargs: dataset,
    )

    cached_ds, cached_path, cached_hub_id = annotator.prepare_data(
        output_dir=tmp_path / "hub-cache",
        prompt_template="Q: {text}",
        prepared_hub_id="owner/prepared",
    )

    assert cached_ds is dataset
    assert cached_path is None
    assert cached_hub_id == "owner/prepared"

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "llm_annotator.annotator.delete_branch",
        lambda *args, **kwargs: calls.append((args[0], kwargs["branch"])),
    )
    monkeypatch.setattr(
        Annotator,
        "_load_dataset",
        lambda self, **kwargs: dataset,
    )
    monkeypatch.setattr(
        Dataset,
        "push_to_hub",
        lambda *args, **kwargs: None,
    )

    rebuilt_ds, rebuilt_path, rebuilt_hub_id = annotator.prepare_data(
        output_dir=tmp_path / "force-rebuild",
        prompt_template="Q: {text}",
        prepared_hub_id="owner/prepared",
        force_data_preparation=True,
    )

    assert rebuilt_ds is dataset
    assert rebuilt_path is not None
    assert rebuilt_hub_id == "owner/prepared"
    assert calls == [("owner/prepared", "prepared_cache")]


def test_run_annotation_validation_and_short_circuit_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Verifies validation failures, missing prepared data, and all-processed short-circuit paths.
    annotator = Annotator(client=DummyClient())
    prepared_ds = Dataset.from_dict(
        {
            "idx": [0],
            "text": ["a"],
            "messages": [[{"role": "user", "content": "Q: a"}]],
        }
    )

    with pytest.raises(ValueError, match="max_samples_per_output_file"):
        annotator.run_annotation(
            output_dir=tmp_path / "bad-max",
            prompt_template="Q: {text}",
            prepared_dataset=prepared_ds,
            max_samples_per_output_file=-1,
        )

    with pytest.raises(ValueError, match="upload_every_n_samples"):
        annotator.run_annotation(
            output_dir=tmp_path / "bad-upload",
            prompt_template="Q: {text}",
            prepared_dataset=prepared_ds,
            upload_every_n_samples=1.5,
            new_hub_id="owner/output",
        )

    with pytest.raises(TypeError, match="decode to a dictionary"):
        annotator.run_annotation(
            output_dir=tmp_path / "bad-schema",
            prompt_template="Q: {text}",
            prepared_dataset=prepared_ds,
            output_schema="[]",
        )

    with pytest.raises(TypeError, match="keep_columns must be"):
        annotator.run_annotation(
            output_dir=tmp_path / "bad-columns",
            prompt_template="Q: {text}",
            prepared_dataset=prepared_ds,
            keep_columns=1,
        )

    prepared_missing_idx = Dataset.from_dict(
        {"messages": [[{"role": "user", "content": "Q: a"}]]}
    )
    with pytest.raises(ValueError, match="Expected index column"):
        annotator.run_annotation(
            output_dir=tmp_path / "missing-idx",
            prompt_template="Q: {text}",
            prepared_dataset=prepared_missing_idx,
        )

    monkeypatch.setattr(
        "llm_annotator.annotator.load_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("missing hub")
        ),
    )
    monkeypatch.setattr(
        Dataset,
        "load_from_disk",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("missing disk")
        ),
    )

    with pytest.raises(ValueError, match="No prepared data found"):
        annotator.run_annotation(
            output_dir=tmp_path / "missing-prepared",
            prompt_template="Q: {text}",
            prepared_hub_id="owner/prepared",
        )

    output_dir = tmp_path / "complete"
    output_dir.mkdir()
    (output_dir / "existing.jsonl").write_text(
        '{"idx": 0, "response": "done"}\n',
        encoding="utf-8",
    )

    completed = annotator.run_annotation(
        output_dir=output_dir,
        prompt_template="Q: {text}",
        prepared_dataset=prepared_ds,
    )

    assert len(completed) == 1
    assert completed["response"] == ["done"]


def test_run_annotation_keeps_all_columns_when_requested(
    tmp_path: Path,
) -> None:
    # Verifies keep_columns=True preserves the full prepared batch payload.
    annotator = Annotator(client=DummyClient())
    prepared_ds = Dataset.from_dict(
        {
            "idx": [0],
            "text": ["a"],
            "messages": [[{"role": "user", "content": "Q: a"}]],
        }
    )

    done = annotator.run_annotation(
        output_dir=tmp_path / "keep-all",
        prompt_template="Q: {text}",
        prepared_dataset=prepared_ds,
        keep_columns=True,
        keep_idx_column=True,
    )

    assert done.column_names == [
        "idx",
        "text",
        "messages",
        "response",
        "finish_reason",
        "num_tokens",
        "error",
        "error_type",
    ]
