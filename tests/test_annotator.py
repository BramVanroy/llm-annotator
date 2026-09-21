from __future__ import annotations

import functools
import json
import logging
import threading
import types
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, cast

import pytest
from datasets import Dataset, load_dataset

from llm_annotator.annotator import (
    PROGRESS_UPLOAD_FILE,
    Annotator,
    SelectionRecord,
    VLLMQueueAnnotator,
    _callable_component,
    _copy_file_prefix,
    _create_messages,
    _ProgressUploader,
    _resolve_samples_per_output_file,
    destroy_on_error,
    is_retried_error,
)
from llm_annotator.clients.base import (
    Client,
    Provider,
    ProviderRuntimeOptions,
    Response,
)
from llm_annotator.clients.claude_client import ClaudeClient
from llm_annotator.clients.exceptions import (
    TooManyConsecutiveFailedBatchesError,
)
from llm_annotator.clients.openai_client import OpenAIClient
from llm_annotator.clients.vllm_offline_client import VLLMOfflineClient
from llm_annotator.clients.vllm_online_client import VLLMOnlineClient
from llm_annotator.utils import dataset_signature, get_hash


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
        if options and options.output_schema is not None:
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


class TrackingClient(DummyClient):
    """A DummyClient that records every prompt it was asked to answer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.seen_prompts: list[str] = []

    def batch_generate(
        self,
        *,
        messages: list[list[dict[str, str]]],
        options: ProviderRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> list[Response]:
        self.seen_prompts.extend(msg[-1]["content"] for msg in messages)
        return super().batch_generate(
            messages=messages, options=options, gen_kwargs=gen_kwargs
        )


class ScriptedClient(DummyClient):
    """A DummyClient whose answer text comes from a script.

    The script is called as ``script(call_number, messages)`` for every sample
    of a batch, which makes it easy to answer differently on a retry.
    """

    def __init__(
        self,
        script: Callable[[int, list[dict[str, str]]], str],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.script = script
        self.calls = 0

    def batch_generate(
        self,
        *,
        messages: list[list[dict[str, str]]],
        options: ProviderRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> list[Response]:
        _ = options
        _ = gen_kwargs
        self.calls += 1
        return [
            Response(
                text=self.script(self.calls, msg),
                stop_reason="stop",
                provider=self.provider_type,
                model=self.model,
                num_output_tokens=1,
            )
            for msg in messages
        ]


@pytest.fixture
def dummy_annotator() -> Annotator:
    return Annotator(client=DummyClient(), batch_size=2, verbose=True)


def test_load_source_and_selection_validation_errors(
    tmp_path: Path, dummy_annotator: Annotator
) -> None:
    # Verifies the source and selection validation branches.
    ds = Dataset.from_dict({"text": ["a"]})

    with pytest.raises(ValueError, match="Provide only one"):
        dummy_annotator._load_source(dataset=ds, dataset_name="x")

    with pytest.raises(ValueError, match="must be provided"):
        dummy_annotator._load_source()

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


@pytest.mark.parametrize("text", ["[1, 2]", '"text"', "null", "3"])
def test_process_output_marks_a_non_object_response_invalid(
    dummy_annotator: Annotator, text: str
) -> None:
    # Verifies valid JSON that is not an object is invalid instead of raising.
    schema = {
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
    }

    res = dummy_annotator._process_output(
        response=Response(text=text), output_schema=schema
    )

    assert res["valid_fields"] is False
    assert res["label"] is None


def test_process_output_fills_absent_properties_with_none(
    dummy_annotator: Annotator,
) -> None:
    # Verifies every schema property is a key, whether the response holds it
    # or not, so that all rows of a run have the same columns.
    schema = {
        "type": "object",
        "properties": {"label": {"type": "string"}, "score": {"type": "int"}},
        "required": ["label"],
    }

    res = dummy_annotator._process_output(
        response=Response(text='{"label": "good"}'), output_schema=schema
    )

    assert res["valid_fields"] is True
    assert res["score"] is None


def test_process_output_without_required_is_valid_when_it_parses(
    dummy_annotator: Annotator,
) -> None:
    # Verifies a schema that requires nothing accepts any parsed object.
    schema = {"type": "object", "properties": {"label": {"type": "string"}}}

    res = dummy_annotator._process_output(
        response=Response(text="{}"), output_schema=schema
    )

    assert res["valid_fields"] is True
    assert res["label"] is None


def test_process_output_missing_required_property_is_invalid(
    dummy_annotator: Annotator,
) -> None:
    # Verifies a response that leaves out a required property is invalid.
    schema = {
        "type": "object",
        "properties": {"label": {"type": "string"}, "score": {"type": "int"}},
        "required": ["label", "score"],
    }

    res = dummy_annotator._process_output(
        response=Response(text='{"label": "good"}'), output_schema=schema
    )

    assert res["valid_fields"] is False


def test_process_output_ignores_keys_outside_the_schema(
    dummy_annotator: Annotator, caplog: pytest.LogCaptureFixture
) -> None:
    # Verifies an undeclared key is dropped and reported once per run.
    schema = {
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
    }
    response = Response(text='{"label": "good", "extra": 1, "response": "x"}')

    with caplog.at_level(logging.WARNING):
        first = dummy_annotator._process_output(
            response=response, output_schema=schema
        )
        dummy_annotator._process_output(
            response=response, output_schema=schema
        )

    assert first["label"] == "good"
    assert "extra" not in first
    assert first["response"] == response.text
    warnings = [rec.message for rec in caplog.records]
    assert sum("'extra'" in message for message in warnings) == 1
    assert sum("'response'" in message for message in warnings) == 1


def test_run_annotation_rejects_a_schema_property_named_like_a_column(
    tmp_path: Path,
) -> None:
    # Verifies a collision is refused before any request is sent.
    annotator = Annotator(client=DummyClient())
    prepared = Dataset.from_dict(
        {"idx": [0], "messages": [[{"role": "user", "content": "Q"}]]}
    )

    with pytest.raises(ValueError, match="'idx', 'response'"):
        annotator.run_annotation(
            output_dir=tmp_path / "out",
            prepared_dataset=prepared,
            output_schema={
                "type": "object",
                "properties": {
                    "idx": {"type": "int"},
                    "response": {"type": "string"},
                    "label": {"type": "string"},
                },
            },
            upload_every_n_samples=0,
        )


def test_run_annotation_continues_after_a_non_object_response(
    tmp_path: Path,
) -> None:
    # Verifies a JSON array response does not end the run.
    annotator = Annotator(client=ScriptedClient(lambda call, msg: "[1, 2]"))
    ds = Dataset.from_dict({"text": ["a", "b"]})

    out = annotator.annotate_dataset(
        output_dir=tmp_path / "out",
        prompt_template="x {text}",
        dataset=ds,
        upload_every_n_samples=0,
        num_retries_invalid=0,
        output_schema={
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
        },
    )

    assert out["valid_fields"] == [False, False]
    assert out["label"] == [None, None]


def test_post_annotate_loads_a_schema_column_that_is_null_in_one_file(
    tmp_path: Path, dummy_annotator: Annotator
) -> None:
    # Verifies the final dataset loads when a run errored before it succeeded,
    # which types the schema column as null in the older progress file.
    progress_dir = tmp_path / "out" / "progress_backup"
    progress_dir.mkdir(parents=True)
    (progress_dir / "progress_0.jsonl").write_text(
        '{"idx": 0, "response": "", "valid_fields": false, "label": null}\n',
        encoding="utf-8",
    )
    (progress_dir / "progress_1.jsonl").write_text(
        '{"idx": 1, "response": "{}", "valid_fields": true, "label": "ok"}\n',
        encoding="utf-8",
    )

    ds = dummy_annotator._post_annotate(
        process_pdout=progress_dir, idx_column="idx", keep_idx_column=True
    )

    assert ds["label"] == [None, "ok"]
    assert Dataset.load_from_disk(tmp_path / "out")["label"] == [None, "ok"]


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
        options=ProviderRuntimeOptions(output_schema=schema),
        validate_fn=lambda x: x.get("label") == "ok",
        postprocess_fn=_post,
    )
    assert len(res) == 2
    assert all(item["post"] is True for item in res)
    assert all(item["valid"] is True for item in res)

    _ = capsys.readouterr()


def test_retried_samples_are_postprocessed(tmp_path: Path) -> None:
    # Verifies postprocess_fn runs on a retry as well as on the first attempt.
    def script(call: int, messages: list[dict[str, str]]) -> str:
        if call == 1 and messages[-1]["content"].endswith("b"):
            return "BAD"
        return "ok"

    annotator = Annotator(client=ScriptedClient(script))
    ds = Dataset.from_dict({"text": ["a", "b", "c"]})

    out = annotator.annotate_dataset(
        output_dir=tmp_path / "out",
        prompt_template="x {text}",
        dataset=ds,
        upload_every_n_samples=0,
        postprocess_fn=lambda sample: {
            **sample,
            "upper": sample["response"].upper(),
        },
        validate_fn=lambda sample: sample["response"] == "ok",
    )

    assert [(row["response"], row["upper"]) for row in out] == [
        ("ok", "OK"),
        ("ok", "OK"),
        ("ok", "OK"),
    ]


def test_validate_fn_sees_postprocessed_columns_on_a_retry(
    tmp_path: Path,
) -> None:
    # Verifies a validate_fn that reads a postprocessed column works on a
    # retry, where the first attempt was invalid.
    def script(call: int, messages: list[dict[str, str]]) -> str:
        _ = messages
        return "ok" if call > 1 else "bad"

    client = ScriptedClient(script)
    annotator = Annotator(client=client)
    ds = Dataset.from_dict({"text": ["a", "b"]})

    out = annotator.annotate_dataset(
        output_dir=tmp_path / "out",
        prompt_template="x {text}",
        dataset=ds,
        upload_every_n_samples=0,
        postprocess_fn=lambda sample: {
            **sample,
            "upper": sample["response"].upper(),
        },
        validate_fn=lambda sample: sample["upper"] == "OK",
    )

    assert client.calls == 2
    assert out["valid"] == [True, True]
    assert out["upper"] == ["OK", "OK"]


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


def test_run_annotation_raises_after_consecutive_failed_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Verifies the circuit breaker aborts once too many batches in a row
    # come back with every sample errored, instead of running to completion.
    annotator = Annotator(
        client=DummyClient(on_error="ignore"), batch_size=1, verbose=False
    )
    prepared_ds = Dataset.from_dict(
        {
            "idx": list(range(5)),
            "text": [str(i) for i in range(5)],
            "messages": [
                [{"role": "user", "content": f"Q: {i}"}] for i in range(5)
            ],
        }
    )

    def _always_failing_batch(
        self: Annotator, **kwargs: Any
    ) -> list[dict[str, Any]]:
        batch = kwargs["batch"]
        size = len(batch["idx"])
        return [
            {
                "response": None,
                "finish_reason": None,
                "num_tokens": None,
                "error": "boom",
                "error_type": "ProviderError",
            }
            for _ in range(size)
        ]

    monkeypatch.setattr(Annotator, "_process_batch", _always_failing_batch)

    with pytest.raises(
        TooManyConsecutiveFailedBatchesError, match="3 consecutive batches"
    ):
        annotator.run_annotation(
            output_dir=tmp_path / "out",
            prompt_template="Q: {text}",
            prepared_dataset=prepared_ds,
            num_retries_invalid=0,
            max_consecutive_failed_batches=3,
        )


def test_run_annotation_consecutive_failure_count_resets_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Verifies a batch that isn't fully failed resets the consecutive count,
    # so an alternating pass/fail run under the threshold does not raise.
    annotator = Annotator(
        client=DummyClient(on_error="ignore"), batch_size=1, verbose=False
    )
    prepared_ds = Dataset.from_dict(
        {
            "idx": list(range(7)),
            "text": [str(i) for i in range(7)],
            "messages": [
                [{"role": "user", "content": f"Q: {i}"}] for i in range(7)
            ],
        }
    )

    calls = {"n": 0}

    def _every_third_batch_succeeds(
        self: Annotator, **kwargs: Any
    ) -> list[dict[str, Any]]:
        calls["n"] += 1
        batch = kwargs["batch"]
        size = len(batch["idx"])
        failed = calls["n"] % 3 != 0
        return [
            {
                "response": None if failed else "ok",
                "finish_reason": None if failed else "stop",
                "num_tokens": None if failed else 1,
                "error": "boom" if failed else None,
                "error_type": "ProviderError" if failed else None,
            }
            for _ in range(size)
        ]

    monkeypatch.setattr(
        Annotator, "_process_batch", _every_third_batch_succeeds
    )

    out = annotator.run_annotation(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        prepared_dataset=prepared_ds,
        num_retries_invalid=0,
        max_consecutive_failed_batches=3,
    )

    assert len(out) == 7


def _write_progress_rows(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    progress_dir = out_dir / "progress_backup"
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / "progress.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("retry_errors", "expected_retried_texts"),
    [
        (False, set()),
        (True, {"Q: a", "Q: b"}),
        (["TypeA"], {"Q: a"}),
    ],
)
def test_run_annotation_retry_errors_selects_which_rows_are_redone(
    tmp_path: Path,
    retry_errors: bool | list[str],
    expected_retried_texts: set[str],
) -> None:
    # Verifies retry_errors=False leaves errored rows final (and never calls
    # the client for them), True redoes every errored row, and a list of
    # error types redoes only the rows of those types.
    out_dir = tmp_path / "out"
    _write_progress_rows(
        out_dir,
        [
            {
                "idx": 0,
                "response": None,
                "error": "boom",
                "error_type": "TypeA",
            },
            {
                "idx": 1,
                "response": None,
                "error": "boom",
                "error_type": "TypeB",
            },
            {"idx": 2, "response": "r2", "error": None, "error_type": None},
        ],
    )

    client = TrackingClient()
    annotator = Annotator(client=client, batch_size=2, verbose=False)
    prepared_ds = Dataset.from_dict(
        {
            "idx": [0, 1, 2],
            "text": ["a", "b", "c"],
            "messages": [
                [{"role": "user", "content": f"Q: {t}"}] for t in "abc"
            ],
        }
    )

    out = annotator.run_annotation(
        output_dir=out_dir,
        prompt_template="Q: {text}",
        prepared_dataset=prepared_ds,
        keep_idx_column=True,
        retry_errors=retry_errors,
    )

    assert sorted(out["idx"]) == [0, 1, 2]
    assert set(client.seen_prompts) == expected_retried_texts


def test_run_annotation_drops_held_back_rows_when_the_run_aborts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Verifies the rows of batches that failed entirely are not written when
    # the circuit breaker raises, so a resumed run annotates them again.
    out_dir = tmp_path / "out"
    prepared_ds = Dataset.from_dict(
        {
            "idx": list(range(4)),
            "text": [str(i) for i in range(4)],
            "messages": [
                [{"role": "user", "content": f"Q: {i}"}] for i in range(4)
            ],
        }
    )

    def _always_failing_batch(
        self: Annotator, **kwargs: Any
    ) -> list[dict[str, Any]]:
        batch = kwargs["batch"]
        size = len(batch["idx"])
        return [
            {
                "response": None,
                "finish_reason": None,
                "num_tokens": None,
                "error": "boom",
                "error_type": "ProviderError",
            }
            for _ in range(size)
        ]

    monkeypatch.setattr(Annotator, "_process_batch", _always_failing_batch)
    annotator = Annotator(
        client=DummyClient(on_error="ignore"), batch_size=1, verbose=False
    )

    with pytest.raises(TooManyConsecutiveFailedBatchesError):
        annotator.run_annotation(
            output_dir=out_dir,
            prompt_template="Q: {text}",
            prepared_dataset=prepared_ds,
            num_retries_invalid=0,
            max_consecutive_failed_batches=2,
        )

    progress_dir = out_dir / "progress_backup"
    written_rows = [
        json.loads(line)
        for pfin in progress_dir.glob("*.jsonl")
        for line in pfin.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert written_rows == []

    monkeypatch.undo()
    healthy_annotator = Annotator(
        client=DummyClient(), batch_size=1, verbose=False
    )
    result = healthy_annotator.run_annotation(
        output_dir=out_dir,
        prompt_template="Q: {text}",
        prepared_dataset=prepared_ds,
        keep_idx_column=True,
    )

    assert sorted(result["idx"]) == [0, 1, 2, 3]
    assert all(error is None for error in result["error"])


def test_run_annotation_writes_held_back_rows_once_a_later_batch_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Verifies a held-back batch's rows are not lost: they are written once a
    # later batch succeeds, not only dropped on abort.
    calls = {"n": 0}

    def _first_batch_fails(
        self: Annotator, **kwargs: Any
    ) -> list[dict[str, Any]]:
        calls["n"] += 1
        batch = kwargs["batch"]
        size = len(batch["idx"])
        if calls["n"] == 1:
            return [
                {
                    "response": None,
                    "finish_reason": None,
                    "num_tokens": None,
                    "error": "boom",
                    "error_type": "ProviderError",
                }
                for _ in range(size)
            ]
        return [
            {
                "response": "ok",
                "finish_reason": "stop",
                "num_tokens": 1,
                "error": None,
                "error_type": None,
            }
            for _ in range(size)
        ]

    monkeypatch.setattr(Annotator, "_process_batch", _first_batch_fails)
    annotator = Annotator(
        client=DummyClient(on_error="ignore"), batch_size=1, verbose=False
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

    result = annotator.run_annotation(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        prepared_dataset=prepared_ds,
        num_retries_invalid=0,
        max_consecutive_failed_batches=5,
        keep_idx_column=True,
    )

    assert sorted(result["idx"]) == [0, 1]
    errors = dict(zip(result["idx"], result["error"], strict=True))
    assert errors == {0: "boom", 1: None}


def test_run_annotation_writes_rows_right_away_when_hold_back_is_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Verifies max_consecutive_failed_batches=0 writes a failed batch's rows
    # immediately instead of holding them back for a later success.
    def _always_failing_batch(
        self: Annotator, **kwargs: Any
    ) -> list[dict[str, Any]]:
        batch = kwargs["batch"]
        size = len(batch["idx"])
        return [
            {
                "response": None,
                "finish_reason": None,
                "num_tokens": None,
                "error": "boom",
                "error_type": "ProviderError",
            }
            for _ in range(size)
        ]

    monkeypatch.setattr(Annotator, "_process_batch", _always_failing_batch)
    annotator = Annotator(
        client=DummyClient(on_error="ignore"), batch_size=1, verbose=False
    )
    prepared_ds = Dataset.from_dict(
        {
            "idx": [0],
            "text": ["a"],
            "messages": [[{"role": "user", "content": "Q: a"}]],
        }
    )

    result = annotator.run_annotation(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        prepared_dataset=prepared_ds,
        num_retries_invalid=0,
        max_consecutive_failed_batches=0,
        keep_idx_column=True,
    )

    assert result["idx"] == [0]
    assert result["error"] == ["boom"]


def test_run_annotation_summary_warns_with_error_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Verifies the end-of-run summary is a WARNING naming the error type
    # when the run produced errors.
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

    def _first_sample_errors(
        self: Annotator, **kwargs: Any
    ) -> list[dict[str, Any]]:
        batch = kwargs["batch"]
        return [
            {
                "response": None if idx == 0 else "ok",
                "finish_reason": None if idx == 0 else "stop",
                "num_tokens": None if idx == 0 else 1,
                "error": "boom" if idx == 0 else None,
                "error_type": "ProviderError" if idx == 0 else None,
            }
            for idx in batch["idx"]
        ]

    monkeypatch.setattr(Annotator, "_process_batch", _first_sample_errors)
    annotator = Annotator(client=DummyClient(), batch_size=2, verbose=False)

    with caplog.at_level(logging.INFO, logger="llm_annotator.annotator"):
        annotator.run_annotation(
            output_dir=tmp_path / "out",
            prompt_template="Q: {text}",
            prepared_dataset=prepared_ds,
            num_retries_invalid=0,
        )

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("ProviderError" in r.message for r in warnings)
    assert any("retry_errors" in r.message for r in warnings)


def test_run_annotation_summary_is_info_for_a_clean_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Verifies a run without errors logs the summary at INFO, not WARNING.
    prepared_ds = Dataset.from_dict(
        {
            "idx": [0],
            "text": ["a"],
            "messages": [[{"role": "user", "content": "Q: a"}]],
        }
    )
    annotator = Annotator(client=DummyClient(), batch_size=2, verbose=False)

    with caplog.at_level(logging.INFO, logger="llm_annotator.annotator"):
        annotator.run_annotation(
            output_dir=tmp_path / "out",
            prompt_template="Q: {text}",
            prepared_dataset=prepared_ds,
        )

    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_run_annotation_guard_rails(tmp_path: Path) -> None:
    annotator = Annotator(client=DummyClient())
    prepared_ds = Dataset.from_dict(
        {
            "idx": [0],
            "messages": [[{"role": "user", "content": "Q: a"}]],
        }
    )
    # When upload_every_n_samples is set but hub_id is not provided,
    # the code silently disables uploads (sets upload_every_n_samples=0).
    result = annotator.run_annotation(
        output_dir=tmp_path / "x2",
        prompt_template="{text}",
        prepared_dataset=prepared_ds,
        upload_every_n_samples=10,
    )
    # Verify the annotation completed successfully
    assert len(result) == 1


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
        process_pdout=p, idx_column="idx", keep_idx_column=False
    )
    assert done.column_names == ["response"]

    single = dummy_annotator._get_pfout_name(
        process_pdout=p, max_samples_per_output_file=0, processed_n_samples=0
    )
    chunked = dummy_annotator._get_pfout_name(
        process_pdout=p, max_samples_per_output_file=10, processed_n_samples=25
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
        "llm_annotator.annotator.upload_folder",
        lambda *args, **kwargs: called.append("upload"),
    )
    monkeypatch.setattr(
        "llm_annotator.annotator.upload_file",
        lambda *args, **kwargs: called.append("record"),
    )

    annotator.push_progress_to_hub(tmp_path, hub_id="me/test")
    assert called == ["repo", "branch", "upload"]


def test_push_progress_to_hub_uploads_the_selection_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Verifies the record next to the progress directory travels with the
    # backup, so a restore keeps the checks on the run's settings.
    uploads: list[dict[str, Any]] = []

    monkeypatch.setattr(
        "llm_annotator.annotator.create_repo", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "llm_annotator.annotator.create_branch", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "llm_annotator.annotator.upload_folder", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "llm_annotator.annotator.upload_file",
        lambda *a, **kw: uploads.append(kw),
    )

    progress_dir = tmp_path / "qa_progress_backup"
    progress_dir.mkdir()
    SelectionRecord(
        max_num_samples=None,
        source_rows=1,
        selected_rows=1,
        components={"prompt_template": "abc"},
    ).write(tmp_path, "qa_")

    Annotator(client=DummyClient()).push_progress_to_hub(
        progress_dir, hub_id="me/test", task_prefix="qa_"
    )

    assert len(uploads) == 1
    assert uploads[0]["path_in_repo"] == "qa_selection.json"
    assert uploads[0]["revision"] == "qa_progress_backup"
    assert uploads[0]["repo_type"] == "dataset"


def test_copy_file_prefix_copies_exactly_n_bytes(tmp_path: Path) -> None:
    src = tmp_path / "src.jsonl"
    src.write_bytes(b"0123456789")
    dest = tmp_path / "dest.jsonl"

    _copy_file_prefix(src=src, dest=dest, num_bytes=4)

    assert dest.read_bytes() == b"0123"


def test_copy_file_prefix_copies_a_shorter_source_whole(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src.jsonl"
    src.write_bytes(b"abc")
    dest = tmp_path / "dest.jsonl"

    _copy_file_prefix(src=src, dest=dest, num_bytes=100)

    assert dest.read_bytes() == b"abc"


def test_push_progress_to_hub_uploads_only_a_prefix_of_the_active_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The writer keeps growing the active file, so the backup must upload a
    # byte-exact copy of the prefix that was written when the cycle was
    # requested, not the file itself.
    folder_calls: list[dict[str, Any]] = []
    file_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        "llm_annotator.annotator.create_repo", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "llm_annotator.annotator.create_branch", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "llm_annotator.annotator.upload_folder",
        lambda *a, **kw: folder_calls.append(kw),
    )

    def _fake_upload_file(**kwargs: Any) -> None:
        content = Path(kwargs["path_or_fileobj"]).read_bytes()
        file_calls.append({**kwargs, "content": content})

    monkeypatch.setattr(
        "llm_annotator.annotator.upload_file", _fake_upload_file
    )

    progress_dir = tmp_path / "progress_backup"
    progress_dir.mkdir()
    active_path = progress_dir / "active.jsonl"
    full_content = b'{"idx": 0}\n{"idx": 1}\n{"idx": 2}\n'
    active_path.write_bytes(full_content)
    active_bytes = len(b'{"idx": 0}\n')

    Annotator(client=DummyClient()).push_progress_to_hub(
        progress_dir,
        hub_id="me/test",
        active_path=active_path,
        active_bytes=active_bytes,
    )

    assert len(folder_calls) == 1
    assert folder_calls[0]["allow_patterns"] == ["*.jsonl"]
    assert folder_calls[0]["ignore_patterns"] == ["active.jsonl"]

    assert len(file_calls) == 1
    assert file_calls[0]["path_in_repo"] == "active.jsonl"
    assert file_calls[0]["content"] == full_content[:active_bytes]
    assert file_calls[0]["content"] != full_content

    staged = progress_dir.parent / PROGRESS_UPLOAD_FILE
    assert not staged.exists()


def test_run_annotation_refuses_to_overwrite_a_hub_backup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Verifies a run that starts with no local progress files stops instead
    # of replacing the Hub backup with its own, shorter files.
    prepared_ds = Dataset.from_dict(
        {"idx": [0], "qa_messages": [[{"role": "user", "content": "Q"}]]}
    )
    branch = types.SimpleNamespace(name="qa_progress_backup")
    monkeypatch.setattr(
        "llm_annotator.annotator.list_repo_refs",
        lambda *a, **kw: types.SimpleNamespace(branches=[branch]),
    )

    with pytest.raises(ValueError) as excinfo:
        Annotator(client=DummyClient()).run_annotation(
            output_dir=tmp_path / "out",
            prompt_template="Q: {text}",
            prepared_dataset=prepared_ds,
            task_prefix="qa_",
            hub_id="me/test",
            upload_every_n_samples=1,
        )

    message = str(excinfo.value)
    assert "restore_progress_from_hub.py" in message
    assert "--task-prefix qa_" in message
    assert "overwrite=True" in message


def test_run_annotation_starts_when_the_backup_branch_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Verifies an ordinary first run, and a repository that cannot be
    # listed, both go ahead.
    prepared_ds = Dataset.from_dict(
        {"idx": [0], "messages": [[{"role": "user", "content": "Q"}]]}
    )
    monkeypatch.setattr(
        "llm_annotator.annotator.create_repo", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "llm_annotator.annotator.create_branch", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "llm_annotator.annotator.upload_folder", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "llm_annotator.annotator.upload_file", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "llm_annotator.annotator.delete_branch", lambda *a, **kw: None
    )
    monkeypatch.setattr(Dataset, "push_to_hub", lambda *a, **kw: None)

    monkeypatch.setattr(
        "llm_annotator.annotator.list_repo_refs",
        lambda *a, **kw: types.SimpleNamespace(branches=[]),
    )
    empty = Annotator(client=DummyClient()).run_annotation(
        output_dir=tmp_path / "empty",
        prompt_template="Q: {text}",
        prepared_dataset=prepared_ds,
        hub_id="me/test",
        upload_every_n_samples=1,
    )
    assert len(empty) == 1

    def _no_repo(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("no such repository")

    monkeypatch.setattr("llm_annotator.annotator.list_repo_refs", _no_repo)
    unreachable = Annotator(client=DummyClient()).run_annotation(
        output_dir=tmp_path / "unreachable",
        prompt_template="Q: {text}",
        prepared_dataset=prepared_ds,
        hub_id="me/test",
        upload_every_n_samples=1,
    )
    assert len(unreachable) == 1


def test_run_annotation_pushes_progress_to_the_prefixed_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Verifies the progress backup lands on the same prefixed branch that
    # _post_annotate deletes, so several tasks can share one hub_id.
    prepared_ds = Dataset.from_dict(
        {
            "idx": [0, 1],
            "tp_messages": [
                [{"role": "user", "content": "Q: a"}],
                [{"role": "user", "content": "Q: b"}],
            ],
        }
    )

    branches: list[str] = []
    revisions: list[str] = []
    deleted: list[str] = []

    monkeypatch.setattr(
        "llm_annotator.annotator.create_repo", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "llm_annotator.annotator.create_branch",
        lambda *a, **kw: branches.append(kw["branch"]),
    )

    def _record_upload(*args: Any, **kwargs: Any) -> None:
        # The metadata upload shares this helper and has no revision.
        if "revision" in kwargs:
            revisions.append(kwargs["revision"])

    monkeypatch.setattr(
        "llm_annotator.annotator.upload_folder", _record_upload
    )
    monkeypatch.setattr(
        "llm_annotator.annotator.delete_branch",
        lambda *a, **kw: deleted.append(kw["branch"]),
    )
    monkeypatch.setattr(
        "llm_annotator.annotator.upload_file", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "llm_annotator.annotator.list_repo_refs",
        lambda *a, **kw: types.SimpleNamespace(branches=[]),
    )
    monkeypatch.setattr(Dataset, "push_to_hub", lambda *a, **kw: None)

    Annotator(client=DummyClient()).run_annotation(
        output_dir=tmp_path / "prefixed",
        prompt_template="Q: {text}",
        prepared_dataset=prepared_ds,
        task_prefix="tp_",
        hub_id="me/test",
        upload_every_n_samples=1,
    )

    assert branches == ["tp_progress_backup"] * len(branches)
    assert revisions == ["tp_progress_backup"] * len(revisions)
    assert revisions
    assert "tp_progress_backup" in deleted


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
    [OpenAIClient, ClaudeClient, VLLMOnlineClient, VLLMOfflineClient],
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


def test_load_source_with_dataset_name_split_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Verifies _load_source defaults to the only split of a named dataset.
    annotator = Annotator(client=DummyClient(), verbose=False)

    monkeypatch.setattr(
        "llm_annotator.annotator.get_dataset_split_names",
        lambda dataset_name, config_name=None, **kwargs: ["train"],
    )
    monkeypatch.setattr(
        "llm_annotator.annotator.load_dataset",
        lambda *args, **kwargs: Dataset.from_dict({"text": ["a", "b"]}),
    )

    loaded = annotator._load_source(dataset_name="dummy/name")

    assert len(loaded) == 2


def test_load_source_split_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Verifies _load_source raises on ambiguous or unknown split names.
    annotator = Annotator(client=DummyClient(), verbose=False)

    monkeypatch.setattr(
        "llm_annotator.annotator.get_dataset_split_names",
        lambda dataset_name, config_name=None, **kwargs: [
            "train",
            "test",
        ],
    )

    with pytest.raises(ValueError, match="multiple splits"):
        annotator._load_source(dataset_name="dummy/name")

    with pytest.raises(ValueError, match="does not have a split"):
        annotator._load_source(
            dataset_name="dummy/name", dataset_split="validation"
        )


def test_load_dataset_from_local_jsonl_data_dir(tmp_path: Path) -> None:
    # Exercises the real datasets.load_dataset/get_dataset_split_names calls
    # (not mocked) against a local directory of .jsonl files, the scenario
    # 'name: "json"' + 'data_dir' is meant to support.
    (tmp_path / "data.jsonl").write_text(
        "\n".join(json.dumps({"text": t}) for t in ["a", "b", "c"]),
        encoding="utf-8",
    )
    annotator = Annotator(client=DummyClient(), verbose=False)

    source = annotator._load_source(
        dataset_name="json", data_dir=str(tmp_path)
    )
    loaded = annotator._load_dataset(
        prompt_template="Q: {text}",
        idx_column="idx",
        dataset=source,
        prompt_fields=("text",),
    )

    assert len(loaded) == 3
    assert "messages" in loaded.column_names


def test_load_source_from_local_jsonl_data_files(tmp_path: Path) -> None:
    # 'data_files' selects specific files instead of a whole directory.
    (tmp_path / "keep.jsonl").write_text(
        json.dumps({"text": "a"}) + "\n", encoding="utf-8"
    )
    (tmp_path / "skip.jsonl").write_text(
        json.dumps({"text": "b"}) + "\n", encoding="utf-8"
    )
    annotator = Annotator(client=DummyClient(), verbose=False)

    loaded = annotator._load_source(
        dataset_name="json", data_files=str(tmp_path / "keep.jsonl")
    )

    assert len(loaded) == 1


def test_run_annotation_output_schema_validation(tmp_path: Path) -> None:
    # Verifies output_schema decoding and that options cannot carry a schema.
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

    for output_schema in [None, {"type": "object"}]:
        with pytest.raises(ValueError, match="not as 'options.output_schema'"):
            annotator.run_annotation(
                output_dir=tmp_path / "b",
                prompt_template="Q: {text}",
                prepared_dataset=prepared_ds,
                options=ProviderRuntimeOptions(
                    output_schema={"type": "object"}
                ),
                output_schema=output_schema,
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


def test_run_annotation_keep_columns_and_validation_fields(
    tmp_path: Path,
) -> None:
    # Verifies run_annotation preserves requested columns and writes validation metadata.
    def my_validator(sample: dict) -> bool:
        return bool(sample.get("response"))

    options = ProviderRuntimeOptions(max_completion_tokens=64)
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
        "reasoning",
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

    # The guard applies to this call only; the setting itself is untouched,
    # so a later call with another client still uses it.
    assert annotator.num_proc == 2
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

    monkeypatch.setattr(
        Dataset,
        "save_to_disk",
        lambda self, path, **kwargs: None,
    )

    cached_ds, cached_path, cached_hub_id = annotator.prepare_data(
        output_dir=tmp_path / "hub-cache",
        prompt_template="Q: {text}",
        hub_id="owner/prepared",
    )

    assert cached_ds is dataset
    assert cached_path == tmp_path / "hub-cache" / "prepared_dataset"
    assert cached_hub_id == "owner/prepared"

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "llm_annotator.annotator.delete_branch",
        lambda *args, **kwargs: calls.append((args[0], kwargs["branch"])),
    )
    mock_prepared = Dataset.from_dict(
        {
            "idx": [0, 1],
            "messages": [[{"role": "user", "content": "Q: a"}], []],
        }
    )
    monkeypatch.setattr(
        Annotator,
        "_load_dataset",
        lambda self, **kwargs: mock_prepared,
    )
    monkeypatch.setattr(
        Dataset,
        "push_to_hub",
        lambda *args, **kwargs: None,
    )

    rebuilt_ds, rebuilt_path, rebuilt_hub_id = annotator.prepare_data(
        output_dir=tmp_path / "force-rebuild",
        prompt_template="Q: {text}",
        dataset=Dataset.from_dict({"text": ["a", "b"]}),
        hub_id="owner/prepared",
        force_data_preparation=True,
    )

    assert set(rebuilt_ds.column_names) == {"idx", "messages"}
    assert rebuilt_path is not None
    assert rebuilt_hub_id == "owner/prepared"
    assert calls == [("owner/prepared", "prepared_dataset")]


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
            hub_id="owner/output",
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
            hub_id="owner/prepared",
        )

    # Restore monkeypatches before the successful run
    monkeypatch.setattr(
        "llm_annotator.annotator.load_dataset",
        load_dataset,
    )
    monkeypatch.setattr(
        Dataset,
        "load_from_disk",
        Dataset.load_from_disk,
    )

    output_dir = tmp_path / "complete"
    output_dir.mkdir()
    # Create progress backup subdirectory with existing JSONL
    progress_dir = output_dir / "progress_backup"
    progress_dir.mkdir()
    (progress_dir / "existing.jsonl").write_text(
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
        "reasoning",
    ]


class ReasoningClient(DummyClient):
    """A client whose provider hands back a reasoning trace of its own."""

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        options: ProviderRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> Response:
        response = super().generate(
            messages=messages, options=options, gen_kwargs=gen_kwargs
        )
        return replace(response, reasoning="because of X")


def test_run_annotation_writes_reasoning_column(tmp_path: Path) -> None:
    # Verifies a provider-separated reasoning trace gets its own column.
    prepared_ds = Dataset.from_dict(
        {
            "idx": [0],
            "tp_messages": [[{"role": "user", "content": "Q: a"}]],
        }
    )

    done = Annotator(client=ReasoningClient()).run_annotation(
        output_dir=tmp_path / "with-reasoning",
        prompt_template="Q: {text}",
        prepared_dataset=prepared_ds,
        task_prefix="tp_",
    )

    assert done["tp_reasoning"] == ["because of X"]


def test_run_annotation_reasoning_column_is_none_without_a_trace(
    tmp_path: Path,
) -> None:
    # Verifies the column is still written for a provider that returns none,
    # so stacking rows from any client keeps one Arrow schema.
    prepared_ds = Dataset.from_dict(
        {
            "idx": [0],
            "messages": [[{"role": "user", "content": "Q: a"}]],
        }
    )

    done = Annotator(client=DummyClient()).run_annotation(
        output_dir=tmp_path / "no-reasoning",
        prompt_template="Q: {text}",
        prepared_dataset=prepared_ds,
    )

    assert done["reasoning"] == [None]


def test_load_progress_files_unions_mismatched_columns(
    tmp_path: Path,
    dummy_annotator: Annotator,
) -> None:
    # Verifies a run whose progress files were written by two library
    # versions still loads, with the older rows padded out.
    progress_dir = tmp_path / "progress_backup"
    progress_dir.mkdir()
    # `error` is null in every row of the first file, so that file types it
    # as null; the union has to take the second file's concrete type.
    (progress_dir / "progress_0.jsonl").write_text(
        '{"idx": 0, "response": "old", "error": null}\n', encoding="utf-8"
    )
    (progress_dir / "progress_1.jsonl").write_text(
        '{"idx": 1, "response": "new", "error": "boom",'
        ' "reasoning": "because"}\n',
        encoding="utf-8",
    )

    ds = dummy_annotator._load_progress_files(progress_dir).sort("idx")

    assert ds["response"] == ["old", "new"]
    assert ds["error"] == [None, "boom"]
    assert ds["reasoning"] == [None, "because"]


def test_prepare_data_strips_original_columns(tmp_path: Path) -> None:
    # Verifies prepare_data removes source columns not needed for inference.
    annotator = Annotator(client=DummyClient())
    ds = Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})

    prepared_ds, _, _ = annotator.prepare_data(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
    )

    assert set(prepared_ds.column_names) == {"idx", "messages"}


def test_prepare_data_keep_columns_retains_named_columns(
    tmp_path: Path,
) -> None:
    # Verifies keep_columns preserves the requested source column alongside essentials.
    annotator = Annotator(client=DummyClient())
    ds = Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})

    prepared_ds, _, _ = annotator.prepare_data(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
        keep_columns=["text"],
    )

    assert set(prepared_ds.column_names) == {"idx", "messages", "text"}
    assert "label" not in prepared_ds.column_names


def test_prepare_data_keep_columns_true_warns_and_keeps_all(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Verifies keep_columns=True suppresses stripping and emits a size warning.
    import logging

    annotator = Annotator(client=DummyClient())
    ds = Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})

    with caplog.at_level(logging.WARNING, logger="llm_annotator.annotator"):
        prepared_ds, _, _ = annotator.prepare_data(
            output_dir=tmp_path / "out",
            prompt_template="Q: {text}",
            dataset=ds,
            keep_columns=True,
        )

    assert any("disk space" in r.message for r in caplog.records)
    assert "text" in prepared_ds.column_names
    assert "label" in prepared_ds.column_names


def test_post_annotate_cleanup_respects_task_prefix(tmp_path: Path) -> None:
    # Verifies _post_annotate removes the prefixed cache dir, not a hardcoded one.
    annotator = Annotator(client=DummyClient())
    root_out = tmp_path / "out"
    root_out.mkdir()

    # Create progress subdirectory
    pdout = root_out / "progress_backup"
    pdout.mkdir()
    (pdout / "out.jsonl").write_text(
        '{"my_idx": 0, "response": "ok"}\n', encoding="utf-8"
    )

    # Create prefixed cache directory in root (should be deleted)
    prefixed_cache = root_out / "my_prepared_dataset"
    prefixed_cache.mkdir()
    (prefixed_cache / "data.arrow").write_text("x")

    # Create unprefixed cache directory in root (should NOT be deleted)
    unprefixed_cache = root_out / "prepared_dataset"
    unprefixed_cache.mkdir()
    (unprefixed_cache / "data.arrow").write_text("x")

    annotator._post_annotate(
        process_pdout=pdout,
        idx_column="my_idx",
        task_prefix="my_",
    )

    assert not prefixed_cache.exists()
    assert unprefixed_cache.exists()


def test_post_annotate_deletes_hub_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Verifies _post_annotate deletes both the upload and prepared-data Hub branches.
    annotator = Annotator(client=DummyClient())
    root_out = tmp_path / "out"
    root_out.mkdir()

    # Create progress subdirectory with JSONL
    pdout = root_out / "progress_backup"
    pdout.mkdir()
    (pdout / "out.jsonl").write_text(
        '{"idx": 0, "response": "ok"}\n', encoding="utf-8"
    )

    deleted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "llm_annotator.annotator.delete_branch",
        lambda repo_id, *, branch, repo_type: deleted.append(
            (repo_id, branch)
        ),
    )
    monkeypatch.setattr(
        Dataset,
        "push_to_hub",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "llm_annotator.annotator.upload_folder",
        lambda *args, **kwargs: None,
    )

    annotator._post_annotate(
        process_pdout=pdout,
        idx_column="idx",
        hub_id="owner/output",
        task_prefix="",
    )

    assert ("owner/output", "progress_backup") in deleted
    assert ("owner/output", "prepared_dataset") in deleted


def test_overwrite_deletes_only_the_progress_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Verifies overwrite drops the Hub backup of the discarded rows and
    # leaves the prepared-data branch, like it does on disk.
    annotator = Annotator(client=DummyClient())
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    deleted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "llm_annotator.annotator.delete_branch",
        lambda repo_id, *, branch, repo_type: deleted.append(
            (repo_id, branch)
        ),
    )

    annotator._remove_task_output(
        root_pdout=out_dir, task_prefix="qa_", hub_id="owner/output"
    )

    assert deleted == [("owner/output", "qa_progress_backup")]


def test_get_skip_idxs_repairs_truncated_last_line(
    tmp_path: Path, dummy_annotator: Annotator
) -> None:
    # Verifies a half-written final line (killed job) is dropped and removed
    # from the file so later reads see valid JSONL only.
    p = tmp_path / "out"
    p.mkdir()
    pfout = p / "out.jsonl"
    pfout.write_text(
        json.dumps({"idx": 0, "response": "a"})
        + "\n"
        + json.dumps({"idx": 1, "response": "b"})
        + "\n"
        + '{"idx": 2, "response": "hal',
        encoding="utf-8",
    )

    assert dummy_annotator._get_skip_idxs(
        process_pdout=p, idx_column="idx"
    ) == {
        0,
        1,
    }

    lines = pfout.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["idx"] for line in lines] == [0, 1]


def test_get_skip_idxs_raises_on_corrupt_earlier_line(
    tmp_path: Path, dummy_annotator: Annotator
) -> None:
    # Verifies corruption in the middle of a file is surfaced instead of
    # silently skipped: only the last line can be an interrupted write.
    p = tmp_path / "out"
    p.mkdir()
    (p / "out.jsonl").write_text(
        '{"idx": 0, "respo\n' + json.dumps({"idx": 1}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(json.JSONDecodeError):
        dummy_annotator._get_skip_idxs(process_pdout=p, idx_column="idx")


def test_get_skip_idxs_requires_idx_column(
    tmp_path: Path, dummy_annotator: Annotator
) -> None:
    # Verifies resumption fails loudly when existing output has no idx column.
    p = tmp_path / "out"
    p.mkdir()
    (p / "out.jsonl").write_text('{"response": "a"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="not found in existing output file"):
        dummy_annotator._get_skip_idxs(process_pdout=p, idx_column="idx")


def test_get_skip_idxs_reads_rows_with_the_id_in_any_position(
    tmp_path: Path, dummy_annotator: Annotator
) -> None:
    # The id column can sit anywhere in a row, since the writer emits the
    # kept source columns in their own order. Both rows count as done.
    p = tmp_path / "out"
    p.mkdir()
    (p / "out.jsonl").write_text(
        json.dumps({"idx": 0, "response": "a"})
        + "\n"
        + json.dumps({"response": "b", "idx": 1})
        + "\n",
        encoding="utf-8",
    )

    assert dummy_annotator._get_skip_idxs(
        process_pdout=p, idx_column="idx"
    ) == {0, 1}


def test_run_annotation_resumes_past_rows_with_the_id_in_any_position(
    tmp_path: Path,
) -> None:
    # End-to-end version of the above: a resumed run must only annotate the
    # row that is missing, and the final dataset must hold every row.
    out_dir = tmp_path / "out"
    progress_dir = out_dir / "progress_backup"
    progress_dir.mkdir(parents=True)
    (progress_dir / "old.jsonl").write_text(
        json.dumps({"idx": 0, "response": "old-0"})
        + "\n"
        + json.dumps({"response": "old-1", "idx": 1})
        + "\n",
        encoding="utf-8",
    )

    prepared_ds = Dataset.from_dict(
        {
            "idx": [0, 1, 2],
            "text": ["a", "b", "c"],
            "messages": [
                [{"role": "user", "content": f"Q: {t}"}] for t in "abc"
            ],
        }
    )

    done = Annotator(client=DummyClient(), batch_size=2).run_annotation(
        output_dir=out_dir,
        prompt_template="Q: {text}",
        prepared_dataset=prepared_ds,
        keep_idx_column=True,
    )

    assert sorted(done["idx"]) == [0, 1, 2]
    by_idx = dict(zip(done["idx"], done["response"]))
    assert by_idx[0] == "old-0"
    assert by_idx[1] == "old-1"
    assert by_idx[2] == "Q: c"


def test_process_batch_rejects_short_response_list(
    dummy_annotator: Annotator,
) -> None:
    # Verifies a client returning fewer responses than inputs is an error
    # instead of a silent, misaligned sample drop.
    client = cast(DummyClient, dummy_annotator.client)
    client.batch_generate = types.MethodType(  # type: ignore[method-assign]
        lambda self, *, messages, options=None, gen_kwargs=None: [
            Response(text="ok", provider=self.provider_type, model=self.model)
        ],
        client,
    )

    with pytest.raises(ValueError, match="exactly one response per input"):
        dummy_annotator._process_batch(
            batch={
                "messages": [
                    [{"role": "user", "content": "a"}],
                    [{"role": "user", "content": "b"}],
                ]
            },
            options=None,
        )


def test_process_batch_on_an_empty_batch_returns_nothing(
    dummy_annotator: Annotator,
) -> None:
    # Verifies a batch without samples returns an empty result and sends no
    # request, instead of raising an IndexError.
    client = cast(DummyClient, dummy_annotator.client)
    client.batch_generate = types.MethodType(  # type: ignore[method-assign]
        lambda self, *, messages, options=None, gen_kwargs=None: (
            _ for _ in ()
        ).throw(AssertionError("the client was called for an empty batch")),
        client,
    )

    assert (
        dummy_annotator._process_batch(batch={"messages": []}, options=None)
        == []
    )


def test_run_annotation_without_prompt_template(tmp_path: Path) -> None:
    # Verifies prepared data can be annotated without repeating the template,
    # which the prepared messages already encode.
    annotator = Annotator(client=DummyClient(), batch_size=2)
    prepared_ds = Dataset.from_dict(
        {
            "idx": [0, 1],
            "messages": [
                [{"role": "user", "content": "a"}],
                [{"role": "user", "content": "b"}],
            ],
        }
    )

    out = annotator.run_annotation(
        output_dir=tmp_path / "out",
        prepared_dataset=prepared_ds,
        keep_idx_column=True,
    )

    assert out["idx"] == [0, 1]
    assert out["response"] == ["a", "b"]


def test_generate_dataset_end_to_end(tmp_path: Path) -> None:
    # Verifies the synthetic-prompt entry point runs through to a dataset.
    annotator = Annotator(client=DummyClient(), batch_size=2)

    out = annotator.generate_dataset(
        output_dir=tmp_path / "out",
        prompts=["Tell me about cats", "Tell me about dogs"],
        keep_idx_column=True,
    )

    assert out["idx"] == [0, 1]
    assert out["response"] == ["Tell me about cats", "Tell me about dogs"]


def test_generate_dataset_rejects_an_edited_prompt_prefix(
    tmp_path: Path,
) -> None:
    annotator = Annotator(client=DummyClient(), batch_size=2)
    prompts = ["Tell me about cats", "Tell me about dogs"]
    annotator.generate_dataset(
        output_dir=tmp_path / "out",
        prompts=prompts,
        prompt_prefix="In Dutch. ",
        upload_every_n_samples=0,
    )

    with pytest.raises(ValueError, match="the prompt template changed"):
        annotator.generate_dataset(
            output_dir=tmp_path / "out",
            prompts=prompts,
            prompt_prefix="In English. ",
            upload_every_n_samples=0,
        )


def test_post_annotate_deduplicates_repeated_idxs(
    tmp_path: Path, dummy_annotator: Annotator
) -> None:
    # Verifies overlapping writers (e.g. a requeued job racing its predecessor)
    # cannot produce duplicate samples in the final dataset.
    pdout = tmp_path / "out" / "progress_backup"
    pdout.mkdir(parents=True)
    (pdout / "a.jsonl").write_text(
        json.dumps({"idx": 0, "response": "first"})
        + "\n"
        + json.dumps({"idx": 1, "response": "first"})
        + "\n",
        encoding="utf-8",
    )
    (pdout / "b.jsonl").write_text(
        json.dumps({"idx": 1, "response": "second"})
        + "\n"
        + json.dumps({"idx": 2, "response": "second"})
        + "\n",
        encoding="utf-8",
    )

    ds = dummy_annotator._post_annotate(
        process_pdout=pdout, idx_column="idx", keep_idx_column=True
    )

    assert ds["idx"] == [0, 1, 2]
    assert len(ds) == 3


def test_run_annotation_chunks_output_files_without_hub(
    tmp_path: Path,
) -> None:
    # Verifies max_samples_per_output_file caps file size even when nothing is
    # uploaded, so restarts do not have to parse one unbounded JSONL.
    annotator = Annotator(client=DummyClient(), batch_size=2)
    prepared_ds = Dataset.from_dict(
        {
            "idx": list(range(10)),
            "messages": [
                [{"role": "user", "content": f"q{idx}"}] for idx in range(10)
            ],
        }
    )

    out_dir = tmp_path / "out"
    result = annotator.run_annotation(
        output_dir=out_dir,
        prepared_dataset=prepared_ds,
        max_samples_per_output_file=4,
        keep_idx_column=True,
    )

    files = sorted((out_dir / "progress_backup").glob("*.jsonl"))
    assert [pfin.name for pfin in files] == [
        "progress_backup_0.jsonl",
        "progress_backup_1.jsonl",
        "progress_backup_2.jsonl",
    ]
    assert [
        len(pfin.read_text(encoding="utf-8").splitlines()) for pfin in files
    ] == [4, 4, 2]
    assert result["idx"] == list(range(10))


@pytest.mark.parametrize(
    ("value", "num_rows", "expected"),
    [
        ("auto", 500_000, 5000),
        ("auto", 2_000, 1000),
        (250, 500_000, 250),
        (0, 500_000, 0),
    ],
)
def test_resolve_samples_per_output_file(
    value: Any, num_rows: int, expected: int
) -> None:
    # Verifies "auto" scaling above the floor, the floor itself, and that a
    # fixed int and 0 pass through unchanged.
    assert (
        _resolve_samples_per_output_file(value, num_rows=num_rows) == expected
    )


@pytest.mark.parametrize("value", [-1, "nope"])
def test_resolve_samples_per_output_file_rejects_bad_values(
    value: Any,
) -> None:
    # Verifies a negative int and a non-"auto" string both raise.
    with pytest.raises(ValueError, match="max_samples_per_output_file"):
        _resolve_samples_per_output_file(value, num_rows=10)


def test_run_annotation_auto_chunks_by_the_resolved_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Verifies max_samples_per_output_file="auto" resolves against the
    # prepared dataset's row count and chunks the progress files accordingly.
    monkeypatch.setattr(
        "llm_annotator.annotator.MIN_AUTO_SAMPLES_PER_OUTPUT_FILE", 2
    )
    monkeypatch.setattr(
        "llm_annotator.annotator.AUTO_OUTPUT_FILE_FRACTION", 0.5
    )
    annotator = Annotator(client=DummyClient(), batch_size=2)
    prepared_ds = Dataset.from_dict(
        {
            "idx": list(range(8)),
            "messages": [
                [{"role": "user", "content": f"q{idx}"}] for idx in range(8)
            ],
        }
    )

    out_dir = tmp_path / "out"
    result = annotator.run_annotation(
        output_dir=out_dir,
        prepared_dataset=prepared_ds,
        max_samples_per_output_file="auto",
        keep_idx_column=True,
    )

    files = sorted((out_dir / "progress_backup").glob("*.jsonl"))
    assert [pfin.name for pfin in files] == [
        "progress_backup_0.jsonl",
        "progress_backup_1.jsonl",
    ]
    assert [
        len(pfin.read_text(encoding="utf-8").splitlines()) for pfin in files
    ] == [4, 4]
    assert result["idx"] == list(range(8))


def test_run_annotation_auto_below_floor_writes_one_file(
    tmp_path: Path,
) -> None:
    # Verifies that with the real constants, a run under the 1000-sample
    # floor still writes a single progress file.
    annotator = Annotator(client=DummyClient(), batch_size=2)
    prepared_ds = Dataset.from_dict(
        {
            "idx": list(range(10)),
            "messages": [
                [{"role": "user", "content": f"q{idx}"}] for idx in range(10)
            ],
        }
    )

    out_dir = tmp_path / "out"
    result = annotator.run_annotation(
        output_dir=out_dir,
        prepared_dataset=prepared_ds,
        max_samples_per_output_file="auto",
        keep_idx_column=True,
    )

    files = sorted((out_dir / "progress_backup").glob("*.jsonl"))
    assert [pfin.name for pfin in files] == ["progress_backup_0.jsonl"]
    assert result["idx"] == list(range(10))


# --- selection records ---------------------------------------------------------


def test_selection_record_read_returns_none_for_an_empty_dir(
    tmp_path: Path,
) -> None:
    assert SelectionRecord.read(tmp_path) is None


def test_selection_record_write_read_round_trip(tmp_path: Path) -> None:
    record = SelectionRecord(
        max_num_samples=10,
        source_rows=40,
        selected_rows=10,
        components={"shuffle_seed": "42", "dataset": "abc123"},
    )
    record.write(tmp_path, task_prefix="p_")

    assert (
        SelectionRecord.path(tmp_path, "p_") == tmp_path / "p_selection.json"
    )
    assert SelectionRecord.read(tmp_path, "p_") == record
    # A different (here: empty) task_prefix names a different file.
    assert SelectionRecord.read(tmp_path) is None
    stored = json.loads(
        (tmp_path / "p_selection.json").read_text(encoding="utf-8")
    )
    assert stored == {
        "max_num_samples": 10,
        "source_rows": 40,
        "selected_rows": 10,
        "components": {"shuffle_seed": "42", "dataset": "abc123"},
    }


@pytest.mark.parametrize(
    ("components", "expected_changed"),
    [
        ({"shuffle_seed": "42"}, []),
        ({"shuffle_seed": "7"}, ["shuffle_seed"]),
        ({"dataset": "abc"}, []),
        ({"dataset": "other"}, ["dataset"]),
        ({"prompt_template": "anything"}, []),
        (
            {"dataset": "other", "shuffle_seed": "7"},
            ["dataset", "shuffle_seed"],
        ),
    ],
)
def test_selection_record_changed_components(
    components: dict[str, str], expected_changed: list[str]
) -> None:
    # A setting the record does not hold is left out: an older record cannot
    # answer for it.
    record = SelectionRecord(
        max_num_samples=10,
        source_rows=40,
        selected_rows=10,
        components={"shuffle_seed": "42", "dataset": "abc"},
    )
    assert record.changed_components(components) == expected_changed


def test_prepare_data_writes_the_selection_record(tmp_path: Path) -> None:
    annotator = Annotator(client=DummyClient())
    ds = Dataset.from_dict({"text": [f"row {i}" for i in range(5)]})

    annotator.prepare_data(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
        max_num_samples=3,
        shuffle_seed=1,
    )

    record = SelectionRecord.read(tmp_path / "out")
    assert record is not None
    assert record.max_num_samples == 3
    assert record.source_rows == 5
    assert record.selected_rows == 3
    assert record.components["shuffle_seed"] == "1"
    assert record.components["reuse_idx_column"] == "False"
    assert record.components["dataset"] == dataset_signature(ds)
    assert record.components["prompt_template"] == get_hash("Q: {text}")
    assert record.components["system_message"] == "None"


def test_prepare_data_rebuilds_a_stale_local_cache(tmp_path: Path) -> None:
    # A leftover cache built for another cap must be rebuilt, not reused.
    annotator = Annotator(client=DummyClient())
    ds = Dataset.from_dict({"text": [f"row {i}" for i in range(10)]})

    first, _, _ = annotator.prepare_data(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
        max_num_samples=2,
    )
    assert len(first) == 2

    second, _, _ = annotator.prepare_data(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
        max_num_samples=4,
    )
    assert len(second) == 4


def test_prepare_data_reuses_a_cache_that_has_no_record(
    tmp_path: Path,
) -> None:
    # A cache from a version that never wrote a record is reused as is, even
    # though the new request asks for a higher cap.
    annotator = Annotator(client=DummyClient())
    ds = Dataset.from_dict({"text": [f"row {i}" for i in range(10)]})

    first, _, _ = annotator.prepare_data(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
        max_num_samples=2,
    )
    assert len(first) == 2
    SelectionRecord.path(tmp_path / "out").unlink()

    second, _, _ = annotator.prepare_data(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
        max_num_samples=4,
    )
    assert len(second) == 2

    # The settings of that run are recorded, so the next edit is caught.
    record = SelectionRecord.read(tmp_path / "out")
    assert record is not None
    assert record.components["prompt_template"] == get_hash("Q: {text}")

    third, _, _ = annotator.prepare_data(
        output_dir=tmp_path / "out",
        prompt_template="A: {text}",
        dataset=ds,
        max_num_samples=4,
    )
    assert third[0]["messages"][-1]["content"] == "A: row 0"


def _upper_case(*, dataset: Dataset) -> Dataset:
    return dataset.map(lambda row: {"text": row["text"].upper()})


def _lower_case(*, dataset: Dataset) -> Dataset:
    return dataset.map(lambda row: {"text": row["text"].lower()})


def test_callable_component_follows_the_source(
    caplog: pytest.LogCaptureFixture,
) -> None:
    same_name = _lower_case
    same_name.__qualname__ = _upper_case.__qualname__
    # Only the body differs now, so the two are told apart by their source.
    assert _callable_component(
        _upper_case, setting="preprocess_fn"
    ) != _callable_component(same_name, setting="preprocess_fn")
    assert _callable_component(None, setting="preprocess_fn") == "None"

    with caplog.at_level(logging.WARNING):
        component = _callable_component(
            functools.partial(_upper_case), setting="preprocess_fn"
        )
    assert "cannot be read" in caplog.text
    assert component == _callable_component(
        functools.partial(_upper_case), setting="preprocess_fn"
    )


# --- settings that the finished rows depend on ----------------------------------


def _finished_run(
    tmp_path: Path, **kwargs: Any
) -> tuple[Annotator, Dataset, Path]:
    """Annotate four rows so that the output directory holds progress files."""
    annotator = Annotator(client=TrackingClient(), batch_size=2)
    ds = Dataset.from_dict({"text": [f"row {i}" for i in range(4)]})
    out_dir = tmp_path / "out"
    annotator.annotate_dataset(
        output_dir=out_dir,
        prompt_template="Q: {text}",
        dataset=ds,
        upload_every_n_samples=0,
        **kwargs,
    )
    return annotator, ds, out_dir


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        ({"prompt_template": "A: {text}"}, "the prompt template changed"),
        ({"system_message": "Be brief."}, "the system message changed"),
        ({"sort_by_length": True}, "'sort_by_length' changed"),
        ({"preprocess_fn": _upper_case}, "'preprocess_fn' changed"),
        ({"idx_column": "sample_id"}, "'idx_column' changed"),
    ],
)
def test_annotate_dataset_rejects_changed_settings(
    tmp_path: Path, changed: dict[str, Any], expected: str
) -> None:
    annotator, ds, out_dir = _finished_run(tmp_path)
    kwargs: dict[str, Any] = {
        "output_dir": out_dir,
        "prompt_template": "Q: {text}",
        "dataset": ds,
        "upload_every_n_samples": 0,
        **changed,
    }

    with pytest.raises(ValueError, match=expected):
        annotator.annotate_dataset(**kwargs)

    # The finished rows are still there; nothing is removed before the
    # request is refused.
    assert list((out_dir / "progress_backup").glob("*.jsonl"))


def test_annotate_dataset_accepts_a_changed_setting_with_overwrite(
    tmp_path: Path,
) -> None:
    annotator, ds, out_dir = _finished_run(tmp_path)

    result = annotator.annotate_dataset(
        output_dir=out_dir,
        prompt_template="A: {text}",
        dataset=ds,
        upload_every_n_samples=0,
        overwrite=True,
    )
    assert result["response"] == [f"A: row {i}" for i in range(4)]


def test_prepare_data_rebuilds_for_an_edited_prompt_without_progress_files(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    annotator = Annotator(client=DummyClient())
    ds = Dataset.from_dict({"text": ["a", "b"]})
    annotator.prepare_data(
        output_dir=tmp_path / "out", prompt_template="Q: {text}", dataset=ds
    )

    with caplog.at_level(logging.INFO):
        prepared, _, _ = annotator.prepare_data(
            output_dir=tmp_path / "out",
            prompt_template="A: {text}",
            dataset=ds,
        )

    assert "the prompt template changed" in caplog.text
    assert prepared["messages"][0][-1]["content"] == "A: a"


def test_the_sample_cap_is_outside_the_recorded_settings(
    tmp_path: Path,
) -> None:
    # Only because of this can a run grow under a higher cap.
    annotator = Annotator(client=DummyClient())
    ds = Dataset.from_dict({"text": [f"row {i}" for i in range(4)]})
    annotator.prepare_data(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
        max_num_samples=2,
    )

    record = SelectionRecord.read(tmp_path / "out")
    assert record is not None
    assert "max_num_samples" not in record.components
    assert record.max_num_samples == 2


def test_annotate_dataset_rejects_a_changed_output_schema(
    tmp_path: Path,
) -> None:
    schema = {
        "type": "object",
        "properties": {"label": {"type": "string"}},
    }
    annotator, ds, out_dir = _finished_run(tmp_path, output_schema=schema)

    other = {
        "type": "object",
        "properties": {"sentiment": {"type": "string"}},
    }
    with pytest.raises(ValueError, match="the output schema changed"):
        annotator.annotate_dataset(
            output_dir=out_dir,
            prompt_template="Q: {text}",
            dataset=ds,
            upload_every_n_samples=0,
            output_schema=other,
        )


def test_run_annotation_records_the_output_schema_once(
    tmp_path: Path,
) -> None:
    annotator = Annotator(client=DummyClient(), batch_size=2)
    ds = Dataset.from_dict({"text": ["a", "b"]})
    prepared, _, _ = annotator.prepare_data(
        output_dir=tmp_path / "out", prompt_template="Q: {text}", dataset=ds
    )

    record = SelectionRecord.read(tmp_path / "out")
    assert record is not None
    assert "output_schema" not in record.components

    annotator.run_annotation(
        output_dir=tmp_path / "out",
        prepared_dataset=prepared,
        upload_every_n_samples=0,
    )

    record = SelectionRecord.read(tmp_path / "out")
    assert record is not None
    assert record.components["output_schema"] == "None"


def test_a_record_without_components_is_refused(tmp_path: Path) -> None:
    annotator, ds, out_dir = _finished_run(tmp_path)
    # What a release that only recorded the sample selection wrote.
    SelectionRecord.path(out_dir).write_text(
        json.dumps(
            {
                "max_num_samples": None,
                "shuffle_seed": None,
                "source_signature": "abc",
                "source_rows": 4,
                "selected_rows": 4,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not describe the settings"):
        annotator.annotate_dataset(
            output_dir=out_dir,
            prompt_template="Q: {text}",
            dataset=ds,
            upload_every_n_samples=0,
        )


def test_removing_the_record_keeps_the_finished_rows(tmp_path: Path) -> None:
    # The documented way out of a record that cannot be compared: the
    # finished rows stay, and the settings of this run are recorded without
    # a comparison.
    annotator = Annotator(client=TrackingClient(), batch_size=2)
    ds = Dataset.from_dict({"text": [f"row {i}" for i in range(4)]})
    out_dir = tmp_path / "out"
    annotator.annotate_dataset(
        output_dir=out_dir,
        prompt_template="Q: {text}",
        dataset=ds,
        max_num_samples=2,
        upload_every_n_samples=0,
    )
    SelectionRecord.path(out_dir).unlink()

    client = TrackingClient()
    result = Annotator(client=client, batch_size=2).annotate_dataset(
        output_dir=out_dir,
        prompt_template="A: {text}",
        dataset=ds,
        max_num_samples=4,
        upload_every_n_samples=0,
    )

    assert client.seen_prompts == ["A: row 2", "A: row 3"]
    assert sorted(result["response"]) == [
        "A: row 2",
        "A: row 3",
        "Q: row 0",
        "Q: row 1",
    ]
    record = SelectionRecord.read(out_dir)
    assert record is not None
    assert record.components["prompt_template"] == get_hash("A: {text}")


# --- growing an annotate_dataset run --------------------------------------------


def test_annotate_dataset_growth_sends_only_new_samples(
    tmp_path: Path,
) -> None:
    client = TrackingClient()
    annotator = Annotator(client=client, batch_size=2)
    ds = Dataset.from_dict({"text": [f"row {i}" for i in range(12)]})

    first = annotator.annotate_dataset(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
        max_num_samples=3,
        shuffle_seed=1,
    )
    assert len(first) == 3
    assert len(client.seen_prompts) == 3
    client.seen_prompts.clear()

    second = annotator.annotate_dataset(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
        max_num_samples=6,
        shuffle_seed=1,
    )
    assert len(second) == 6
    assert len(client.seen_prompts) == 3


def test_annotate_dataset_growth_rejects_a_changed_seed(
    tmp_path: Path,
) -> None:
    annotator = Annotator(client=DummyClient())
    ds = Dataset.from_dict({"text": [f"row {i}" for i in range(12)]})

    annotator.annotate_dataset(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
        max_num_samples=3,
        shuffle_seed=1,
    )

    with pytest.raises(ValueError, match="shuffle_seed"):
        annotator.annotate_dataset(
            output_dir=tmp_path / "out",
            prompt_template="Q: {text}",
            dataset=ds,
            max_num_samples=6,
            shuffle_seed=2,
        )


def test_annotate_dataset_growth_rejects_a_shrunk_cap(
    tmp_path: Path,
) -> None:
    annotator = Annotator(client=DummyClient())
    ds = Dataset.from_dict({"text": [f"row {i}" for i in range(12)]})

    annotator.annotate_dataset(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
        max_num_samples=6,
        shuffle_seed=1,
    )

    with pytest.raises(ValueError, match="shrank"):
        annotator.annotate_dataset(
            output_dir=tmp_path / "out",
            prompt_template="Q: {text}",
            dataset=ds,
            max_num_samples=3,
            shuffle_seed=1,
        )


def test_annotate_dataset_growth_rejects_a_changed_source(
    tmp_path: Path,
) -> None:
    annotator = Annotator(client=DummyClient())
    ds = Dataset.from_dict({"text": [f"row {i}" for i in range(12)]})

    annotator.annotate_dataset(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
        max_num_samples=3,
        shuffle_seed=1,
    )

    other = Dataset.from_dict({"text": [f"other {i}" for i in range(12)]})
    with pytest.raises(ValueError, match="source dataset changed"):
        annotator.annotate_dataset(
            output_dir=tmp_path / "out",
            prompt_template="Q: {text}",
            dataset=other,
            max_num_samples=6,
            shuffle_seed=1,
        )


def test_annotate_dataset_growth_accepts_appended_rows_without_a_shuffle(
    tmp_path: Path,
) -> None:
    client = TrackingClient()
    annotator = Annotator(client=client, batch_size=2)
    ds = Dataset.from_dict({"text": [f"row {i}" for i in range(5)]})

    annotator.annotate_dataset(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
    )
    client.seen_prompts.clear()

    grown = Dataset.from_dict(
        {"text": [f"row {i}" for i in range(5)] + ["row 5", "row 6"]}
    )
    second = annotator.annotate_dataset(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=grown,
    )
    assert len(second) == 7
    assert client.seen_prompts == ["Q: row 5", "Q: row 6"]


def test_annotate_dataset_growth_rejects_appended_rows_with_a_shuffle(
    tmp_path: Path,
) -> None:
    annotator = Annotator(client=DummyClient())
    ds = Dataset.from_dict({"text": [f"row {i}" for i in range(5)]})

    annotator.annotate_dataset(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
        shuffle_seed=1,
    )

    grown = Dataset.from_dict(
        {"text": [f"row {i}" for i in range(5)] + ["row 5"]}
    )
    with pytest.raises(ValueError, match="source dataset changed"):
        annotator.annotate_dataset(
            output_dir=tmp_path / "out",
            prompt_template="Q: {text}",
            dataset=grown,
            shuffle_seed=1,
        )


def test_annotate_dataset_overwrite_with_a_changed_seed_restarts(
    tmp_path: Path,
) -> None:
    client = TrackingClient()
    annotator = Annotator(client=client, batch_size=2)
    ds = Dataset.from_dict({"text": [f"row {i}" for i in range(12)]})

    annotator.annotate_dataset(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
        max_num_samples=6,
        shuffle_seed=1,
    )
    client.seen_prompts.clear()

    second = annotator.annotate_dataset(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
        max_num_samples=6,
        shuffle_seed=2,
        overwrite=True,
    )
    assert len(second) == 6
    assert len(client.seen_prompts) == 6


# --- reusing an existing idx column ---------------------------------------------


def test_reuse_idx_column_keeps_existing_ids(tmp_path: Path) -> None:
    annotator = Annotator(client=DummyClient())
    ds = Dataset.from_dict({"idx": [5, 9, 12], "text": ["a", "b", "c"]})

    out = annotator.annotate_dataset(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
        reuse_idx_column=True,
        keep_idx_column=True,
    )

    assert sorted(out["idx"]) == [5, 9, 12]


def test_reuse_idx_column_rejects_duplicates(tmp_path: Path) -> None:
    annotator = Annotator(client=DummyClient())
    ds = Dataset.from_dict({"idx": [1, 1, 2], "text": ["a", "b", "c"]})

    with pytest.raises(ValueError, match="duplicate values"):
        annotator.prepare_data(
            output_dir=tmp_path / "out",
            prompt_template="Q: {text}",
            dataset=ds,
            reuse_idx_column=True,
        )


def test_existing_idx_column_still_raises_without_the_reuse_flag(
    tmp_path: Path,
) -> None:
    annotator = Annotator(client=DummyClient())
    ds = Dataset.from_dict({"idx": [0, 1], "text": ["a", "b"]})

    with pytest.raises(ValueError, match="already contains a column"):
        annotator.prepare_data(
            output_dir=tmp_path / "out",
            prompt_template="Q: {text}",
            dataset=ds,
        )


def test_run_annotation_overwrite_keeps_the_selection_record(
    tmp_path: Path,
) -> None:
    annotator = Annotator(client=DummyClient())
    ds = Dataset.from_dict({"text": ["a", "b"]})

    prepared, _, _ = annotator.prepare_data(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
    )
    record_path = SelectionRecord.path(tmp_path / "out")
    assert record_path.is_file()

    annotator.run_annotation(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        prepared_dataset=prepared,
        overwrite=True,
    )

    assert record_path.is_file()


class CrashingClient(DummyClient):
    """A DummyClient that fails every request with a RuntimeError."""

    def batch_generate(
        self,
        *,
        messages: list[list[dict[str, str]]],
        options: ProviderRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> list[Response]:
        _ = messages
        _ = options
        _ = gen_kwargs
        raise RuntimeError("backend down")


def test_overwrite_keeps_the_prepared_data_of_a_crashed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Verifies a crashed overwrite run leaves its prepared data on disk and
    # that the next run reuses it instead of preparing it again.
    out_dir = tmp_path / "out"
    ds = Dataset.from_dict({"text": ["a", "b"]})
    kwargs: dict[str, Any] = {
        "output_dir": out_dir,
        "prompt_template": "Q: {text}",
        "dataset": ds,
        "upload_every_n_samples": 0,
        "overwrite": True,
    }

    with pytest.raises(RuntimeError, match="backend down"):
        Annotator(client=CrashingClient()).annotate_dataset(**kwargs)

    assert (out_dir / "prepared_dataset").is_dir()
    assert SelectionRecord.path(out_dir).is_file()

    def _no_rebuild(self: Annotator, **_kwargs: Any) -> Dataset:
        raise AssertionError("the prepared data was rebuilt")

    monkeypatch.setattr(Annotator, "_load_dataset", _no_rebuild)
    result = Annotator(client=DummyClient()).annotate_dataset(**kwargs)

    assert result["response"] == ["Q: a", "Q: b"]


def test_overwrite_spares_the_artifacts_of_another_task(
    tmp_path: Path,
) -> None:
    # Verifies two tasks can share one output_dir: overwriting the second
    # leaves the first task's progress files, prepared data, record and
    # metadata file in place.
    out_dir = tmp_path / "out"
    annotator = Annotator(client=DummyClient())
    ds = Dataset.from_dict({"text": ["a", "b"]})

    annotator.annotate_dataset(
        output_dir=out_dir,
        prompt_template="Q: {text}",
        dataset=ds,
        task_prefix="first_",
        keep_idx_column=True,
        upload_every_n_samples=0,
    )
    # A finished run removes its own prepared data, so put a file back that
    # stands for the prepared data of a task that is still running.
    (out_dir / "first_prepared_dataset").mkdir()
    (out_dir / "first_prepared_dataset" / "state.json").write_text(
        "{}", encoding="utf-8"
    )
    first_rows = sorted((out_dir / "first_progress_backup").glob("*.jsonl"))
    assert first_rows

    annotator.annotate_dataset(
        output_dir=out_dir,
        prompt_template="A: {text}",
        dataset=ds,
        task_prefix="second_",
        keep_idx_column=True,
        upload_every_n_samples=0,
        overwrite=True,
    )

    assert sorted((out_dir / "first_progress_backup").glob("*.jsonl")) == (
        first_rows
    )
    assert (out_dir / "first_prepared_dataset" / "state.json").is_file()
    assert SelectionRecord.path(out_dir, "first_").is_file()
    assert (out_dir / "metadata" / "first_annotation_metadata.json").is_file()
    assert (out_dir / "metadata" / "second_annotation_metadata.json").is_file()
    assert (out_dir / "metadata" / "_version.json").is_file()


def test_overwrite_removes_the_final_dataset_of_the_task(
    tmp_path: Path,
) -> None:
    # Verifies the shared final dataset in the root is replaced rather than
    # merged with the shards of the run before it.
    out_dir = tmp_path / "out"
    annotator = Annotator(client=DummyClient())

    annotator.annotate_dataset(
        output_dir=out_dir,
        prompt_template="Q: {text}",
        dataset=Dataset.from_dict({"text": ["a", "b", "c"]}),
        upload_every_n_samples=0,
    )
    (out_dir / "data-00000-of-00002.arrow").write_bytes(b"stale")

    annotator.annotate_dataset(
        output_dir=out_dir,
        prompt_template="Q: {text}",
        dataset=Dataset.from_dict({"text": ["a"]}),
        upload_every_n_samples=0,
        overwrite=True,
    )

    assert not (out_dir / "data-00000-of-00002.arrow").exists()
    assert Dataset.load_from_disk(out_dir)["response"] == ["Q: a"]


class _BlockingAnnotator:
    """Stand-in for the ``Annotator`` a ``_ProgressUploader`` calls back.

    ``push_progress_to_hub`` blocks on ``release`` and sets ``started`` right
    before blocking, so a test can wait for the upload to begin and then
    control exactly when it finishes.
    """

    def __init__(self, *, release: threading.Event) -> None:
        self.release = release
        self.started = threading.Event()
        self.finished = threading.Event()
        self.calls = 0

    def push_progress_to_hub(self, *args: Any, **kwargs: Any) -> None:
        _ = args
        _ = kwargs
        self.calls += 1
        self.started.set()
        assert self.release.wait(timeout=5), "test did not release in time"
        self.finished.set()


def test_progress_uploader_drops_a_request_while_one_is_running(
    tmp_path: Path,
) -> None:
    release = threading.Event()
    fake = _BlockingAnnotator(release=release)
    uploader = _ProgressUploader(
        annotator=cast(Any, fake),
        process_pdout=tmp_path,
        hub_id="me/test",
        task_prefix="",
    )
    active_path = tmp_path / "active.jsonl"
    active_path.write_text('{"idx": 0}\n', encoding="utf-8")

    uploader.request(active_path=active_path, active_bytes=1)
    assert fake.started.wait(timeout=5)
    # The first upload is still running (blocked on the event), so this
    # second request must be dropped rather than queued.
    uploader.request(active_path=active_path, active_bytes=1)

    release.set()
    uploader.close()

    assert fake.calls == 1


def test_progress_uploader_close_waits_for_an_in_flight_upload(
    tmp_path: Path,
) -> None:
    release = threading.Event()
    fake = _BlockingAnnotator(release=release)
    uploader = _ProgressUploader(
        annotator=cast(Any, fake),
        process_pdout=tmp_path,
        hub_id="me/test",
        task_prefix="",
    )
    active_path = tmp_path / "active.jsonl"
    active_path.write_text('{"idx": 0}\n', encoding="utf-8")

    uploader.request(active_path=active_path, active_bytes=1)
    assert fake.started.wait(timeout=5)
    assert not fake.finished.is_set()

    release.set()
    uploader.close()

    assert fake.finished.is_set()


def test_progress_uploader_logs_a_warning_and_swallows_the_exception(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    class _FailingAnnotator:
        def push_progress_to_hub(self, *args: Any, **kwargs: Any) -> None:
            _ = args
            _ = kwargs
            raise RuntimeError("boom")

    uploader = _ProgressUploader(
        annotator=cast(Any, _FailingAnnotator()),
        process_pdout=tmp_path,
        hub_id="me/test",
        task_prefix="",
    )
    active_path = tmp_path / "active.jsonl"
    active_path.write_text('{"idx": 0}\n', encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="llm_annotator.annotator"):
        uploader.request(active_path=active_path, active_bytes=1)
        uploader.close()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("boom" in r.message for r in warnings)


def _leftover_upload_threads() -> list[threading.Thread]:
    return [
        t
        for t in threading.enumerate()
        if t.name.startswith("progress-upload")
    ]


def test_run_annotation_joins_the_uploader_when_it_aborts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The circuit breaker must not leave the backup thread running behind it.
    monkeypatch.setattr(
        "llm_annotator.annotator.list_repo_refs",
        lambda *a, **kw: types.SimpleNamespace(branches=[]),
    )
    prepared_ds = Dataset.from_dict(
        {
            "idx": list(range(3)),
            "text": [str(i) for i in range(3)],
            "messages": [
                [{"role": "user", "content": f"Q: {i}"}] for i in range(3)
            ],
        }
    )

    def _always_failing_batch(
        self: Annotator, **kwargs: Any
    ) -> list[dict[str, Any]]:
        batch = kwargs["batch"]
        return [
            {
                "response": None,
                "finish_reason": None,
                "num_tokens": None,
                "error": "boom",
                "error_type": "ProviderError",
            }
            for _ in batch["idx"]
        ]

    monkeypatch.setattr(Annotator, "_process_batch", _always_failing_batch)
    annotator = Annotator(
        client=DummyClient(on_error="ignore"), batch_size=1, verbose=False
    )

    with pytest.raises(TooManyConsecutiveFailedBatchesError):
        annotator.run_annotation(
            output_dir=tmp_path / "out",
            prompt_template="Q: {text}",
            prepared_dataset=prepared_ds,
            num_retries_invalid=0,
            max_consecutive_failed_batches=2,
            hub_id="me/test",
            upload_every_n_samples=1,
        )

    assert _leftover_upload_threads() == []


class _RaisingClient(DummyClient):
    """A DummyClient whose batch_generate raises instead of answering."""

    def __init__(self, exc: BaseException, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._exc = exc

    def batch_generate(
        self,
        *,
        messages: list[list[dict[str, str]]],
        options: ProviderRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> list[Response]:
        raise self._exc


def test_run_annotation_joins_the_uploader_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "llm_annotator.annotator.list_repo_refs",
        lambda *a, **kw: types.SimpleNamespace(branches=[]),
    )
    prepared_ds = Dataset.from_dict(
        {"idx": [0], "messages": [[{"role": "user", "content": "Q"}]]}
    )
    client = _RaisingClient(KeyboardInterrupt())
    annotator = Annotator(client=client, batch_size=1, verbose=False)

    with pytest.raises(KeyboardInterrupt):
        annotator.run_annotation(
            output_dir=tmp_path / "out",
            prompt_template="Q: {text}",
            prepared_dataset=prepared_ds,
            hub_id="me/test",
            upload_every_n_samples=1,
        )

    assert _leftover_upload_threads() == []
    assert client.destroy_called == 1


def test_run_summary_counts_only_this_invocations_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # run_summary must report the rows and tokens this invocation produced,
    # not the rows a previous invocation already left on disk, and it must
    # treat an errored row's None token count as 0.
    out_dir = tmp_path / "out"
    _write_progress_rows(
        out_dir,
        [
            {
                "idx": 0,
                "response": "old",
                "num_tokens": 99,
                "error": None,
                "error_type": None,
            }
        ],
    )

    prepared_ds = Dataset.from_dict(
        {
            "idx": [0, 1, 2],
            "text": ["a", "b", "c"],
            "messages": [
                [{"role": "user", "content": f"Q: {t}"}] for t in "abc"
            ],
        }
    )

    def _mixed_batch(self: Annotator, **kwargs: Any) -> list[dict[str, Any]]:
        batch = kwargs["batch"]
        return [
            {
                "response": None if idx == 2 else "ok",
                "finish_reason": None if idx == 2 else "stop",
                "num_tokens": None if idx == 2 else 4,
                "error": "boom" if idx == 2 else None,
                "error_type": "ProviderError" if idx == 2 else None,
            }
            for idx in batch["idx"]
        ]

    monkeypatch.setattr(Annotator, "_process_batch", _mixed_batch)
    annotator = Annotator(
        client=DummyClient(on_error="ignore"), batch_size=2, verbose=False
    )

    annotator.run_annotation(
        output_dir=out_dir,
        prompt_template="Q: {text}",
        prepared_dataset=prepared_ds,
        num_retries_invalid=0,
    )

    metadata_path = out_dir / "metadata" / "annotation_metadata.json"
    summary = json.loads(metadata_path.read_text(encoding="utf-8"))[
        "run_summary"
    ]

    assert summary is not None
    assert set(summary) == {
        "num_rows",
        "num_output_tokens",
        "elapsed_seconds",
        "rows_per_second",
        "output_tokens_per_second",
    }
    # Only idx 1 and 2 were annotated by this invocation; idx 0 was already
    # on disk and must not be counted.
    assert summary["num_rows"] == 2
    # idx 1 contributes 4 tokens, idx 2 errored and contributes 0.
    assert summary["num_output_tokens"] == 4


def test_run_summary_is_null_when_nothing_new_is_annotated(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    _write_progress_rows(
        out_dir,
        [
            {
                "idx": 0,
                "response": "old",
                "num_tokens": 1,
                "error": None,
                "error_type": None,
            }
        ],
    )
    prepared_ds = Dataset.from_dict(
        {
            "idx": [0],
            "text": ["a"],
            "messages": [[{"role": "user", "content": "Q: a"}]],
        }
    )

    Annotator(client=DummyClient(), verbose=False).run_annotation(
        output_dir=out_dir,
        prompt_template="Q: {text}",
        prepared_dataset=prepared_ds,
    )

    metadata_path = out_dir / "metadata" / "annotation_metadata.json"
    summary = json.loads(metadata_path.read_text(encoding="utf-8"))[
        "run_summary"
    ]
    assert summary is None


def test_is_retried_error_accepts_a_single_error_type_string() -> None:
    # A plain string is treated as a one-element sequence: it matches a row
    # of that error type and no other.
    row = {"error": "boom", "error_type": "APITimeoutError"}
    assert is_retried_error(row, retry_errors="APITimeoutError") is True
    assert is_retried_error(row, retry_errors="ConnectError") is False


def test_get_skip_idxs_returns_empty_set_when_dir_is_missing(
    tmp_path: Path, dummy_annotator: Annotator
) -> None:
    # Verifies a first run, with no progress directory yet, skips nothing.
    result = dummy_annotator._get_skip_idxs(
        process_pdout=tmp_path / "missing", idx_column="idx"
    )
    assert result == set()


def test_get_skip_idxs_skips_blank_lines(
    tmp_path: Path, dummy_annotator: Annotator
) -> None:
    # Verifies a blank line between two rows is skipped without raising,
    # and the ids on the other lines are still found.
    p = tmp_path / "out"
    p.mkdir()
    (p / "out.jsonl").write_text(
        json.dumps({"idx": 1}) + "\n\n" + json.dumps({"idx": 2}) + "\n",
        encoding="utf-8",
    )

    result = dummy_annotator._get_skip_idxs(process_pdout=p, idx_column="idx")

    assert result == {1, 2}


def test_prepare_data_sort_by_length_orders_prompts(tmp_path: Path) -> None:
    # Verifies 'longest_first' puts the longest prompt first and
    # 'shortest_first' puts the shortest prompt first.
    annotator = Annotator(client=DummyClient())
    ds = Dataset.from_dict({"text": ["a", "aaaaaaaaaa", "aaa"]})

    longest_first, _, _ = annotator.prepare_data(
        output_dir=tmp_path / "longest",
        prompt_template="Q: {text}",
        dataset=ds,
        sort_by_length="longest_first",
        keep_columns=["text"],
    )
    assert longest_first["text"][0] == "aaaaaaaaaa"

    shortest_first, _, _ = annotator.prepare_data(
        output_dir=tmp_path / "shortest",
        prompt_template="Q: {text}",
        dataset=ds,
        sort_by_length="shortest_first",
        keep_columns=["text"],
    )
    assert shortest_first["text"][0] == "a"


def test_run_annotation_rejects_a_negative_max_consecutive_failed_batches(
    tmp_path: Path,
) -> None:
    # Verifies the guard rejects a negative value before any batch runs.
    annotator = Annotator(client=DummyClient())
    prepared_ds = Dataset.from_dict(
        {"idx": [0], "messages": [[{"role": "user", "content": "Q"}]]}
    )

    with pytest.raises(ValueError, match="must be 0 or a positive integer"):
        annotator.run_annotation(
            output_dir=tmp_path / "out",
            prompt_template="Q: {text}",
            prepared_dataset=prepared_ds,
            max_consecutive_failed_batches=-1,
        )


def test_run_annotation_warns_and_fails_when_prepared_data_path_is_bad(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Verifies a prepared_data_path that fails to load logs a warning naming
    # that path, and, with no other source given, the run then raises
    # because no prepared data could be resolved.
    annotator = Annotator(client=DummyClient())
    bad_path = tmp_path / "does_not_exist"

    with caplog.at_level(logging.WARNING, logger="llm_annotator.annotator"):
        with pytest.raises(ValueError, match="No prepared data found"):
            annotator.run_annotation(
                output_dir=tmp_path / "out",
                prompt_template="Q: {text}",
                prepared_data_path=str(bad_path),
            )

    assert any(str(bad_path) in r.message for r in caplog.records)


def test_generate_dataset_with_a_single_prompt_and_no_cap(
    tmp_path: Path,
) -> None:
    # Verifies a single prompt string with no max_num_samples produces
    # exactly one row.
    annotator = Annotator(client=DummyClient())
    out = annotator.generate_dataset(
        output_dir=tmp_path / "out",
        prompts="Tell me a story.",
        upload_every_n_samples=0,
    )
    assert len(out) == 1


def test_generate_dataset_rejects_an_empty_prompt_list(
    tmp_path: Path,
) -> None:
    # Verifies an empty prompt sequence is refused before any data is built.
    annotator = Annotator(client=DummyClient())

    with pytest.raises(
        ValueError, match="At least one prompt must be provided."
    ):
        annotator.generate_dataset(output_dir=tmp_path / "out", prompts=[])


def test_prepare_data_keep_columns_accepts_a_plain_string(
    tmp_path: Path,
) -> None:
    # Verifies keep_columns given as a single string keeps that one column,
    # not just an iterable of strings.
    annotator = Annotator(client=DummyClient())
    ds = Dataset.from_dict({"text": ["a", "b"], "label": ["x", "y"]})

    prepared, _, _ = annotator.prepare_data(
        output_dir=tmp_path / "out",
        prompt_template="Q: {text}",
        dataset=ds,
        keep_columns="label",
    )

    assert prepared["label"] == ["x", "y"]


def test_add_client_rejects_a_client_that_is_not_vllm_online() -> None:
    # Verifies add_client (via _add_client_locked) refuses a client whose
    # provider_type is not vllm_online, the same guard the constructor uses.
    vllm_client = object.__new__(VLLMOnlineClient)
    vllm_client.model = "fake-model"
    vllm_client.base_url = "http://worker"
    annotator = VLLMQueueAnnotator(clients=[vllm_client])

    with pytest.raises(TypeError, match="only supports vLLM server clients"):
        annotator.add_client(DummyClient())
