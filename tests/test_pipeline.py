from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset

import llm_annotator.pipeline as pipeline_mod
from llm_annotator.annotator import (
    Annotator,
    SelectionRecord,
    VLLMQueueAnnotator,
)
from llm_annotator.clients.base import (
    Client,
    Provider,
    ProviderRuntimeOptions,
    Response,
)
from llm_annotator.config import ClientConfig, PipelineConfig
from llm_annotator.pipeline import (
    STEP_ANNOTATE_SUBDIR,
    STEP_OUTPUT_SUBDIR,
    _is_complete,
    _load_input_dataset,
    _pool_source_override,
    main,
    run_pipeline,
)


class EchoClient(Client[ProviderRuntimeOptions]):
    """Client that answers with the schema's properties filled from the prompt.

    Every property of the requested schema is returned, so schema validation
    passes, and the rendered prompt is echoed back so a test can assert which
    columns were interpolated into it.
    """

    provider_type = Provider.OPENAI

    def __init__(self, model: str = "echo", **kwargs: Any) -> None:
        super().__init__(model=model, on_error="raise")
        self.destroy_called = 0
        self.init_kwargs = kwargs
        self.seen_prompts: list[str] = []

    def _process_response(self, response: str) -> Response:
        return Response(text=response, provider=self.provider_type)

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        options: ProviderRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> Response:
        _ = gen_kwargs
        prompt = messages[-1]["content"]
        self.seen_prompts.append(prompt)

        text = prompt
        if options is not None and options.json_schema is not None:
            properties = options.json_schema.get("properties", {})
            payload: dict[str, Any] = {}
            for name, spec in properties.items():
                if spec.get("type") == "integer":
                    payload[name] = 4
                else:
                    payload[name] = f"{name}::{prompt}"
            text = json.dumps(payload)

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


class BrokenJSONClient(EchoClient):
    """Client whose structured answers never parse, so nothing is ever valid."""

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        options: ProviderRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> Response:
        _ = gen_kwargs
        _ = options
        return Response(
            text="not json at all",
            stop_reason="stop",
            provider=self.provider_type,
            model=self.model,
        )


class FailingTextClient(EchoClient):
    """EchoClient that errors whenever a prompt contains one of some texts."""

    def __init__(self, *, fail_texts: frozenset[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fail_texts = fail_texts

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        options: ProviderRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> Response:
        prompt = messages[-1]["content"]
        if any(text in prompt for text in self.fail_texts):
            self.seen_prompts.append(prompt)
            return Response(
                text="",
                error="boom",
                error_type="ProviderError",
                provider=self.provider_type,
                model=self.model,
            )
        return super().generate(
            messages=messages, options=options, gen_kwargs=gen_kwargs
        )


@pytest.fixture
def built_clients(monkeypatch: pytest.MonkeyPatch) -> list[EchoClient]:
    """Route every client construction to EchoClient and record the instances."""
    created: list[EchoClient] = []

    def fake_build_client(
        self: ClientConfig, root: Path
    ) -> Client[Any] | list[Client[Any]]:
        _ = root
        client = EchoClient(model=self.model or "echo", **self.init)
        created.append(client)
        return client

    monkeypatch.setattr(ClientConfig, "build_client", fake_build_client)
    return created


def source_dataset(tmp_path: Path, num_rows: int = 4) -> Path:
    """Write a tiny source dataset to disk and return its path."""
    dataset = Dataset.from_dict(
        {"text": [f"document {i}" for i in range(num_rows)]}
    )
    path = tmp_path / "source"
    dataset.save_to_disk(str(path))
    return path


def source_jsonl_dir(tmp_path: Path, num_rows: int = 4) -> Path:
    """Write a directory of .jsonl files holding a tiny source dataset."""
    path = tmp_path / "source_jsonl"
    path.mkdir()
    (path / "data.jsonl").write_text(
        "\n".join(
            json.dumps({"text": f"document {i}"}) for i in range(num_rows)
        ),
        encoding="utf-8",
    )
    return path


def qa_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["question", "answer"],
        "properties": {
            "question": {"type": "string"},
            "answer": {"type": "string"},
        },
    }


def rating_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["rating"],
        "properties": {"rating": {"type": "integer"}},
    }


def two_step_config(tmp_path: Path, **overrides: Any) -> PipelineConfig:
    """Build a write-then-rate pipeline over a local dataset."""
    data: dict[str, Any] = {
        "output_dir": tmp_path / "out",
        "config_dir": tmp_path,
        "verbose": False,
        "dataset": {"path": source_dataset(tmp_path)},
        "client": {
            "provider": "openai",
            "model": "writer",
            "batch_size": 2,
            "num_proc": None,
        },
        "steps": [
            {
                "name": "write",
                "prompt": "Ask about: {text}",
                "output_schema": qa_schema(),
                "rename": {"question": "question_v1"},
            },
            {
                "name": "rate",
                "prompt": "Rate {question_v1} for {text}",
                "output_schema": rating_schema(),
                "client": {"model": "judge"},
            },
        ],
    }
    data.update(overrides)
    return PipelineConfig.model_validate(data)


def three_step_config(tmp_path: Path, **overrides: Any) -> PipelineConfig:
    """Build a three-step pipeline, for testing selection contiguity."""
    data: dict[str, Any] = {
        "output_dir": tmp_path / "out",
        "config_dir": tmp_path,
        "verbose": False,
        "dataset": {"path": source_dataset(tmp_path, num_rows=2)},
        "client": {
            "provider": "openai",
            "model": "m",
            "batch_size": 2,
            "num_proc": None,
        },
        "steps": [
            {"name": "one", "prompt": "1 {text}"},
            {"name": "two", "prompt": "2 {text}"},
            {"name": "three", "prompt": "3 {text}"},
        ],
    }
    data.update(overrides)
    return PipelineConfig.model_validate(data)


def growth_config(
    tmp_path: Path,
    big_source: Path,
    *,
    max_num_samples: int,
    shuffle_seed: int | None = 1,
    **overrides: Any,
) -> PipelineConfig:
    """Build a two-step pipeline over a growable source dataset."""
    return two_step_config(
        tmp_path,
        dataset={
            "path": big_source,
            "max_num_samples": max_num_samples,
            "shuffle_seed": shuffle_seed,
        },
        **overrides,
    )


def generate_first_config(
    tmp_path: Path, prompts: list[str], **overrides: Any
) -> PipelineConfig:
    """Build a single-step pipeline whose first step generates its data."""
    data: dict[str, Any] = {
        "output_dir": tmp_path / "out",
        "config_dir": tmp_path,
        "verbose": False,
        "client": {
            "provider": "openai",
            "model": "gen",
            "batch_size": 2,
            "num_proc": None,
        },
        "steps": [{"name": "make", "type": "generate", "prompts": prompts}],
    }
    data.update(overrides)
    return PipelineConfig.model_validate(data)


# --- source dataset loading ---------------------------------------------------


def test_load_input_dataset_reads_a_save_to_disk_path(tmp_path: Path) -> None:
    config = two_step_config(tmp_path)
    dataset = _load_input_dataset(config)
    assert dataset is not None
    assert len(dataset) == 4
    assert dataset.column_names == ["text"]


def test_load_input_dataset_resolves_relative_to_config_dir(
    tmp_path: Path,
) -> None:
    source_dataset(tmp_path, num_rows=3)
    config = PipelineConfig.model_validate(
        {
            "output_dir": tmp_path / "out",
            "config_dir": tmp_path,
            "dataset": {"path": "source"},
            "client": {"provider": "openai", "model": "m"},
            "steps": [{"name": "one", "prompt": "1 {text}"}],
        }
    )
    dataset = _load_input_dataset(config)
    assert dataset is not None
    assert len(dataset) == 3


def test_load_input_dataset_returns_none_for_a_hub_style_source(
    tmp_path: Path,
) -> None:
    # A Hub id or builder name is resolved later, by annotate_dataset.
    config = PipelineConfig.model_validate(
        {
            "output_dir": tmp_path / "out",
            "config_dir": tmp_path,
            "dataset": {"name": "json", "data_dir": str(tmp_path)},
            "client": {"provider": "openai", "model": "m"},
            "steps": [{"name": "one", "prompt": "1 {text}"}],
        }
    )
    assert _load_input_dataset(config) is None


def test_pipeline_runs_over_a_local_jsonl_data_dir(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    # 'name: "json"' + 'data_dir' should load a folder of .jsonl files end
    # to end, exercising the real datasets.load_dataset call.
    jsonl_dir = source_jsonl_dir(tmp_path, num_rows=3)
    config = two_step_config(
        tmp_path,
        dataset={"name": "json", "data_dir": str(jsonl_dir)},
    )
    dataset = run_pipeline(config)
    assert len(dataset) == 3
    assert {"text", "question_v1", "rating"} <= set(dataset.column_names)


def test_pipeline_runs_over_local_jsonl_data_files(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    # 'data_files' selects specific files instead of a whole directory.
    jsonl_dir = source_jsonl_dir(tmp_path, num_rows=2)
    config = two_step_config(
        tmp_path,
        dataset={
            "name": "json",
            "data_files": str(jsonl_dir / "data.jsonl"),
        },
    )
    dataset = run_pipeline(config)
    assert len(dataset) == 2


# --- chaining ----------------------------------------------------------------


def test_two_steps_chain_their_columns(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    dataset = run_pipeline(two_step_config(tmp_path))

    assert len(dataset) == 4
    columns = set(dataset.column_names)
    # The source column survives, step 1's renamed output is present, and
    # step 2 could only have produced `rating` by reading step 1's columns.
    assert {"text", "question_v1", "answer", "rating"} <= columns
    # Each step's bookkeeping is namespaced by its own task prefix.
    assert {"write_response", "rate_response"} <= columns
    assert "response" not in columns


def test_second_step_prompt_sees_first_step_output(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    run_pipeline(two_step_config(tmp_path))

    judge = built_clients[-1]
    assert judge.model == "judge"
    # The judge's prompt must contain what the writer produced, not a
    # placeholder that was never filled in.
    assert judge.seen_prompts
    for prompt in judge.seen_prompts:
        assert "question::" in prompt
        assert "{question_v1}" not in prompt


def test_rendered_messages_columns_are_pruned(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    dataset = run_pipeline(two_step_config(tmp_path))
    assert not [c for c in dataset.column_names if c.endswith("messages")]


def test_keep_messages_retains_the_column(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    config = two_step_config(tmp_path)
    config.steps[0].keep_messages = True
    dataset = run_pipeline(config)
    assert "write_messages" in dataset.column_names
    assert "rate_messages" not in dataset.column_names


# --- client lifecycle --------------------------------------------------------


def test_client_is_reused_when_settings_match(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    config = two_step_config(tmp_path)
    # Same model for both steps: the (expensive) client must be built once.
    config.steps[1].client = {"options": {"max_completion_tokens": 16}}
    run_pipeline(config)
    assert len(built_clients) == 1


def test_client_is_rebuilt_when_the_model_changes(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    run_pipeline(two_step_config(tmp_path))
    assert [c.model for c in built_clients] == ["writer", "judge"]
    # The superseded client must be released, not leaked.
    assert built_clients[0].destroy_called >= 1


def test_clients_are_destroyed_on_failure(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    config = two_step_config(tmp_path)
    config.steps[1].prompt = "Rate {does_not_exist}"
    with pytest.raises(ValueError, match="not present in dataset"):
        run_pipeline(config)
    assert built_clients[0].destroy_called >= 1


# --- column bookkeeping ------------------------------------------------------


def test_rename_reports_unknown_columns(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    config = two_step_config(tmp_path)
    config.steps[0].rename = {"nope": "renamed"}
    with pytest.raises(ValueError, match="'rename' refers to column"):
        run_pipeline(config)


def test_rename_refuses_to_clobber_an_existing_column(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    config = two_step_config(tmp_path)
    config.steps[0].rename = {"question": "text"}
    with pytest.raises(ValueError, match="already"):
        run_pipeline(config)


def test_drop_columns_removes_and_validates(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    config = two_step_config(tmp_path)
    config.steps[0].drop_columns = ["answer"]
    dataset = run_pipeline(config)
    assert "answer" not in dataset.column_names

    config = two_step_config(tmp_path / "b")
    config.steps[0].drop_columns = ["ghost"]
    with pytest.raises(ValueError, match="'drop_columns' refers to column"):
        run_pipeline(config)


def test_filter_invalid_drops_unparseable_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only the first step is broken, so the pipeline must stop there rather
    # than hand a half-empty dataset to step 2.
    def fake_build_client(
        self: ClientConfig, root: Path
    ) -> Client[Any] | list[Client[Any]]:
        _ = root
        return BrokenJSONClient(model=self.model or "echo")

    monkeypatch.setattr(ClientConfig, "build_client", fake_build_client)

    config = two_step_config(tmp_path)
    config.steps[0].filter_invalid = True
    config.steps[0].num_retries_invalid = 0
    with pytest.raises(ValueError, match="failed schema validation"):
        run_pipeline(config)


# --- resumption --------------------------------------------------------------


def test_finished_steps_are_skipped_on_rerun(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    config = two_step_config(tmp_path)
    first = run_pipeline(config)
    built_clients.clear()

    second = run_pipeline(two_step_config(tmp_path))
    # Nothing left to do, so no client is constructed at all.
    assert built_clients == []
    assert second.to_dict() == first.to_dict()


def test_overwrite_reruns_every_step(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    run_pipeline(two_step_config(tmp_path))
    built_clients.clear()

    run_pipeline(two_step_config(tmp_path, overwrite=True))
    assert [c.model for c in built_clients] == ["writer", "judge"]


@pytest.fixture
def failing_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> list[FailingTextClient]:
    """Route every client construction to a client that fails "document 0"."""
    created: list[FailingTextClient] = []

    def fake_build_client(self: ClientConfig, root: Path) -> Client[Any]:
        _ = root
        client = FailingTextClient(
            fail_texts=frozenset({"document 0"}),
            model=self.model or "echo",
            **self.init,
        )
        created.append(client)
        return client

    monkeypatch.setattr(ClientConfig, "build_client", fake_build_client)
    return created


def test_retry_errors_false_leaves_errored_rows_final(
    tmp_path: Path, failing_clients: list[FailingTextClient]
) -> None:
    first = run_pipeline(two_step_config(tmp_path))
    failing_clients.clear()

    second = run_pipeline(two_step_config(tmp_path))

    assert failing_clients == []
    assert second.to_dict() == first.to_dict()
    assert first["write_error"].count("boom") == 1


def test_retry_errors_true_redoes_the_same_rows_in_every_selected_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failing_clients: list[FailingTextClient],
) -> None:
    first = run_pipeline(two_step_config(tmp_path))

    # Only row 0 ("document 0") errors in the first step, which cascades
    # into the second step's prompt for the same row (it still names
    # "document 0", even though "question_v1" came back as None).
    assert first["write_error"].count("boom") == 1
    assert first["rate_error"].count("boom") == 1

    retried_clients: list[EchoClient] = []

    def fake_healthy_build_client(
        self: ClientConfig, root: Path
    ) -> Client[Any]:
        _ = root
        client = EchoClient(model=self.model or "echo", **self.init)
        retried_clients.append(client)
        return client

    monkeypatch.setattr(
        ClientConfig, "build_client", fake_healthy_build_client
    )

    final = run_pipeline(two_step_config(tmp_path), retry_errors=True)

    write_retry, rate_retry = retried_clients
    assert write_retry.seen_prompts == ["Ask about: document 0"]
    assert rate_retry.seen_prompts == [
        "Rate question::Ask about: document 0 for document 0"
    ]
    assert all(err is None for err in final["write_error"])
    assert all(err is None for err in final["rate_error"])


def test_step_snapshots_and_final_dataset_are_written(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    config = two_step_config(tmp_path)
    run_pipeline(config)

    assert (config.output_dir / "pipeline.json").is_file()
    assert (config.output_dir / "01-write" / "output").is_dir()
    assert (config.output_dir / "02-rate" / "output").is_dir()
    final = Dataset.load_from_disk(str(config.output_dir / "final"))
    assert len(final) == 4


# --- generate steps ----------------------------------------------------------


def test_generate_step_creates_the_dataset(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    config = PipelineConfig.model_validate(
        {
            "output_dir": tmp_path / "out",
            "config_dir": tmp_path,
            "verbose": False,
            "client": {
                "provider": "openai",
                "model": "gen",
                "batch_size": 2,
                "num_proc": None,
            },
            "steps": [
                {
                    "name": "make",
                    "type": "generate",
                    "prompts": ["Write a fact."],
                    "num_samples": 3,
                    "output_schema": qa_schema(),
                }
            ],
        }
    )
    dataset = run_pipeline(config)
    assert len(dataset) == 3
    assert {"prompt", "question", "answer"} <= set(dataset.column_names)


def test_generate_step_template_must_use_the_placeholder(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    config = PipelineConfig.model_validate(
        {
            "output_dir": tmp_path / "out",
            "config_dir": tmp_path,
            "verbose": False,
            "client": {
                "provider": "openai",
                "model": "gen",
                "num_proc": None,
            },
            "steps": [
                {
                    "name": "make",
                    "type": "generate",
                    "prompt": "A template without the placeholder",
                    "prompts": ["Write a fact."],
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="must\n?\\s*contain the"):
        run_pipeline(config)


def test_generate_step_prefix_wraps_each_prompt(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    config = PipelineConfig.model_validate(
        {
            "output_dir": tmp_path / "out",
            "config_dir": tmp_path,
            "verbose": False,
            "client": {
                "provider": "openai",
                "model": "gen",
                "num_proc": None,
            },
            "steps": [
                {
                    "name": "make",
                    "type": "generate",
                    "prompt": "In Dutch. {prompt}",
                    "prompts": ["Write a fact."],
                }
            ],
        }
    )
    run_pipeline(config)
    assert built_clients[0].seen_prompts == ["In Dutch. Write a fact."]


# --- running part of a pipeline ----------------------------------------------


def test_selected_step_runs_alone(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    config = two_step_config(tmp_path)
    run_pipeline(config, selected=["write"])

    assert (config.output_dir / "01-write" / "output").is_dir()
    assert not (config.output_dir / "02-rate" / "output").exists()
    # A partial run has not produced the pipeline's result, so it must not
    # publish one.
    assert not (config.output_dir / "final").exists()
    assert [c.model for c in built_clients] == ["writer"]


def test_step_at_a_time_matches_one_shot(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    # The property the SLURM per-step submission depends on.
    one_shot = run_pipeline(two_step_config(tmp_path / "a"))

    piecewise_config = two_step_config(tmp_path / "b")
    run_pipeline(piecewise_config, selected=["write"])
    piecewise = run_pipeline(piecewise_config, selected=["rate"])

    assert piecewise.column_names == one_shot.column_names
    assert piecewise.to_dict() == one_shot.to_dict()
    assert (piecewise_config.output_dir / "final").is_dir()


def test_second_step_reads_the_first_steps_snapshot(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    config = two_step_config(tmp_path)
    run_pipeline(config, selected=["write"])
    built_clients.clear()

    dataset = run_pipeline(config, selected=["rate"])
    # Only the judge was built, yet it saw what the writer produced.
    assert [c.model for c in built_clients] == ["judge"]
    assert "question_v1" in dataset.column_names
    assert all("question::" in p for p in built_clients[0].seen_prompts)


def test_selecting_a_step_whose_predecessor_never_ran(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    config = two_step_config(tmp_path)
    with pytest.raises(ValueError, match="Step 'write' has not run yet"):
        run_pipeline(config, selected=["rate"])


def test_unknown_step_name_is_rejected(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    config = two_step_config(tmp_path)
    with pytest.raises(ValueError, match="Unknown step"):
        run_pipeline(config, selected=["nope"])


def test_non_contiguous_selection_is_rejected(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    config = three_step_config(tmp_path)
    with pytest.raises(ValueError, match="contiguously"):
        run_pipeline(config, selected=["one", "three"])


def test_overwrite_spares_unselected_steps(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    config = two_step_config(tmp_path)
    run_pipeline(config, selected=["write"])
    marker = config.output_dir / "01-write" / "output" / "state.json"
    stamp = marker.stat().st_mtime_ns

    # Re-running the second step with overwrite must not discard the first
    # step's result, or every job in a chain would destroy its own input.
    config.overwrite = True
    run_pipeline(config, selected=["rate"])
    assert marker.stat().st_mtime_ns == stamp


# --- command line ------------------------------------------------------------


def test_cli_runs_a_config_file_end_to_end(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    # Exercises the whole argparse -> load -> run path, including the
    # config-relative resolution of prompt, schema and dataset paths.
    import yaml

    source = source_dataset(tmp_path, num_rows=2)
    (tmp_path / "write.md").write_text("Ask about: {text}", encoding="utf-8")
    (tmp_path / "qa.json").write_text(
        json.dumps(qa_schema()), encoding="utf-8"
    )

    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "out"),
                "verbose": False,
                "dataset": {"path": source.name},
                "client": {
                    "provider": "openai",
                    "model": "writer",
                    "num_proc": None,
                },
                "steps": [
                    {
                        "name": "write",
                        "prompt_file": "write.md",
                        "output_schema_file": "qa.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    main([str(config_path)])

    dataset = Dataset.load_from_disk(str(tmp_path / "out" / "final"))
    assert len(dataset) == 2
    assert {"text", "question", "answer"} <= set(dataset.column_names)


def test_cli_flags_override_the_config(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    import yaml

    source = source_dataset(tmp_path, num_rows=2)
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "ignored"),
                "verbose": False,
                "dataset": {"path": str(source)},
                "client": {
                    "provider": "openai",
                    "model": "writer",
                    "num_proc": None,
                },
                "steps": [{"name": "write", "prompt": "About: {text}"}],
            }
        ),
        encoding="utf-8",
    )

    chosen = tmp_path / "chosen"
    main([str(config_path), "--output-dir", str(chosen)])

    assert (chosen / "final").is_dir()
    assert not (tmp_path / "ignored").exists()


def _write_cli_config(tmp_path: Path, num_rows: int = 4) -> Path:
    """Write a one-step annotate config over a local source and return it."""
    import yaml

    source = source_dataset(tmp_path, num_rows=num_rows)
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "out"),
                "verbose": False,
                "dataset": {"path": str(source)},
                "client": {
                    "provider": "openai",
                    "model": "writer",
                    "num_proc": None,
                },
                "steps": [{"name": "write", "prompt": "About: {text}"}],
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_cli_max_num_samples_caps_the_run(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    config_path = _write_cli_config(tmp_path, num_rows=4)
    main([str(config_path), "--max-num-samples", "2", "--shuffle-seed", "7"])

    dataset = Dataset.load_from_disk(str(tmp_path / "out" / "final"))
    assert len(dataset) == 2

    # The run records what it actually used, so growing it later compares
    # against the resolved value rather than the file.
    snapshot = json.loads(
        (tmp_path / "out" / "pipeline.json").read_text(encoding="utf-8")
    )
    assert snapshot["dataset"]["max_num_samples"] == 2
    assert snapshot["dataset"]["shuffle_seed"] == 7


def test_cli_max_num_samples_grows_a_finished_run(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    config_path = _write_cli_config(tmp_path, num_rows=4)
    main([str(config_path), "--max-num-samples", "2"])
    main([str(config_path), "--max-num-samples", "4"])

    dataset = Dataset.load_from_disk(str(tmp_path / "out" / "final"))
    assert len(dataset) == 4
    # Two rows from the pilot, four rows minus those two from the growth.
    assert sum(len(client.seen_prompts) for client in built_clients) == 4


@pytest.mark.parametrize(
    ("cli_args", "expected"),
    [
        ([], False),
        (["--retry-errors"], True),
        (
            ["--retry-errors", "ConnectError", "APITimeoutError"],
            ["ConnectError", "APITimeoutError"],
        ),
    ],
)
def test_cli_retry_errors_flag_maps_to_run_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cli_args: list[str],
    expected: bool | list[str],
) -> None:
    config_path = _write_cli_config(tmp_path, num_rows=1)
    seen: dict[str, Any] = {}

    def fake_run_pipeline(config: PipelineConfig, **kwargs: Any) -> Dataset:
        _ = config
        seen.update(kwargs)
        return Dataset.from_dict({})

    monkeypatch.setattr(pipeline_mod, "run_pipeline", fake_run_pipeline)

    main([str(config_path), *cli_args])

    assert seen["retry_errors"] == expected


def test_cli_set_reaches_any_key(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    config_path = _write_cli_config(tmp_path, num_rows=4)
    main(
        [
            str(config_path),
            "--set",
            "dataset.max_num_samples=3",
            "--set",
            "steps.0.client.batch_size=1",
            "--set",
            "client.options.temperature=0.25",
        ]
    )

    assert len(Dataset.load_from_disk(str(tmp_path / "out" / "final"))) == 3
    snapshot = json.loads(
        (tmp_path / "out" / "pipeline.json").read_text(encoding="utf-8")
    )
    assert snapshot["steps"][0]["client"]["batch_size"] == 1
    assert snapshot["client"]["options"] == {"temperature": 0.25}


def test_cli_set_rejects_a_malformed_assignment(tmp_path: Path) -> None:
    config_path = _write_cli_config(tmp_path)
    with pytest.raises(ValueError, match="expects 'key=value'"):
        main([str(config_path), "--set", "dataset.max_num_samples"])


def test_cli_set_rejects_a_key_given_twice(tmp_path: Path) -> None:
    config_path = _write_cli_config(tmp_path)
    with pytest.raises(ValueError, match="set twice"):
        main(
            [
                str(config_path),
                "--max-num-samples",
                "2",
                "--set",
                "dataset.max_num_samples=3",
            ]
        )


def test_cli_dataset_flags_need_a_dataset_block(tmp_path: Path) -> None:
    import yaml

    config_path = tmp_path / "generate.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "out"),
                "client": {"provider": "openai", "model": "gen"},
                "steps": [
                    {"name": "make", "type": "generate", "prompts": ["x"]}
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no 'dataset' block"):
        main([str(config_path), "--max-num-samples", "2"])


def _write_mixed_config(tmp_path: Path) -> Path:
    """Write a pooled-vLLM + hosted-provider config and return its path."""
    import yaml

    config_path = tmp_path / "mixed.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "out"),
                "verbose": False,
                "dataset": {"name": "stanfordnlp/imdb", "split": "test"},
                "steps": [
                    {
                        "name": "write",
                        "prompt": "x {text}",
                        "client": {
                            "provider": "vllm_online",
                            "model": "Qwen/Qwen3-8B",
                            "engine": {
                                "tensor_parallel_size": 2,
                                "max_model_len": 8192,
                                "speculative_config": {
                                    "model": "draft",
                                    "num_speculative_tokens": 4,
                                },
                            },
                            "pool": {"servers": 4},
                        },
                    },
                    {
                        "name": "judge",
                        "prompt": "y {text}",
                        "client": {
                            "provider": "claude",
                            "model": "claude-haiku-4-5",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_cli_describe_steps_emits_json_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main([str(_write_mixed_config(tmp_path)), "--describe-steps"])

    rows = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert [r["name"] for r in rows] == ["write", "judge"]
    assert [r["kind"] for r in rows] == ["vllm_pool", "api"]
    assert rows[0]["model"] == "Qwen/Qwen3-8B"
    assert (rows[0]["servers"], rows[0]["gpus_per_vllm_server"]) == (4, 2)
    # The concurrency travels in the same line, so a submitter can size the
    # servers it is about to start.
    assert rows[0]["max_requests_per_server"] == 1024
    assert rows[0]["max_requests_in_flight"] == 4096
    assert rows[1]["max_requests_in_flight"] is None


def test_cli_url_glob_reaches_the_step_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Once --url-glob names servers for the 'write' step, that step points at
    # servers that already exist instead of needing its own pool started.
    config_path = _write_mixed_config(tmp_path)
    pattern = str(tmp_path / "pool_*" / "*.url")

    main([str(config_path), "--describe-steps", "--url-glob", pattern])

    rows = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert [r["kind"] for r in rows] == ["vllm_online", "api"]


def test_cli_serve_args_prints_one_argument_per_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The output is pasteable after `vllm serve`, spaces and all."""
    main([str(_write_mixed_config(tmp_path)), "--serve-args", "write"])

    args = capsys.readouterr().out.splitlines()
    assert args[:3] == [
        "Qwen/Qwen3-8B",
        "--served-model-name",
        "Qwen/Qwen3-8B",
    ]
    assert args[args.index("--tensor-parallel-size") + 1] == "2"
    assert args[args.index("--max-model-len") + 1] == "8192"

    # A JSON value stays one argument even though it contains no spaces here;
    # what matters is that it is not split across lines.
    spec = args[args.index("--speculative-config") + 1]
    assert json.loads(spec)["num_speculative_tokens"] == 4

    # --host and --port are the node's business, not the config's.
    assert "--host" not in args and "--port" not in args


def test_cli_serve_args_rejects_steps_with_nothing_to_serve(
    tmp_path: Path,
) -> None:
    config_path = str(_write_mixed_config(tmp_path))

    with pytest.raises(ValueError, match="hosted provider"):
        main([config_path, "--serve-args", "judge"])

    with pytest.raises(ValueError, match="no step 'nope'"):
        main([config_path, "--serve-args", "nope"])


def test_cli_describe_steps_annotates_nothing(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    config_path = _write_mixed_config(tmp_path)
    main([str(config_path), "--describe-steps"])
    assert built_clients == []
    assert not (tmp_path / "out" / "final").exists()


def test_cli_hosts_file_targets_only_the_vllm_step(tmp_path: Path) -> None:
    # This config is exactly the shape that a top-level hosts_file could not
    # express: the hosted step would inherit a pool it cannot use.
    config_path = _write_mixed_config(tmp_path)
    hosts = tmp_path / "hosts.txt"
    hosts.write_text("http://a:8000/v1\n", encoding="utf-8")

    override = _pool_source_override(config_path, hosts, None, None)
    assert override is not None
    assert set(override) == {"write"}
    # A command-line path means what the shell means by it, not what the
    # config directory would make of it.
    assert override["write"]["hosts_file"] == str(hosts.resolve())


def test_cli_hosts_file_without_a_vllm_step(tmp_path: Path) -> None:
    config_path = _write_mixed_config(tmp_path)
    with pytest.raises(ValueError, match="none of the steps being run"):
        _pool_source_override(
            config_path, tmp_path / "hosts.txt", None, ["judge"]
        )


def test_cli_url_glob_targets_only_the_vllm_step(tmp_path: Path) -> None:
    config_path = _write_mixed_config(tmp_path)

    override = _pool_source_override(config_path, None, "pool_*/*.url", None)

    assert override is not None
    assert set(override) == {"write"}
    assert "*" in override["write"]["url_glob"]
    assert Path(override["write"]["url_glob"]).is_absolute()


def test_cli_url_glob_absolute_pattern_is_kept_verbatim(
    tmp_path: Path,
) -> None:
    # An already-absolute pattern must not be touched beyond expanduser: the
    # wildcard has to survive so resolve_base_urls can still glob it later.
    config_path = _write_mixed_config(tmp_path)
    pattern = str(tmp_path / "pool_*" / "*.url")

    override = _pool_source_override(config_path, None, pattern, None)

    assert override is not None
    assert override["write"]["url_glob"] == pattern


def test_cli_hosts_file_and_url_glob_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    config_path = _write_mixed_config(tmp_path)
    with pytest.raises(ValueError, match="both name the pool's servers"):
        _pool_source_override(
            config_path, tmp_path / "hosts.txt", "pool_*/*.url", None
        )


def test_cli_hosts_file_refuses_two_models_on_one_pool(tmp_path: Path) -> None:
    # One set of servers serves one model, so this must fail loudly at submit
    # time rather than at inference time deep inside step two.
    import yaml

    config_path = tmp_path / "two_models.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "out"),
                "dataset": {"name": "stanfordnlp/imdb", "split": "test"},
                "client": {
                    "provider": "vllm_online",
                    "model": "Qwen/Qwen3-8B",
                },
                "steps": [
                    {"name": "a", "prompt": "x {text}"},
                    {
                        "name": "b",
                        "prompt": "y {text}",
                        "client": {"model": "meta-llama/Llama-3.3-70B"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different models"):
        _pool_source_override(config_path, tmp_path / "hosts.txt", None, None)


def test_cli_steps_flag_runs_one_step(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    import yaml

    source = source_dataset(tmp_path, num_rows=2)
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "out"),
                "verbose": False,
                "dataset": {"path": str(source)},
                "client": {
                    "provider": "openai",
                    "model": "m",
                    "num_proc": None,
                },
                "steps": [
                    {"name": "one", "prompt": "1 {text}"},
                    {"name": "two", "prompt": "2 {text}"},
                ],
            }
        ),
        encoding="utf-8",
    )

    main([str(config_path), "--steps", "one"])
    assert (tmp_path / "out" / "01-one" / "output").is_dir()
    assert not (tmp_path / "out" / "final").exists()

    main([str(config_path), "--steps", "two"])
    assert (tmp_path / "out" / "final").is_dir()


# --- annotator wiring --------------------------------------------------------


def test_batch_size_follows_the_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A reused client must still pick up the next step's batch size.
    seen: list[int] = []
    original = Annotator.annotate_dataset

    def spy(self: Annotator, *args: Any, **kwargs: Any) -> Any:
        seen.append(self.batch_size)
        return original(self, *args, **kwargs)

    def fake_build_client(
        self: ClientConfig, root: Path
    ) -> Client[Any] | list[Client[Any]]:
        _ = root
        return EchoClient(model=self.model or "echo")

    monkeypatch.setattr(ClientConfig, "build_client", fake_build_client)
    monkeypatch.setattr(Annotator, "annotate_dataset", spy)

    config = two_step_config(tmp_path)
    config.steps[1].client = {"batch_size": 1}
    run_pipeline(config)
    assert seen == [2, 1]


def test_max_consecutive_failed_batches_follows_the_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Verifies StepConfig.max_consecutive_failed_batches reaches
    # annotate_dataset, defaulting for one step and overridden for the next.
    seen: list[int] = []
    original = Annotator.annotate_dataset

    def spy(self: Annotator, *args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["max_consecutive_failed_batches"])
        return original(self, *args, **kwargs)

    def fake_build_client(
        self: ClientConfig, root: Path
    ) -> Client[Any] | list[Client[Any]]:
        _ = root
        return EchoClient(model=self.model or "echo")

    monkeypatch.setattr(ClientConfig, "build_client", fake_build_client)
    monkeypatch.setattr(Annotator, "annotate_dataset", spy)

    config = two_step_config(tmp_path)
    config.steps[1].max_consecutive_failed_batches = 3
    run_pipeline(config)
    assert seen == [10, 3]


def test_queue_settings_follow_the_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `queue_size` is deliberately absent from `cache_key`, so a reused pool
    # would otherwise silently keep the previous step's value.
    class PoolClient(EchoClient):
        provider_type = Provider.VLLM_ONLINE

    seen: list[tuple[int, int]] = []
    original = Annotator.annotate_dataset

    def spy(self: Annotator, *args: Any, **kwargs: Any) -> Any:
        assert isinstance(self, VLLMQueueAnnotator)
        assert self.queue_size is not None  # resolved in __post_init__
        seen.append((self.queue_size, self.max_concurrent_batches_per_client))
        return original(self, *args, **kwargs)

    def fake_build_client(
        self: ClientConfig, root: Path
    ) -> Client[Any] | list[Client[Any]]:
        _ = root
        return [PoolClient(model=self.model or "echo") for _ in range(2)]

    monkeypatch.setattr(ClientConfig, "build_client", fake_build_client)
    monkeypatch.setattr(Annotator, "annotate_dataset", spy)

    config = two_step_config(
        tmp_path,
        client={
            "provider": "vllm_online",
            "model": "m",
            "batch_size": 2,
            "num_proc": None,
            "base_urls": ["http://a:8000/v1", "http://b:8000/v1"],
            "queue_size": 8,
            "max_concurrent_batches_per_client": 2,
        },
    )
    # Same cache key as step 1, so the pool is reused rather than rebuilt.
    config.steps[1].client = {
        "queue_size": 3,
        "max_concurrent_batches_per_client": 1,
    }
    run_pipeline(config)
    assert seen == [(8, 2), (3, 1)]


# --- growing a run -------------------------------------------------------------


def test_pipeline_growth_annotates_only_new_rows(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    big_source = source_dataset(tmp_path / "growth", num_rows=40)

    first = run_pipeline(
        growth_config(tmp_path, big_source, max_num_samples=10)
    )
    assert len(first) == 10
    first_texts = set(first["text"])
    built_clients.clear()

    second_config = growth_config(tmp_path, big_source, max_num_samples=20)
    second = run_pipeline(second_config)

    writer, judge = built_clients
    assert len(writer.seen_prompts) == 10
    assert len(judge.seen_prompts) == 10

    assert len(second) == 20
    second_texts = second["text"]
    assert len(set(second_texts)) == 20
    assert first_texts <= set(second_texts)

    assert "idx" not in second.column_names
    step1_output = Dataset.load_from_disk(
        str(second_config.step_dir(0) / "output")
    )
    assert "idx" in step1_output.column_names

    # Every judge prompt embeds both 'question_v1' (rendered by step 1) and
    # 'text' (the source column) for what must be the same source row; a
    # misalignment between the growth run's old and new rows would show up
    # as two different document ids in one prompt.
    for prompt in judge.seen_prompts:
        ids = set(re.findall(r"document \d+", prompt))
        assert len(ids) == 1


def test_pipeline_growth_third_run_with_no_change_builds_nothing(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    big_source = source_dataset(tmp_path / "growth", num_rows=40)
    run_pipeline(growth_config(tmp_path, big_source, max_num_samples=10))
    run_pipeline(growth_config(tmp_path, big_source, max_num_samples=20))
    built_clients.clear()

    run_pipeline(growth_config(tmp_path, big_source, max_num_samples=20))
    assert built_clients == []


def test_pipeline_growth_step_at_a_time_matches_one_shot(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    big_source = source_dataset(tmp_path / "growth", num_rows=40)

    one_shot_dir = tmp_path / "one-shot"
    run_pipeline(growth_config(one_shot_dir, big_source, max_num_samples=10))
    one_shot = run_pipeline(
        growth_config(one_shot_dir, big_source, max_num_samples=20)
    )

    piecewise_dir = tmp_path / "piecewise"
    run_pipeline(growth_config(piecewise_dir, big_source, max_num_samples=10))
    piecewise_config = growth_config(
        piecewise_dir, big_source, max_num_samples=20
    )
    run_pipeline(piecewise_config, selected=["write"])
    piecewise = run_pipeline(piecewise_config, selected=["rate"])

    assert piecewise.column_names == one_shot.column_names
    assert piecewise.to_dict() == one_shot.to_dict()


def test_pipeline_growth_selecting_only_the_stale_downstream_step_raises(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    big_source = source_dataset(tmp_path / "growth", num_rows=40)
    run_pipeline(growth_config(tmp_path, big_source, max_num_samples=10))

    with pytest.raises(ValueError, match="finished with other settings"):
        run_pipeline(
            growth_config(tmp_path, big_source, max_num_samples=20),
            selected=["rate"],
        )


def test_pipeline_growth_rejects_a_changed_seed(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    big_source = source_dataset(tmp_path / "growth", num_rows=40)
    run_pipeline(growth_config(tmp_path, big_source, max_num_samples=10))

    with pytest.raises(ValueError, match="shuffle_seed"):
        run_pipeline(
            growth_config(
                tmp_path, big_source, max_num_samples=20, shuffle_seed=2
            )
        )


def test_pipeline_growth_rejects_a_shrunk_cap(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    big_source = source_dataset(tmp_path / "growth", num_rows=40)
    run_pipeline(growth_config(tmp_path, big_source, max_num_samples=20))

    with pytest.raises(ValueError, match="shrank"):
        run_pipeline(growth_config(tmp_path, big_source, max_num_samples=10))


def test_pipeline_growth_recovers_after_a_rejected_attempt(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    big_source = source_dataset(tmp_path / "growth", num_rows=40)
    run_pipeline(growth_config(tmp_path, big_source, max_num_samples=10))

    with pytest.raises(ValueError, match="shuffle_seed"):
        run_pipeline(
            growth_config(
                tmp_path, big_source, max_num_samples=20, shuffle_seed=2
            )
        )

    built_clients.clear()
    final = run_pipeline(
        growth_config(tmp_path, big_source, max_num_samples=10)
    )
    assert len(final) == 10
    # Nothing changed relative to what already finished, so no client sends
    # any prompt again.
    for client in built_clients:
        assert client.seen_prompts == []


def test_pipeline_growth_recovers_from_a_crash_during_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    built_clients: list[EchoClient],
) -> None:
    big_source = source_dataset(tmp_path / "growth", num_rows=40)

    # The pilot run is unaffected: only the extension run's 'judge' client
    # must fail, and only on its first batch.
    run_pipeline(growth_config(tmp_path, big_source, max_num_samples=10))
    built_clients.clear()

    class FlakyClient(EchoClient):
        """Fails the first batch a 'judge' instance is asked to answer."""

        fail_next = True

        def batch_generate(
            self,
            *,
            messages: list[list[dict[str, str]]],
            options: ProviderRuntimeOptions | None = None,
            gen_kwargs: dict[str, Any] | None = None,
        ) -> list[Response]:
            if FlakyClient.fail_next and self.model == "judge":
                FlakyClient.fail_next = False
                raise RuntimeError("boom")
            return super().batch_generate(
                messages=messages, options=options, gen_kwargs=gen_kwargs
            )

    created: list[EchoClient] = []

    def fake_build_client(
        self: ClientConfig, root: Path
    ) -> Client[Any] | list[Client[Any]]:
        _ = root
        client = FlakyClient(model=self.model or "echo")
        created.append(client)
        return client

    monkeypatch.setattr(ClientConfig, "build_client", fake_build_client)

    with pytest.raises(RuntimeError, match="boom"):
        run_pipeline(growth_config(tmp_path, big_source, max_num_samples=20))

    assert [c.model for c in created] == ["writer", "judge"]
    writer_before_crash = created[0]
    assert len(writer_before_crash.seen_prompts) == 10
    created.clear()

    final = run_pipeline(
        growth_config(tmp_path, big_source, max_num_samples=20)
    )
    assert len(final) == 20
    assert len(set(final["text"])) == 20
    # Step 1 already finished (and was saved) during the crashed run, so
    # resuming only rebuilds step 2's client; it sends the 10 rows it never
    # got to before the crash, not step 1's rows again.
    assert [c.model for c in created] == ["judge"]
    assert len(created[0].seen_prompts) == 10


def test_legacy_run_without_a_selection_record_blocks_growth(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    config = two_step_config(tmp_path)
    run_pipeline(config)

    record_path = SelectionRecord.path(
        config.step_dir(0) / STEP_ANNOTATE_SUBDIR,
        config.steps[0].resolved_task_prefix(),
    )
    assert record_path.is_file()
    record_path.unlink()

    grown = two_step_config(
        tmp_path,
        dataset={"path": tmp_path / "source", "max_num_samples": 2},
    )
    with pytest.raises(
        ValueError, match="did not record its sample selection"
    ):
        run_pipeline(grown)

    # A second attempt raises again: the failed run must not have rewritten
    # pipeline.json with the new, unrecorded cap.
    with pytest.raises(
        ValueError, match="did not record its sample selection"
    ):
        run_pipeline(grown)

    # Restoring the original settings still works: nothing changed relative
    # to what pipeline.json remembers.
    restored = run_pipeline(two_step_config(tmp_path))
    assert len(restored) == 4

    # overwrite=True on a selection that includes step 1 skips the guard.
    overwritten = run_pipeline(
        two_step_config(
            tmp_path,
            overwrite=True,
            dataset={"path": tmp_path / "source", "max_num_samples": 2},
        )
    )
    assert len(overwritten) == 2


def test_generate_step_growth_sends_only_new_prompts(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    prompts = [f"Write fact {i}." for i in range(5)]
    first = run_pipeline(generate_first_config(tmp_path, prompts))
    assert len(first) == 5
    built_clients.clear()

    grown_prompts = prompts + [f"Write fact {i}." for i in range(5, 8)]
    second = run_pipeline(generate_first_config(tmp_path, grown_prompts))

    assert len(second) == 8
    assert len(built_clients) == 1
    assert built_clients[0].seen_prompts == grown_prompts[5:]


def test_generate_step_editing_a_prompt_raises(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    prompts = [f"Write fact {i}." for i in range(5)]
    run_pipeline(generate_first_config(tmp_path, prompts))

    edited = list(prompts)
    edited[2] = "A different prompt entirely."
    with pytest.raises(ValueError, match="source dataset changed"):
        run_pipeline(generate_first_config(tmp_path, edited))


# --- editing a step's settings --------------------------------------------------


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        ({"prompt": "Ask something else: {text}"}, "the prompt template"),
        ({"system_prompt": "Be brief."}, "the system message"),
        ({"sort_by_length": True}, "'sort_by_length' changed"),
    ],
)
def test_pipeline_rejects_an_edited_step(
    tmp_path: Path,
    built_clients: list[EchoClient],
    edit: dict[str, Any],
    expected: str,
) -> None:
    run_pipeline(two_step_config(tmp_path))
    built_clients.clear()

    edited = two_step_config(tmp_path)
    for name, value in edit.items():
        setattr(edited.steps[0], name, value)

    with pytest.raises(ValueError, match=expected):
        run_pipeline(edited)

    # Refused before anything is removed, and without a model call.
    assert built_clients == []
    assert _is_complete(edited.step_dir(0) / STEP_OUTPUT_SUBDIR)
    assert _is_complete(edited.step_dir(1) / STEP_OUTPUT_SUBDIR)


def test_pipeline_rejects_an_edited_output_schema(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    run_pipeline(two_step_config(tmp_path))

    edited = two_step_config(tmp_path)
    edited.steps[1].output_schema = {
        "type": "object",
        "required": ["score"],
        "properties": {"score": {"type": "integer"}},
    }

    with pytest.raises(ValueError, match="the output schema changed"):
        run_pipeline(edited)


def test_pipeline_overwrite_reruns_the_edited_step_and_the_next_one(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    run_pipeline(three_step_config(tmp_path))
    built_clients.clear()

    edited = three_step_config(tmp_path, overwrite=True)
    edited.steps[1].prompt = "2! {text}"
    run_pipeline(edited, selected=["two", "three"])

    seen = [
        prompt for client in built_clients for prompt in client.seen_prompts
    ]
    assert [prompt for prompt in seen if prompt.startswith("2! ")]
    # The step before the edited one keeps its finished result.
    assert not [prompt for prompt in seen if prompt.startswith("1 ")]


def test_pipeline_step_outside_the_selection_must_match_too(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    run_pipeline(three_step_config(tmp_path))

    edited = three_step_config(tmp_path)
    edited.steps[1].prompt = "2! {text}"

    with pytest.raises(ValueError, match="finished with other settings"):
        run_pipeline(edited, selected=["three"])


def _edited_write_config(tmp_path: Path, **overrides: Any) -> PipelineConfig:
    """Build the two-step pipeline with another prompt in its first step."""
    config = two_step_config(tmp_path, **overrides)
    config.steps[0].prompt = "Ask something else about: {text}"
    return config


def test_pipeline_refuses_a_step_whose_input_was_annotated_again(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    run_pipeline(two_step_config(tmp_path))

    # The way out of an edited prompt: 'write' is annotated again from
    # scratch, so the questions that 'rate' judged no longer exist.
    run_pipeline(_edited_write_config(tmp_path, overwrite=True), ["write"])
    built_clients.clear()

    with pytest.raises(
        ValueError, match="annotated again from scratch"
    ) as exc:
        run_pipeline(_edited_write_config(tmp_path))

    assert "--steps rate --overwrite" in str(exc.value)
    assert built_clients == []

    # That command annotates 'rate' against the new questions, and leaves
    # 'write' alone.
    result = run_pipeline(
        _edited_write_config(tmp_path, overwrite=True), ["rate"]
    )
    assert [client.model for client in built_clients] == ["judge"]
    assert all(
        "Ask something else about" in prompt
        for prompt in built_clients[0].seen_prompts
    )
    assert len(result) == 4


def test_pipeline_growth_keeps_a_later_step_on_its_rows(
    tmp_path: Path, built_clients: list[EchoClient]
) -> None:
    # A step that resumes (rather than starting over) leaves the steps after
    # it on the rows they already annotated.
    big_source = source_dataset(tmp_path / "growth", num_rows=40)
    run_pipeline(growth_config(tmp_path, big_source, max_num_samples=10))
    built_clients.clear()

    grown = run_pipeline(
        growth_config(tmp_path, big_source, max_num_samples=20)
    )

    assert len(grown) == 20
    assert [len(client.seen_prompts) for client in built_clients] == [10, 10]
