from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset

from llm_annotator.annotator import Annotator, VLLMQueueAnnotator
from llm_annotator.clients.base import (
    Client,
    Provider,
    ProviderRuntimeOptions,
    Response,
)
from llm_annotator.config import ClientConfig, PipelineConfig
from llm_annotator.pipeline import (
    _hosts_file_override,
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


def three_step_config(tmp_path: Path) -> PipelineConfig:
    """Build a three-step pipeline, for testing selection contiguity."""
    return PipelineConfig.model_validate(
        {
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
    )


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

    override = _hosts_file_override(config_path, hosts, None)
    assert override is not None
    assert set(override) == {"write"}
    # A command-line path means what the shell means by it, not what the
    # config directory would make of it.
    assert override["write"]["hosts_file"] == str(hosts.resolve())


def test_cli_hosts_file_without_a_vllm_step(tmp_path: Path) -> None:
    config_path = _write_mixed_config(tmp_path)
    with pytest.raises(ValueError, match="none of the steps being run"):
        _hosts_file_override(config_path, tmp_path / "hosts.txt", ["judge"])


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
        _hosts_file_override(config_path, tmp_path / "hosts.txt", None)


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


def test_queue_size_follows_the_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `queue_size` is deliberately absent from `cache_key`, so a reused pool
    # would otherwise silently keep the previous step's value.
    class PoolClient(EchoClient):
        provider_type = Provider.VLLM_ONLINE

    seen: list[int] = []
    original = Annotator.annotate_dataset

    def spy(self: Annotator, *args: Any, **kwargs: Any) -> Any:
        assert isinstance(self, VLLMQueueAnnotator)
        assert self.queue_size is not None  # resolved in __post_init__
        seen.append(self.queue_size)
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
        },
    )
    # Same cache key as step 1, so the pool is reused rather than rebuilt.
    config.steps[1].client = {"queue_size": 3}
    run_pipeline(config)
    assert seen == [8, 3]
