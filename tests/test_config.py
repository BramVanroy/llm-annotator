from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llm_annotator.config import (
    ClientConfig,
    PipelineConfig,
    StepConfig,
    load_config_file,
    load_pipeline_config,
)


EXAMPLE_DIR = Path(__file__).parent.parent / "examples" / "pipeline-qa"


def minimal_config(**overrides: Any) -> dict[str, Any]:
    """Build the smallest config dict that validates, with optional overrides."""
    data: dict[str, Any] = {
        "output_dir": "outputs/test",
        "dataset": {"name": "stanfordnlp/imdb", "split": "test"},
        "client": {"provider": "openai", "model": "gpt-4o-mini"},
        "steps": [{"name": "classify", "prompt": "Rate: {text}"}],
    }
    data.update(overrides)
    return data


def write_config(tmp_path: Path, data: dict[str, Any], suffix: str) -> Path:
    """Write a config dict to tmp_path in the given format and return its path."""
    pfout = tmp_path / f"config{suffix}"
    if suffix == ".json":
        pfout.write_text(json.dumps(data), encoding="utf-8")
    else:
        import yaml

        pfout.write_text(yaml.safe_dump(data), encoding="utf-8")
    return pfout


# --- loading -----------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_load_config_file_formats(tmp_path: Path, suffix: str) -> None:
    # Both JSON and YAML files decode to the same mapping.
    path = write_config(tmp_path, minimal_config(), suffix)
    assert load_config_file(path) == minimal_config()


def test_load_config_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_config_file(tmp_path / "nope.yaml")


def test_load_config_file_rejects_non_mapping(tmp_path: Path) -> None:
    pfout = tmp_path / "config.json"
    pfout.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping at the top level"):
        load_config_file(pfout)


def test_config_dir_defaults_to_config_location(tmp_path: Path) -> None:
    # Relative paths must resolve against the config file, not the cwd.
    path = write_config(tmp_path, minimal_config(), ".yaml")
    config = load_pipeline_config(path)
    assert config.config_dir == tmp_path.resolve()


def test_overrides_take_precedence(tmp_path: Path) -> None:
    path = write_config(tmp_path, minimal_config(), ".yaml")
    config = load_pipeline_config(
        path, overrides={"output_dir": "elsewhere", "overwrite": True}
    )
    assert config.output_dir == Path("elsewhere")
    assert config.overwrite is True


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    # extra="forbid" turns a typo into an error instead of a silent no-op.
    path = write_config(tmp_path, minimal_config(outupt_dir="typo"), ".yaml")
    with pytest.raises(ValueError, match="outupt_dir"):
        load_pipeline_config(path)


def test_example_config_json_yaml_parity() -> None:
    # The shipped example exists in both formats and must describe one pipeline.
    from_yaml = load_pipeline_config(EXAMPLE_DIR / "config.yaml")
    from_json = load_pipeline_config(EXAMPLE_DIR / "config.json")
    assert from_yaml.model_dump() == from_json.model_dump()


def test_example_config_resolves_its_files() -> None:
    config = load_pipeline_config(EXAMPLE_DIR / "config.yaml")
    root = config.config_dir
    first, second = config.steps

    assert "{text}" in (first.resolved_prompt(root) or "")
    assert first.resolved_system_prompt(root)
    assert first.resolved_output_schema(root) is not None
    # Step 2 reads the columns step 1 renamed, which is the point of chaining.
    prompt = second.resolved_prompt(root) or ""
    assert "{question_v1}" in prompt
    assert "{answer_v1}" in prompt


def test_json_catalog_system_prompt_is_rendered_to_text(
    tmp_path: Path,
) -> None:
    """JSON catalog files should be converted to prompt-readable text."""
    catalog_path = tmp_path / "taxonomy.json"
    catalog_path.write_text(
        json.dumps(
            {
                "instruction": "Use only the code.",
                "categories": [
                    {"code": "alpha", "description": "First category."},
                    {"code": "beta", "description": "Second category."},
                ],
            }
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps(
            minimal_config(
                steps=[
                    {
                        "name": "classify",
                        "prompt": "Classify: {text}",
                        "system_prompt_file": str(catalog_path),
                    }
                ]
            )
        ),
        encoding="utf-8",
    )
    rendered = (
        load_pipeline_config(config_path)
        .steps[0]
        .resolved_system_prompt(tmp_path)
    )

    assert rendered is not None
    assert "Use only the code." in rendered
    assert "- `alpha` — First category." in rendered
    assert "- `beta` — Second category." in rendered


# --- provider handling -------------------------------------------------------


@pytest.mark.parametrize(
    "given",
    ["openai", "claude", "vllm_online", "vllm_offline"],
)
def test_provider_canonical_names_accepted(given: str) -> None:
    client = ClientConfig.model_validate({"provider": given, "model": "m"})
    assert client.provider == given


@pytest.mark.parametrize(
    "given",
    [
        "bedrock",
        "anthropic",
        "Anthropic",
        "vllm",
        "vllm-offline",
        "vllm-server",
    ],
)
def test_non_canonical_provider_rejected(given: str) -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        ClientConfig.model_validate({"provider": given, "model": "m"})


def test_model_required_except_for_vllm_server() -> None:
    # A vLLM server can report which model it serves; nothing else can.
    assert ClientConfig(provider="vllm_online").model is None
    with pytest.raises(ValueError, match="needs an explicit 'model'"):
        ClientConfig(provider="openai")


def test_unknown_option_names_the_valid_ones() -> None:
    with pytest.raises(ValueError) as excinfo:
        ClientConfig(provider="openai", model="m", options={"top_kk": 5})
    message = str(excinfo.value)
    assert "top_kk" in message
    assert "temperature" in message


def test_build_options_uses_provider_dataclass() -> None:
    options = ClientConfig(
        provider="claude", model="m", options={"max_completion_tokens": 32}
    ).build_options()
    assert type(options).__name__ == "ClaudeRuntimeOptions"
    assert options.max_completion_tokens == 32


def test_build_options_rejects_double_schema() -> None:
    client = ClientConfig(
        provider="openai",
        model="m",
        options={"json_schema": {"type": "object"}},
    )
    with pytest.raises(ValueError, match="not both"):
        client.build_options({"type": "object"})


# --- server pool -------------------------------------------------------------


def test_pool_requires_vllm_provider() -> None:
    with pytest.raises(ValueError, match="needs provider 'vllm_online'"):
        ClientConfig(
            provider="openai", model="m", base_urls=["http://a:8000/v1"]
        )


def test_pool_sources_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at most one of"):
        ClientConfig(
            provider="vllm_online",
            model="m",
            base_urls=["http://a:8000/v1"],
            hosts_file=tmp_path / "hosts.txt",
        )


def test_is_pool_flag() -> None:
    assert not ClientConfig(provider="vllm_online", model="m").is_pool()
    assert ClientConfig(
        provider="vllm_online", model="m", base_urls=["http://a:8000/v1"]
    ).is_pool()


def test_resolve_base_urls_from_hosts_file(tmp_path: Path) -> None:
    hosts = tmp_path / "hosts.txt"
    hosts.write_text(
        "http://a:8000/v1\n\nhttp://b:8000/v1\n", encoding="utf-8"
    )
    client = ClientConfig(
        provider="vllm_online", model="m", hosts_file=Path("hosts.txt")
    )
    assert client.resolve_base_urls(tmp_path) == [
        "http://a:8000/v1",
        "http://b:8000/v1",
    ]


def test_resolve_base_urls_from_url_glob(tmp_path: Path) -> None:
    pool = tmp_path / "pool_1"
    pool.mkdir()
    (pool / "1.url").write_text("http://a:8000/v1\n", encoding="utf-8")
    (pool / "2.url").write_text("http://b:8000/v1\n", encoding="utf-8")
    client = ClientConfig(
        provider="vllm_online", model="m", url_glob="pool_*/*.url"
    )
    assert client.resolve_base_urls(tmp_path) == [
        "http://a:8000/v1",
        "http://b:8000/v1",
    ]


def test_resolve_base_urls_from_absolute_url_glob(tmp_path: Path) -> None:
    # A scheduler writing its pool directory somewhere central has no relative
    # path to offer, and `Path.glob` refuses absolute patterns outright.
    pool = tmp_path / "pool_1"
    pool.mkdir()
    (pool / "1.url").write_text("http://a:8000/v1\n", encoding="utf-8")
    client = ClientConfig(
        provider="vllm_online",
        model="m",
        url_glob=str(tmp_path / "pool_*" / "*.url"),
    )
    assert client.resolve_base_urls(Path.cwd()) == ["http://a:8000/v1"]


def test_resolve_base_urls_reports_empty_absolute_glob(tmp_path: Path) -> None:
    client = ClientConfig(
        provider="vllm_online",
        model="m",
        url_glob=str(tmp_path / "pool_*" / "*.url"),
    )
    with pytest.raises(ValueError, match="matched no files"):
        client.resolve_base_urls(Path.cwd())


def test_resolve_base_urls_reports_empty_glob(tmp_path: Path) -> None:
    client = ClientConfig(
        provider="vllm_online", model="m", url_glob="pool_*/*.url"
    )
    with pytest.raises(ValueError, match="matched no files"):
        client.resolve_base_urls(tmp_path)


# --- step validation ---------------------------------------------------------


def test_inline_and_file_prompt_are_exclusive() -> None:
    with pytest.raises(ValueError, match="not both"):
        StepConfig(name="s", prompt="hi", prompt_file=Path("p.md"))


def test_annotate_step_needs_a_prompt() -> None:
    with pytest.raises(ValueError, match="needs 'prompt' or 'prompt_file'"):
        StepConfig(name="s")


def test_generate_step_needs_prompts() -> None:
    with pytest.raises(ValueError, match="needs 'prompts'"):
        StepConfig(name="s", type="generate")


def test_generate_only_keys_rejected_on_annotate_step() -> None:
    with pytest.raises(ValueError, match="only\n?\\s*apply to a 'generate'"):
        StepConfig(name="s", prompt="x", prompts=["a"])


def test_filter_invalid_needs_a_schema() -> None:
    with pytest.raises(ValueError, match="'filter_invalid' relies on schema"):
        StepConfig(name="s", prompt="x", filter_invalid=True)


def test_task_prefix_defaults_to_step_name() -> None:
    assert (
        StepConfig(name="rate", prompt="x").resolved_task_prefix() == "rate_"
    )
    assert (
        StepConfig(
            name="rate", prompt="x", task_prefix="p_"
        ).resolved_task_prefix()
        == "p_"
    )


def test_resolved_prompts_repeats_single_prompt() -> None:
    step = StepConfig(
        name="gen", type="generate", prompts=["Write a fact."], num_samples=3
    )
    assert step.resolved_prompts(Path(".")) == ["Write a fact."] * 3


def test_resolved_prompts_truncates_a_list() -> None:
    step = StepConfig(
        name="gen", type="generate", prompts=["a", "b", "c"], num_samples=2
    )
    assert step.resolved_prompts(Path(".")) == ["a", "b"]


def test_resolved_prompts_from_file(tmp_path: Path) -> None:
    (tmp_path / "prompts.txt").write_text("one\n\ntwo\n", encoding="utf-8")
    step = StepConfig(
        name="gen", type="generate", prompts=tmp_path / "prompts.txt"
    )
    assert step.resolved_prompts(tmp_path) == ["one", "two"]


def test_missing_referenced_file_reports_resolved_path(tmp_path: Path) -> None:
    step = StepConfig(name="s", prompt_file=Path("nope.md"))
    with pytest.raises(FileNotFoundError, match="resolved to"):
        step.resolved_prompt(tmp_path)


def test_schema_file_must_hold_an_object(tmp_path: Path) -> None:
    (tmp_path / "schema.json").write_text("[]", encoding="utf-8")
    step = StepConfig(
        name="s", prompt="x", output_schema_file=Path("schema.json")
    )
    with pytest.raises(ValueError, match="must contain a JSON object"):
        step.resolved_output_schema(tmp_path)


# --- pipeline validation -----------------------------------------------------


def test_duplicate_step_names_rejected() -> None:
    with pytest.raises(ValueError, match="Step names must be unique"):
        PipelineConfig.model_validate(
            minimal_config(
                steps=[
                    {"name": "a", "prompt": "x"},
                    {"name": "a", "prompt": "y"},
                ]
            )
        )


def test_duplicate_task_prefixes_rejected() -> None:
    with pytest.raises(ValueError, match="task prefixes must be unique"):
        PipelineConfig.model_validate(
            minimal_config(
                steps=[
                    {"name": "a", "prompt": "x", "task_prefix": "p_"},
                    {"name": "b", "prompt": "y", "task_prefix": "p_"},
                ]
            )
        )


def test_generate_step_must_come_first() -> None:
    with pytest.raises(ValueError, match="only be the first step"):
        PipelineConfig.model_validate(
            minimal_config(
                steps=[
                    {"name": "a", "prompt": "x"},
                    {"name": "b", "type": "generate", "prompts": ["p"]},
                ]
            )
        )


def test_annotate_first_step_needs_a_dataset() -> None:
    data = minimal_config()
    del data["dataset"]
    with pytest.raises(ValueError, match="needs a 'dataset' block"):
        PipelineConfig.model_validate(data)


def test_generate_first_step_rejects_a_dataset() -> None:
    with pytest.raises(ValueError, match="would be ignored"):
        PipelineConfig.model_validate(
            minimal_config(
                steps=[{"name": "g", "type": "generate", "prompts": ["p"]}]
            )
        )


def test_generate_first_step_without_dataset_is_valid() -> None:
    data = minimal_config(
        steps=[{"name": "g", "type": "generate", "prompts": ["p"]}]
    )
    del data["dataset"]
    config = PipelineConfig.model_validate(data)
    assert config.steps[0].type == "generate"


def test_dataset_needs_exactly_one_source() -> None:
    with pytest.raises(ValueError, match="exactly one of 'name' or 'path'"):
        PipelineConfig.model_validate(
            minimal_config(dataset={"name": "a", "path": "b"})
        )
    with pytest.raises(ValueError, match="exactly one of 'name' or 'path'"):
        PipelineConfig.model_validate(minimal_config(dataset={}))


def test_step_dir_is_numbered() -> None:
    config = PipelineConfig.model_validate(
        minimal_config(
            steps=[
                {"name": "first", "prompt": "x"},
                {"name": "second", "prompt": "y"},
            ]
        )
    )
    assert config.step_dir(0) == Path("outputs/test/01-first")
    assert config.step_dir(1) == Path("outputs/test/02-second")


# --- client merging ----------------------------------------------------------


def test_step_without_client_uses_pipeline_default() -> None:
    config = PipelineConfig.model_validate(minimal_config())
    assert config.step_client(config.steps[0]) is config.client


def test_per_step_clients_need_no_pipeline_default() -> None:
    # A pipeline whose steps each use a different model has no sensible shared
    # default, so the top-level block must be optional.
    data = minimal_config(
        steps=[
            {
                "name": "a",
                "prompt": "x",
                "client": {"provider": "openai", "model": "gpt-4o-mini"},
            },
            {
                "name": "b",
                "prompt": "y",
                "client": {
                    "provider": "claude",
                    "model": "claude-haiku-4-5",
                },
            },
        ]
    )
    del data["client"]

    config = PipelineConfig.model_validate(data)
    assert config.client is None
    assert [config.step_client(s).provider for s in config.steps] == [
        "openai",
        "claude",
    ]


def test_step_with_no_client_at_either_level_is_reported() -> None:
    data = minimal_config()
    del data["client"]
    with pytest.raises(ValueError, match="Step 'classify' has no client"):
        PipelineConfig.model_validate(data)


def test_missing_client_is_reported_per_step() -> None:
    # Only the second step is uncovered; the error must name that one.
    data = minimal_config(
        steps=[
            {
                "name": "a",
                "prompt": "x",
                "client": {"provider": "openai", "model": "m"},
            },
            {"name": "b", "prompt": "y"},
        ]
    )
    del data["client"]
    with pytest.raises(ValueError, match="Step 'b' has no client"):
        PipelineConfig.model_validate(data)


def test_standalone_step_client_must_be_complete() -> None:
    # Without a default to inherit from, a step block is the whole config and
    # must stand on its own.
    data = minimal_config(
        steps=[{"name": "a", "prompt": "x", "client": {"provider": "openai"}}]
    )
    del data["client"]
    with pytest.raises(ValueError, match="needs an explicit 'model'"):
        PipelineConfig.model_validate(data)


def test_step_client_merges_nested_blocks() -> None:
    config = PipelineConfig.model_validate(
        minimal_config(
            client={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "batch_size": 8,
                "init": {"max_workers": 4},
                "options": {"temperature": 0.5, "max_completion_tokens": 100},
            },
            steps=[
                {
                    "name": "a",
                    "prompt": "x",
                    "client": {"options": {"max_completion_tokens": 999}},
                }
            ],
        )
    )
    merged = config.step_client(config.steps[0])
    # Unmentioned keys are inherited; `options` is merged key-by-key.
    assert merged.provider == "openai"
    assert merged.model == "gpt-4o-mini"
    assert merged.batch_size == 8
    assert merged.init == {"max_workers": 4}
    assert merged.options == {"temperature": 0.5, "max_completion_tokens": 999}


def test_step_client_switching_provider_drops_stale_options() -> None:
    # `reasoning_effort` is OpenAI-only, so inheriting it into Claude would
    # fail validation for no reason the user could act on.
    config = PipelineConfig.model_validate(
        minimal_config(
            client={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "options": {"reasoning_effort": "low"},
            },
            steps=[
                {
                    "name": "a",
                    "prompt": "x",
                    "client": {
                        "provider": "claude",
                        "model": "claude-haiku-4-5",
                    },
                }
            ],
        )
    )
    merged = config.step_client(config.steps[0])
    assert merged.provider == "claude"
    assert merged.options == {}


def test_step_client_switching_provider_keeps_only_its_own_options() -> None:
    # Setting an option of its own must not drag the previous provider's
    # options along with it: `top_k` and `seed` mean nothing to Claude.
    config = PipelineConfig.model_validate(
        minimal_config(
            client={
                "provider": "vllm_offline",
                "model": "Qwen/Qwen3-8B",
                "options": {"temperature": 0.7, "top_k": 20, "seed": 1},
            },
            steps=[
                {
                    "name": "a",
                    "prompt": "x",
                    "client": {
                        "provider": "claude",
                        "model": "claude-haiku-4-5",
                        "options": {"max_completion_tokens": 10},
                    },
                }
            ],
        )
    )
    merged = config.step_client(config.steps[0])
    assert merged.provider == "claude"
    assert merged.options == {"max_completion_tokens": 10}


def test_bad_step_client_fails_at_load_time() -> None:
    # A merged step client is validated up front, so a bad option does not
    # surface only once step 5 of a long pipeline starts.
    with pytest.raises(ValueError) as excinfo:
        PipelineConfig.model_validate(
            minimal_config(
                steps=[
                    {
                        "name": "a",
                        "prompt": "x",
                        "client": {"options": {"nonsense": 1}},
                    }
                ]
            )
        )
    message = str(excinfo.value)
    assert "Step 'a'" in message
    assert "nonsense" in message


def test_partial_step_client_needs_no_provider_or_model() -> None:
    # A fragment is legal on its own; only the merged result must be complete.
    step = StepConfig(name="a", prompt="x", client={"batch_size": 4})
    assert step.client == {"batch_size": 4}


# --- step kinds and pool sizing ----------------------------------------------


@pytest.mark.parametrize(
    ("client", "expected"),
    [
        ({"provider": "vllm_online", "model": "m"}, "vllm_pool"),
        (
            {"provider": "vllm_online", "base_urls": ["http://a:8000/v1"]},
            "vllm_online",
        ),
        ({"provider": "vllm_offline", "model": "m"}, "vllm_offline"),
        ({"provider": "openai", "model": "m"}, "api"),
        ({"provider": "claude", "model": "m"}, "api"),
    ],
)
def test_step_kinds(client: dict[str, Any], expected: str) -> None:
    assert ClientConfig.model_validate(client).kind() == expected


def test_pool_block_defaults_and_round_trip() -> None:
    assert ClientConfig(provider="vllm_online", model="m").pool.servers == 1
    assert (
        ClientConfig(
            provider="vllm_online", model="m"
        ).pool.gpus_per_vllm_server
        == 1
    )

    sized = ClientConfig.model_validate(
        {
            "provider": "vllm_online",
            "model": "m",
            "pool": {"servers": 4, "gpus_per_vllm_server": 2},
        }
    )
    assert (sized.pool.servers, sized.pool.gpus_per_vllm_server) == (4, 2)


def test_pool_block_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        ClientConfig.model_validate(
            {"provider": "vllm_online", "model": "m", "pool": {"servers": 0}}
        )
    with pytest.raises(ValueError, match="gpus_per_serve"):
        ClientConfig.model_validate(
            {
                "provider": "vllm_online",
                "model": "m",
                "pool": {"gpus_per_serve": 2},
            }
        )


def test_pool_block_is_not_part_of_the_cache_key() -> None:
    # Sizing describes the servers a scheduler starts, not the client object,
    # so it must not force an (expensive) client rebuild between steps.
    plain = ClientConfig(provider="vllm_online", model="m")
    sized = ClientConfig.model_validate(
        {"provider": "vllm_online", "model": "m", "pool": {"servers": 8}}
    )
    assert plain.cache_key() == sized.cache_key()


def test_describe_steps_reports_every_step() -> None:
    config = PipelineConfig.model_validate(
        minimal_config(
            client={"provider": "vllm_online", "model": "Qwen/Qwen3-8B"},
            steps=[
                {
                    "name": "write",
                    "prompt": "x",
                    "client": {
                        "pool": {"servers": 4, "gpus_per_vllm_server": 2}
                    },
                },
                {
                    "name": "judge",
                    "prompt": "y",
                    "client": {
                        "provider": "claude",
                        "model": "claude-haiku-4-5",
                    },
                },
            ],
        )
    )
    described = config.describe_steps()

    assert [d["name"] for d in described] == ["write", "judge"]
    assert [d["index"] for d in described] == [1, 2]
    assert [d["kind"] for d in described] == ["vllm_pool", "api"]
    assert described[0]["model"] == "Qwen/Qwen3-8B"
    assert (described[0]["servers"], described[0]["gpus_per_vllm_server"]) == (
        4,
        2,
    )
    # The hosted step needs no accelerator, so it reports the neutral default.
    assert (described[1]["servers"], described[1]["gpus_per_vllm_server"]) == (
        1,
        1,
    )


# --- step-scoped client overrides --------------------------------------------


def test_step_client_overrides_target_one_step(tmp_path: Path) -> None:
    # The case that used to be impossible: a pooled vLLM step next to a hosted
    # one. A top-level hosts_file would be inherited by the hosted step and
    # fail validation, so the override has to be per step.
    path = write_config(
        tmp_path,
        minimal_config(
            steps=[
                {
                    "name": "write",
                    "prompt": "x",
                    "client": {"provider": "vllm_online", "model": "m"},
                },
                {
                    "name": "judge",
                    "prompt": "y",
                    "client": {
                        "provider": "claude",
                        "model": "claude-haiku-4-5",
                    },
                },
            ]
        ),
        ".yaml",
    )
    config = load_pipeline_config(
        path, step_client_overrides={"write": {"hosts_file": "/tmp/hosts.txt"}}
    )

    write, judge = (config.step_client(s) for s in config.steps)
    assert write.is_pool()
    assert write.hosts_file == Path("/tmp/hosts.txt")
    assert not judge.is_pool()
    assert judge.provider == "claude"


def test_step_client_overrides_merge_with_the_step_block(
    tmp_path: Path,
) -> None:
    path = write_config(
        tmp_path,
        minimal_config(
            steps=[
                {
                    "name": "write",
                    "prompt": "x",
                    "client": {
                        "provider": "vllm_online",
                        "model": "m",
                        "queue_size": 4,
                    },
                }
            ]
        ),
        ".yaml",
    )
    config = load_pipeline_config(
        path, step_client_overrides={"write": {"hosts_file": "/tmp/h.txt"}}
    )
    client = config.step_client(config.steps[0])
    assert client.queue_size == 4
    assert client.hosts_file == Path("/tmp/h.txt")


def test_step_client_overrides_reject_unknown_steps(tmp_path: Path) -> None:
    path = write_config(tmp_path, minimal_config(), ".yaml")
    with pytest.raises(ValueError, match="unknown step"):
        load_pipeline_config(
            path, step_client_overrides={"nope": {"hosts_file": "/tmp/h"}}
        )


def test_cache_key_ignores_options_but_tracks_init() -> None:
    base = ClientConfig(provider="openai", model="m")
    same_model = ClientConfig(
        provider="openai", model="m", options={"temperature": 0.9}
    )
    other_init = ClientConfig(
        provider="openai", model="m", init={"max_workers": 2}
    )
    other_model = ClientConfig(provider="openai", model="n")

    # Options are per-request, so they must not force a client rebuild.
    assert base.cache_key() == same_model.cache_key()
    assert base.cache_key() != other_init.cache_key()
    assert base.cache_key() != other_model.cache_key()
