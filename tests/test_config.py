from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import llm_annotator.config as config_mod
from llm_annotator.config import (
    ClientConfig,
    DatasetConfig,
    EngineConfig,
    PipelineConfig,
    PoolConfig,
    StepConfig,
    load_config_file,
    load_pipeline_config,
    wait_for_servers,
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
    assert config.output_dir == tmp_path / "elsewhere"
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


def test_json_system_prompt_file_is_read_verbatim(tmp_path: Path) -> None:
    """A '.json' prompt file reaches the model as the text it holds."""
    payload = {"instruction": "Use only the code.", "codes": ["alpha"]}
    catalog_path = tmp_path / "taxonomy.json"
    catalog_path.write_text(json.dumps(payload), encoding="utf-8")

    step = StepConfig(
        name="classify",
        prompt="Classify: {text}",
        system_prompt_file=catalog_path,
    )

    assert step.resolved_system_prompt(tmp_path) == json.dumps(payload)


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


def test_unknown_init_key_names_the_accepted_ones() -> None:
    with pytest.raises(ValueError) as excinfo:
        ClientConfig(provider="openai", model="m", init={"on_eror": "warn"})
    message = str(excinfo.value)

    assert "on_eror" in message
    assert "openai" in message
    assert "'on_error'" in message.replace('"', "'")


@pytest.mark.parametrize(
    "provider, init",
    [
        ("openai", {"api_key": "k", "base_url": "u", "max_workers": 2}),
        ("claude", {"api_key": "k", "on_error": "raise"}),
        ("vllm_online", {"base_url": "http://a:8000/v1"}),
        ("vllm_offline", {"language_model_only": False, "batch_size": 4}),
    ],
)
def test_init_accepts_constructor_arguments(
    provider: str, init: dict[str, Any]
) -> None:
    client = ClientConfig.model_validate(
        {"provider": provider, "model": "m", "init": init}
    )

    assert client.init == init


def test_init_rejects_the_model_key() -> None:
    with pytest.raises(ValueError, match="'init' sets 'model'"):
        ClientConfig(provider="openai", model="m", init={"model": "other"})


def test_init_rejects_a_base_url_for_a_pool() -> None:
    with pytest.raises(ValueError, match="each server of a pool gets its own"):
        ClientConfig(
            provider="vllm_online",
            model="m",
            base_urls=["http://a:8000/v1"],
            init={"base_url": "http://b:8000/v1"},
        )


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


def test_wait_for_servers_returns_the_minimum_ready_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready_urls = {"http://a:8000/v1", "http://b:8000/v1"}
    monkeypatch.setattr(
        "llm_annotator.config.server_is_healthy",
        lambda url, timeout: url in ready_urls,
    )

    assert wait_for_servers(
        ["http://a:8000/v1", "http://b:8000/v1", "http://c:8000/v1"],
        timeout=1,
        min_servers=2,
    ) == ["http://a:8000/v1", "http://b:8000/v1"]


def test_wait_for_servers_validates_against_unique_urls() -> None:
    with pytest.raises(ValueError, match="between 1 and 2"):
        wait_for_servers(
            ["http://a:8000/v1", "http://a:8000/v1", "http://b:8000/v1"],
            timeout=1,
            min_servers=3,
        )


def test_pool_min_servers_cannot_exceed_servers() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        ClientConfig(
            provider="vllm_online",
            model="m",
            pool=PoolConfig(servers=2, min_servers=3),
        )


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


def test_build_annotator_counts_unique_pool_members(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = ClientConfig(
        provider="vllm_online",
        model="m",
        base_urls=[
            "http://a:8000/v1",
            "http://a:8000/v1",
            "http://b:8000/v1",
        ],
    )
    seen: dict[str, Any] = {}

    def fake_build_client(self: ClientConfig, root: Path) -> list[str]:
        _ = self
        _ = root
        return ["a", "b"]

    class FakeAnnotator:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)
            self.max_workers = kwargs["max_workers"]

    monkeypatch.setattr(ClientConfig, "build_client", fake_build_client)
    monkeypatch.setattr(config_mod, "VLLMQueueAnnotator", FakeAnnotator)
    monkeypatch.setattr(
        ClientConfig, "_watch_pool", lambda self, root, annotator: None
    )

    annotator = client.build_annotator(tmp_path)

    assert isinstance(annotator, FakeAnnotator)
    assert seen["max_workers"] == 8


def test_pool_watcher_stops_after_destroy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = ClientConfig(
        provider="vllm_online",
        model="m",
        base_urls=["http://w0:8000/v1", "http://w1:8000/v1"],
    )
    entered = threading.Event()
    ready_check_finished = threading.Event()
    released = threading.Event()

    class FakeAnnotator:
        def __init__(self) -> None:
            self.max_workers = 2
            self.clients = [
                type("ClientRef", (), {"base_url": "http://w0:8000/v1"})()
            ]
            self.added: list[Any] = []
            self._closed = threading.Event()

        @property
        def is_shutting_down(self) -> bool:
            return self._closed.is_set()

        def wait_for_shutdown(self, timeout: float) -> bool:
            return self._closed.wait(timeout)

        def client_count(self) -> int:
            return len(self.clients)

        def client_base_urls(self) -> set[str]:
            return {
                str(getattr(client, "base_url")) for client in self.clients
            }

        def add_client(self, client: Any) -> None:
            self.added.append(client)

        def add_client_for_base_url(
            self, base_url: str, client_factory: Any
        ) -> None:
            self.added.append(client_factory(base_url))

        def destroy(self) -> None:
            self._closed.set()

    def fake_ready(url: str, timeout: float) -> bool:
        _ = url
        _ = timeout
        entered.set()
        try:
            released.wait(5)
            return True
        finally:
            ready_check_finished.set()

    monkeypatch.setattr(config_mod, "server_is_healthy", fake_ready)
    monkeypatch.setattr(
        ClientConfig,
        "resolve_base_urls",
        lambda self, root: ["http://w1:8000/v1"],
    )
    annotator = FakeAnnotator()

    client._watch_pool(tmp_path, annotator)  # type: ignore[arg-type]
    assert entered.wait(5)
    annotator.destroy()
    released.set()
    assert ready_check_finished.wait(5)

    assert annotator.added == []


def test_build_client_caps_the_wait_at_the_servers_it_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A pool directory that is still filling names fewer servers than
    # `min_servers`; waiting for more than it names could never succeed.
    (tmp_path / "pool_1").mkdir()
    (tmp_path / "pool_1" / "0.url").write_text(
        "http://w0:8000/v1\n", encoding="utf-8"
    )
    client = ClientConfig(
        provider="vllm_online",
        model="m",
        url_glob="pool_*/*.url",
        wait_for_servers=1,
        pool=PoolConfig(servers=4, min_servers=3),
    )
    seen: dict[str, Any] = {}

    class FakePooledClient:
        def __init__(self, *, base_url: str, **kwargs: Any) -> None:
            _ = kwargs
            self.base_url = base_url

    def fake_wait(
        base_urls: list[str], timeout: float, min_servers: int = 1
    ) -> list[str]:
        seen["min_servers"] = min_servers
        return base_urls

    monkeypatch.setattr(config_mod, "wait_for_servers", fake_wait)
    monkeypatch.setattr(config_mod, "VLLMOnlineClient", FakePooledClient)

    clients = client.build_client(tmp_path)

    assert isinstance(clients, list)
    assert len(clients) == 1
    assert seen["min_servers"] == 1


def test_pool_watcher_keeps_polling_dynamic_discovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = ClientConfig(
        provider="vllm_online", model="m", url_glob="pool_*/*.url"
    )
    added = threading.Event()
    resolve_calls = 0

    class FakeAnnotator:
        def __init__(self) -> None:
            self.clients = [
                type("ClientRef", (), {"base_url": "http://w0:8000/v1"})()
            ]
            self.added: list[Any] = []
            self._closed = threading.Event()

        @property
        def is_shutting_down(self) -> bool:
            return self._closed.is_set()

        def wait_for_shutdown(self, timeout: float) -> bool:
            _ = timeout
            return self._closed.wait(0.01)

        def client_count(self) -> int:
            return len(self.clients)

        def client_base_urls(self) -> set[str]:
            return {str(client.base_url) for client in self.clients}

        def add_client_for_base_url(
            self, base_url: str, client_factory: Any
        ) -> None:
            discovered_client = client_factory(base_url)
            self.clients.append(discovered_client)
            self.added.append(discovered_client)
            added.set()
            self._closed.set()

    class FakeDiscoveredClient:
        def __init__(self, *, base_url: str, **kwargs: Any) -> None:
            _ = kwargs
            self.base_url = base_url

    def fake_resolve(self: ClientConfig, root: Path) -> list[str]:
        _ = self
        _ = root
        nonlocal resolve_calls
        resolve_calls += 1
        if resolve_calls == 1:
            return ["http://w0:8000/v1"]
        return ["http://w0:8000/v1", "http://w1:8000/v1"]

    monkeypatch.setattr(
        config_mod, "server_is_healthy", lambda url, timeout: True
    )
    monkeypatch.setattr(ClientConfig, "resolve_base_urls", fake_resolve)
    monkeypatch.setattr(config_mod, "VLLMOnlineClient", FakeDiscoveredClient)
    annotator = FakeAnnotator()

    client._watch_pool(tmp_path, annotator)  # type: ignore[arg-type]

    assert added.wait(5)
    assert resolve_calls >= 2
    assert [added_client.base_url for added_client in annotator.added] == [
        "http://w1:8000/v1"
    ]


def test_pool_watcher_readmits_an_evicted_static_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A pool configured with static base_urls keeps probing every URL that
    # the running pool does not currently hold, so a server that the
    # annotator evicted is admitted again once it answers '/health'.
    client = ClientConfig(
        provider="vllm_online",
        model="m",
        base_urls=["http://w0:8000/v1", "http://w1:8000/v1"],
    )
    added = threading.Event()

    class FakeAnnotator:
        def __init__(self) -> None:
            # w0 was evicted earlier; only w1 remains in the pool.
            self.clients = [
                type("ClientRef", (), {"base_url": "http://w1:8000/v1"})()
            ]
            self.added: list[Any] = []
            self._closed = threading.Event()

        @property
        def is_shutting_down(self) -> bool:
            return self._closed.is_set()

        def wait_for_shutdown(self, timeout: float) -> bool:
            _ = timeout
            return self._closed.wait(0.01)

        def client_count(self) -> int:
            return len(self.clients)

        def client_base_urls(self) -> set[str]:
            return {str(client.base_url) for client in self.clients}

        def add_client_for_base_url(
            self, base_url: str, client_factory: Any
        ) -> None:
            discovered_client = client_factory(base_url)
            self.clients.append(discovered_client)
            self.added.append(discovered_client)
            added.set()
            self._closed.set()

    class FakeReadmittedClient:
        def __init__(self, *, base_url: str, **kwargs: Any) -> None:
            _ = kwargs
            self.base_url = base_url

    monkeypatch.setattr(
        config_mod, "server_is_healthy", lambda url, timeout: True
    )
    monkeypatch.setattr(config_mod, "VLLMOnlineClient", FakeReadmittedClient)
    annotator = FakeAnnotator()

    client._watch_pool(tmp_path, annotator)  # type: ignore[arg-type]

    assert added.wait(5)
    assert [added_client.base_url for added_client in annotator.added] == [
        "http://w0:8000/v1"
    ]


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


def test_max_consecutive_failed_batches_defaults_and_rejects_negative() -> (
    None
):
    assert (
        StepConfig(name="s", prompt="x").max_consecutive_failed_batches == 10
    )
    with pytest.raises(ValidationError):
        StepConfig(name="s", prompt="x", max_consecutive_failed_batches=-1)


def test_max_samples_per_output_file_defaults_and_accepts_an_int() -> None:
    assert (
        StepConfig(name="s", prompt="x").max_samples_per_output_file == "auto"
    )
    assert (
        StepConfig(
            name="s", prompt="x", max_samples_per_output_file=500
        ).max_samples_per_output_file
        == 500
    )


@pytest.mark.parametrize("value", [-1, "nope"])
def test_max_samples_per_output_file_rejects_bad_values(
    value: Any,
) -> None:
    with pytest.raises(ValidationError):
        StepConfig(name="s", prompt="x", max_samples_per_output_file=value)


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


def test_resolved_prompts_from_a_json_file(tmp_path: Path) -> None:
    """A '.json' prompts file is a plain list of strings."""
    prompts = ["Write a geography question.", "Write a math question."]
    (tmp_path / "prompts.json").write_text(
        json.dumps(prompts), encoding="utf-8"
    )
    step = StepConfig(
        name="gen", type="generate", prompts=Path("prompts.json")
    )

    assert step.resolved_prompts(tmp_path) == prompts


@pytest.mark.parametrize(
    "payload, message",
    [
        ('{"a": "b"}', "holds a dict"),
        ("[1, 2]", "position(s) [0, 1]"),
        ("[]", "holds an empty list"),
        ("[not json", "is not valid JSON"),
    ],
)
def test_json_prompts_file_rejects_other_shapes(
    tmp_path: Path, payload: str, message: str
) -> None:
    (tmp_path / "prompts.json").write_text(payload, encoding="utf-8")
    step = StepConfig(
        name="gen", type="generate", prompts=Path("prompts.json")
    )

    with pytest.raises(ValueError) as excinfo:
        step.resolved_prompts(tmp_path)

    assert message in str(excinfo.value)
    assert str(tmp_path / "prompts.json") in str(excinfo.value)


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


@pytest.mark.parametrize(
    "data_files",
    [
        "data.jsonl",
        ["a.jsonl", "b.jsonl"],
        {"train": "train.jsonl", "test": ["a.jsonl", "b.jsonl"]},
    ],
)
def test_dataset_config_accepts_data_files_shapes(
    data_files: str | list[str] | dict[str, str | list[str]],
) -> None:
    config = DatasetConfig.model_validate(
        {"name": "json", "data_files": data_files}
    )
    assert config.data_files == data_files


def test_local_data_paths_resolve_against_the_config_dir(
    tmp_path: Path,
) -> None:
    config = DatasetConfig.model_validate(
        {
            "name": "json",
            "data_dir": "data",
            "data_files": {
                "train": ["train/*.jsonl", "extra.jsonl"],
                "test": "test.jsonl",
            },
        }
    )

    assert config.resolved_data_dir(tmp_path) == str(tmp_path / "data")
    assert config.resolved_data_files(tmp_path) == {
        "train": [
            str(tmp_path / "train/*.jsonl"),
            str(tmp_path / "extra.jsonl"),
        ],
        "test": str(tmp_path / "test.jsonl"),
    }


def test_a_local_directory_name_is_a_local_source(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()
    config = DatasetConfig.model_validate(
        {"name": "corpus", "data_files": "part-*.parquet"}
    )

    assert config.is_local_source(tmp_path)
    assert config.resolved_data_files(tmp_path) == str(
        tmp_path / "part-*.parquet"
    )


@pytest.mark.parametrize(
    "name, data_files",
    [
        ("stanfordnlp/imdb", "plain_text/train-*.parquet"),
        ("json", "https://example.com/data.jsonl"),
        ("json", "hf://datasets/user/repo/a.jsonl"),
    ],
)
def test_data_files_that_are_not_local_paths_are_left_alone(
    tmp_path: Path, name: str, data_files: str
) -> None:
    config = DatasetConfig.model_validate(
        {"name": name, "data_files": data_files}
    )

    assert config.resolved_data_files(tmp_path) == data_files


def test_absolute_data_paths_are_left_alone(tmp_path: Path) -> None:
    absolute = str(tmp_path / "somewhere" / "data.jsonl")
    config = DatasetConfig.model_validate(
        {"name": "json", "data_dir": "/data", "data_files": absolute}
    )

    assert config.resolved_data_dir(tmp_path) == "/data"
    assert config.resolved_data_files(tmp_path) == absolute


def test_dataset_config_data_dir_and_data_files_reject_path() -> None:
    with pytest.raises(ValueError, match="only apply to 'name'"):
        DatasetConfig.model_validate({"path": "b", "data_dir": "d"})
    with pytest.raises(ValueError, match="only apply to 'name'"):
        DatasetConfig.model_validate({"path": "b", "data_files": "data.jsonl"})


def test_step_dir_is_numbered(tmp_path: Path) -> None:
    config = PipelineConfig.model_validate(
        minimal_config(
            config_dir=tmp_path,
            steps=[
                {"name": "first", "prompt": "x"},
                {"name": "second", "prompt": "y"},
            ],
        )
    )
    assert config.step_dir(0) == tmp_path / "outputs/test/01-first"
    assert config.step_dir(1) == tmp_path / "outputs/test/02-second"


def test_rename_key_matching_idx_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="renames or drops 'idx'"):
        PipelineConfig.model_validate(
            minimal_config(
                steps=[
                    {
                        "name": "a",
                        "prompt": "x",
                        "rename": {"idx": "row_id"},
                    }
                ]
            )
        )


def test_rename_value_matching_idx_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="renames or drops 'idx'"):
        PipelineConfig.model_validate(
            minimal_config(
                steps=[
                    {
                        "name": "a",
                        "prompt": "x",
                        "rename": {"label": "idx"},
                    }
                ]
            )
        )


def test_drop_columns_matching_idx_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="renames or drops 'idx'"):
        PipelineConfig.model_validate(
            minimal_config(
                steps=[
                    {
                        "name": "a",
                        "prompt": "x",
                        "drop_columns": ["idx"],
                    }
                ]
            )
        )


def test_idx_column_guard_follows_a_custom_name() -> None:
    # A custom 'idx_column' is what gets guarded, not the literal "idx".
    PipelineConfig.model_validate(
        minimal_config(
            idx_column="row_id",
            steps=[{"name": "a", "prompt": "x", "drop_columns": ["idx"]}],
        )
    )
    with pytest.raises(ValueError, match="renames or drops 'row_id'"):
        PipelineConfig.model_validate(
            minimal_config(
                idx_column="row_id",
                steps=[
                    {"name": "a", "prompt": "x", "drop_columns": ["row_id"]}
                ],
            )
        )


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


def test_step_client_switching_provider_drops_inherited_init() -> None:
    # The reproduction of issue #22: an inherited 'init' sent the OpenAI key
    # and base URL to Anthropic.
    config = PipelineConfig.model_validate(
        minimal_config(
            client={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "init": {
                    "api_key": "sk-openai",
                    "base_url": "https://api.openai.com/v1",
                },
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

    assert config.step_client(config.steps[0]).init == {}


def test_step_client_switching_provider_keeps_its_own_init() -> None:
    config = PipelineConfig.model_validate(
        minimal_config(
            client={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "init": {"api_key": "sk-openai"},
            },
            steps=[
                {
                    "name": "a",
                    "prompt": "x",
                    "client": {
                        "provider": "claude",
                        "model": "claude-haiku-4-5",
                        "init": {"api_key": "sk-anthropic"},
                    },
                }
            ],
        )
    )

    assert config.step_client(config.steps[0]).init == {
        "api_key": "sk-anthropic"
    }


def test_step_client_switching_provider_drops_the_pool_blocks() -> None:
    # 'base_urls' is rejected outright on a hosted provider, so inheriting it
    # made the whole config fail to load.
    config = PipelineConfig.model_validate(
        minimal_config(
            client={
                "provider": "vllm_online",
                "model": "Qwen/Qwen3-8B",
                "base_urls": ["http://node01:8000/v1"],
                "queue_size": 16,
                "wait_for_servers": 300,
                "pool": {"servers": 4, "min_servers": 2},
                "engine": {"tensor_parallel_size": 2},
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

    assert merged.base_urls == []
    assert merged.queue_size is None
    assert merged.pool == PoolConfig()
    assert merged.engine == EngineConfig()
    assert merged.wait_for_servers == 60.0


def test_step_client_keeps_the_pool_blocks_within_one_provider() -> None:
    config = PipelineConfig.model_validate(
        minimal_config(
            client={
                "provider": "vllm_online",
                "model": "Qwen/Qwen3-8B",
                "base_urls": ["http://node01:8000/v1"],
                "pool": {"servers": 4, "min_servers": 2},
            },
            steps=[{"name": "a", "prompt": "x", "client": {"batch_size": 8}}],
        )
    )
    merged = config.step_client(config.steps[0])

    assert merged.base_urls == ["http://node01:8000/v1"]
    assert merged.pool == PoolConfig(servers=4, min_servers=2)


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
        ).engine.tensor_parallel_size
        == 1
    )

    sized = ClientConfig.model_validate(
        {
            "provider": "vllm_online",
            "model": "m",
            "engine": {"tensor_parallel_size": 2},
            "pool": {"servers": 4},
        }
    )
    assert (sized.pool.servers, sized.engine.tensor_parallel_size) == (4, 2)


def test_pool_gpus_per_vllm_server_points_at_engine() -> None:
    """The old spelling names its replacement instead of "unknown key"."""
    with pytest.raises(ValueError, match="engine.tensor_parallel_size"):
        ClientConfig.model_validate(
            {
                "provider": "vllm_online",
                "model": "m",
                "pool": {"servers": 4, "gpus_per_vllm_server": 2},
            }
        )


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
                        "engine": {"tensor_parallel_size": 2},
                        "pool": {"servers": 4, "min_servers": 2},
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
    # A pool starts once this many of its servers are ready.
    assert described[0]["min_servers"] == 2
    # The hosted step needs no accelerator, so it reports the neutral default.
    assert (described[1]["servers"], described[1]["gpus_per_vllm_server"]) == (
        1,
        1,
    )
    # A step that never asked for a pool still reports the default minimum.
    assert described[1]["min_servers"] == 1

    # The concurrency a submitter sizes the servers against: 4 servers x 4
    # requests each, 256 samples per request by default.
    assert described[0]["max_concurrent_batches_per_client"] == 4
    assert described[0]["queue_size"] == 64
    assert described[0]["max_requests_per_server"] == 1024
    assert described[0]["max_requests_in_flight"] == 4096
    # The hosted step has no pool, so it reports only its batch size.
    assert described[1]["batch_size"] == 256
    assert described[1]["queue_size"] is None
    assert described[1]["max_requests_in_flight"] is None


# --- pool concurrency --------------------------------------------------------


def pool_client(**overrides: Any) -> dict[str, Any]:
    """Build a vLLM pool client block with optional overrides."""
    data: dict[str, Any] = {"provider": "vllm_online", "model": "m"}
    data.update(overrides)
    return data


def test_queue_size_below_the_pool_minimum_is_rejected() -> None:
    # 4 servers x 4 concurrent requests each = 16 requests in flight, so a
    # queue of 8 could never fill the pool.
    with pytest.raises(ValidationError, match="minimum of 16"):
        ClientConfig.model_validate(
            pool_client(queue_size=8, pool={"servers": 4})
        )


def test_queue_size_minimum_counts_explicit_base_urls() -> None:
    # No `pool` block here: the servers already exist, so they are counted
    # from `base_urls` instead.
    urls = [f"http://node{i:02d}:8000/v1" for i in range(1, 4)]
    with pytest.raises(ValidationError, match="minimum of 12"):
        ClientConfig.model_validate(pool_client(queue_size=6, base_urls=urls))

    at_minimum = ClientConfig.model_validate(
        pool_client(queue_size=12, base_urls=urls)
    )
    assert at_minimum.queue_size == 12


def test_queue_size_minimum_follows_the_per_client_concurrency() -> None:
    # One request per server lowers the floor to the server count.
    client = ClientConfig.model_validate(
        pool_client(
            queue_size=4,
            pool={"servers": 4},
            max_concurrent_batches_per_client=1,
        )
    )
    assert client.queue_size == 4


def test_queue_size_error_names_the_minimum_and_the_default() -> None:
    with pytest.raises(ValidationError) as excinfo:
        ClientConfig.model_validate(
            pool_client(queue_size=3, pool={"servers": 2})
        )

    message = str(excinfo.value)
    assert "at least 8" in message
    # The value a user gets by removing the key altogether.
    assert "(32)" in message


def test_queue_size_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ClientConfig.model_validate(pool_client(queue_size=0))
    with pytest.raises(ValidationError):
        ClientConfig.model_validate(
            pool_client(max_concurrent_batches_per_client=0)
        )


@pytest.mark.parametrize("provider", ["openai", "claude", "vllm_offline"])
@pytest.mark.parametrize(
    "key", ["queue_size", "max_concurrent_batches_per_client"]
)
def test_pool_keys_are_rejected_for_providers_without_a_pool(
    provider: str, key: str
) -> None:
    with pytest.raises(ValidationError, match="size a pool of vLLM servers"):
        ClientConfig.model_validate(
            {"provider": provider, "model": "m", key: 2}
        )


def test_a_step_does_not_inherit_pool_keys_from_another_provider() -> None:
    # The top-level block sizes a pool; the hosted step cannot use those keys
    # and must not be rejected for a value it never wrote.
    config = PipelineConfig.model_validate(
        minimal_config(
            client=pool_client(queue_size=8, engine={"max_model_len": 4096}),
            steps=[
                {
                    "name": "judge",
                    "prompt": "y",
                    "client": {"provider": "claude", "model": "c"},
                }
            ],
        )
    )
    client = config.step_client(config.steps[0])

    assert client.queue_size is None
    assert client.max_concurrent_batches_per_client == 4
    assert client.engine == EngineConfig()


def test_a_step_keeps_pool_keys_it_writes_itself() -> None:
    with pytest.raises(ValidationError, match="size a pool of vLLM servers"):
        PipelineConfig.model_validate(
            minimal_config(
                client=pool_client(),
                steps=[
                    {
                        "name": "judge",
                        "prompt": "y",
                        "client": {
                            "provider": "claude",
                            "model": "c",
                            "queue_size": 8,
                        },
                    }
                ],
            )
        )


def test_a_vllm_step_still_inherits_the_engine_from_the_other_vllm_provider() -> (
    None
):
    config = PipelineConfig.model_validate(
        minimal_config(
            client={
                "provider": "vllm_offline",
                "model": "m",
                "engine": {"tensor_parallel_size": 2},
            },
            steps=[
                {
                    "name": "write",
                    "prompt": "x",
                    "client": {"provider": "vllm_online"},
                }
            ],
        )
    )

    assert config.step_client(config.steps[0]).engine.tensor_parallel_size == 2


def test_concurrency_reports_the_effective_numbers() -> None:
    client = ClientConfig.model_validate(
        pool_client(batch_size=64, pool={"servers": 4})
    )

    assert client.concurrency() == {
        "batch_size": 64,
        "max_concurrent_batches_per_client": 4,
        "queue_size": 64,
        "max_requests_per_server": 256,
        "max_requests_in_flight": 1024,
    }


def test_concurrency_reports_the_requested_queue_size() -> None:
    client = ClientConfig.model_validate(
        pool_client(batch_size=8, queue_size=100, pool={"servers": 2})
    )

    assert client.concurrency()["queue_size"] == 100


def test_concurrency_is_empty_without_a_pool() -> None:
    client = ClientConfig.model_validate(
        {"provider": "claude", "model": "m", "batch_size": 32}
    )

    assert client.concurrency() == {
        "batch_size": 32,
        "max_concurrent_batches_per_client": None,
        "queue_size": None,
        "max_requests_per_server": None,
        "max_requests_in_flight": None,
    }


# --- dotted overrides --------------------------------------------------------


def test_dotted_override_reaches_a_nested_block(tmp_path: Path) -> None:
    path = write_config(tmp_path, minimal_config(), ".yaml")
    config = load_pipeline_config(
        path,
        overrides={
            "dataset.max_num_samples": 2000,
            "dataset.shuffle_seed": 42,
        },
    )
    assert config.dataset is not None
    assert config.dataset.max_num_samples == 2000
    assert config.dataset.shuffle_seed == 42
    assert config.dataset.name == "stanfordnlp/imdb"


def test_dotted_override_creates_a_missing_block(tmp_path: Path) -> None:
    # 'options' is absent from the file, so the path has to be built on the
    # way down rather than assumed to exist.
    path = write_config(tmp_path, minimal_config(), ".yaml")
    config = load_pipeline_config(
        path, overrides={"client.options.temperature": 0.2}
    )
    assert config.client is not None
    assert config.client.options == {"temperature": 0.2}


def test_dotted_override_indexes_a_list(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        minimal_config(
            steps=[
                {"name": "one", "prompt": "a {text}"},
                {"name": "two", "prompt": "b {text}"},
            ]
        ),
        ".yaml",
    )
    config = load_pipeline_config(
        path, overrides={"steps.1.client.batch_size": 8}
    )
    assert config.step_client(config.steps[1]).batch_size == 8
    assert config.step_client(config.steps[0]).batch_size != 8


def test_dotted_override_rejects_an_out_of_range_position(
    tmp_path: Path,
) -> None:
    path = write_config(tmp_path, minimal_config(), ".yaml")
    with pytest.raises(ValueError, match="position 3 is out of range"):
        load_pipeline_config(path, overrides={"steps.3.name": "nope"})


def test_dotted_override_rejects_a_non_integer_list_segment(
    tmp_path: Path,
) -> None:
    path = write_config(tmp_path, minimal_config(), ".yaml")
    with pytest.raises(ValueError, match="is not a list index"):
        load_pipeline_config(path, overrides={"steps.classify.name": "nope"})


def test_dotted_override_rejects_descending_into_a_scalar(
    tmp_path: Path,
) -> None:
    path = write_config(tmp_path, minimal_config(), ".yaml")
    with pytest.raises(ValueError, match="cannot be set inside a str"):
        load_pipeline_config(path, overrides={"output_dir.deeper": 1})


def test_dotted_override_is_validated_like_the_file(tmp_path: Path) -> None:
    path = write_config(tmp_path, minimal_config(), ".yaml")
    with pytest.raises(ValueError, match="max_num_smaples"):
        load_pipeline_config(path, overrides={"dataset.max_num_smaples": 10})


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


def test_engine_serve_args_render() -> None:
    """Every value shape a `vllm serve` flag can take."""
    engine = EngineConfig(
        tensor_parallel_size=2,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        enable_prefix_caching=False,
        speculative_config={"model": "draft", "num_speculative_tokens": 4},
        extra={"reasoning_parser": "qwen3"},
    )
    args = engine.as_serve_args()

    assert args[:2] == ["--tensor-parallel-size", "2"]
    assert "--max-model-len" in args and "8192" in args
    # A true boolean is a bare flag, a false one its --no- form.
    assert "--enforce-eager" in args
    assert "--no-enable-prefix-caching" in args
    # A mapping travels as one JSON argument, so it survives a value with a
    # space in it -- which is what shell word-splitting used to destroy.
    spec = args[args.index("--speculative-config") + 1]
    assert json.loads(spec) == {"model": "draft", "num_speculative_tokens": 4}
    # `extra` keys are kebab-cased like the named ones.
    assert args[-2:] == ["--reasoning-parser", "qwen3"]


def test_engine_drops_unset_fields() -> None:
    """Unset fields defer to vLLM's own defaults instead of passing None."""
    engine = EngineConfig()
    assert engine.as_llm_kwargs() == {"tensor_parallel_size": 1}
    assert engine.as_serve_args() == ["--tensor-parallel-size", "1"]


def test_engine_kwargs_match_offline_client_signature() -> None:
    """Every named engine field is a real VLLMOfflineClient parameter."""
    import inspect

    from llm_annotator.clients.vllm_offline_client import VLLMOfflineClient

    accepted = set(inspect.signature(VLLMOfflineClient.__init__).parameters)
    assert set(EngineConfig.model_fields) - {"extra"} <= accepted


def test_engine_rejected_for_hosted_providers() -> None:
    with pytest.raises(ValueError, match="no use for it"):
        ClientConfig.model_validate(
            {
                "provider": "claude",
                "model": "m",
                "engine": {"max_model_len": 4096},
            }
        )


def test_engine_settings_may_not_hide_in_init() -> None:
    """`init` and `engine` must not both be able to set the same thing."""
    with pytest.raises(ValueError, match="Move .* from 'init' to 'engine'"):
        ClientConfig.model_validate(
            {
                "provider": "vllm_offline",
                "model": "m",
                "init": {"max_model_len": 4096},
            }
        )


def test_cache_key_tracks_engine_but_not_gen_kwargs() -> None:
    base = ClientConfig(provider="vllm_offline", model="m")
    other_engine = ClientConfig(
        provider="vllm_offline",
        model="m",
        engine=EngineConfig(max_model_len=4096),
    )
    other_gen = ClientConfig(
        provider="vllm_offline", model="m", gen_kwargs={"min_p": 0.1}
    )

    # A different engine needs a different engine; gen_kwargs are per request.
    assert base.cache_key() != other_engine.cache_key()
    assert base.cache_key() == other_gen.cache_key()
