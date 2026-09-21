from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, cast

import pytest

import llm_annotator.pool as pool_mod
from llm_annotator.clients.vllm_online_client import VLLMOnlineClient
from llm_annotator.config import ClientConfig, PoolConfig
from llm_annotator.pool import (
    _watch_pool,
    build_annotator,
    build_client,
    wait_for_servers,
)


def test_wait_for_servers_returns_the_minimum_ready_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready_urls = {"http://a:8000/v1", "http://b:8000/v1"}
    monkeypatch.setattr(
        "llm_annotator.pool.server_is_healthy",
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

    def fake_build_client(
        client_config: ClientConfig, root: Path
    ) -> list[str]:
        _ = client_config
        _ = root
        return ["a", "b"]

    class FakeAnnotator:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)
            self.max_workers = kwargs["max_workers"]

    monkeypatch.setattr(pool_mod, "build_client", fake_build_client)
    monkeypatch.setattr(pool_mod, "VLLMQueueAnnotator", FakeAnnotator)
    monkeypatch.setattr(
        pool_mod,
        "_watch_pool",
        lambda client_config, root, annotator: None,
    )

    annotator = build_annotator(client, tmp_path)

    assert isinstance(annotator, FakeAnnotator)
    assert seen["max_workers"] == 8


def test_build_client_forwards_init_to_every_pool_member(
    fake_openai_module: dict[str, Any], tmp_path: Path
) -> None:
    """A pool member is built from `init`, plus the base URL of its server."""
    _ = fake_openai_module
    client_config = ClientConfig(
        provider="vllm_online",
        model="m",
        base_urls=["http://a:8000/v1", "http://b:8000/v1"],
        wait_for_servers=0,
        init={"timeout": 60.0, "max_retries": 0, "max_workers": 8},
    )

    clients = build_client(client_config, tmp_path)

    assert isinstance(clients, list)
    members = [cast(VLLMOnlineClient, client) for client in clients]
    assert [member.base_url for member in members] == [
        "http://a:8000/v1",
        "http://b:8000/v1",
    ]
    for member in members:
        assert member.timeout == 60.0
        assert member.max_retries == 0
        assert member.max_workers == 8


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

    monkeypatch.setattr(pool_mod, "server_is_healthy", fake_ready)
    monkeypatch.setattr(
        ClientConfig,
        "resolve_base_urls",
        lambda self, root: ["http://w1:8000/v1"],
    )
    annotator = FakeAnnotator()

    _watch_pool(client, tmp_path, annotator)  # type: ignore[arg-type]
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

    monkeypatch.setattr(pool_mod, "wait_for_servers", fake_wait)
    monkeypatch.setattr(pool_mod, "VLLMOnlineClient", FakePooledClient)

    clients = build_client(client, tmp_path)

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
        pool_mod, "server_is_healthy", lambda url, timeout: True
    )
    monkeypatch.setattr(ClientConfig, "resolve_base_urls", fake_resolve)
    monkeypatch.setattr(pool_mod, "VLLMOnlineClient", FakeDiscoveredClient)
    annotator = FakeAnnotator()

    _watch_pool(client, tmp_path, annotator)  # type: ignore[arg-type]

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
        pool_mod, "server_is_healthy", lambda url, timeout: True
    )
    monkeypatch.setattr(pool_mod, "VLLMOnlineClient", FakeReadmittedClient)
    annotator = FakeAnnotator()

    _watch_pool(client, tmp_path, annotator)  # type: ignore[arg-type]

    assert added.wait(5)
    assert [added_client.base_url for added_client in annotator.added] == [
        "http://w0:8000/v1"
    ]
