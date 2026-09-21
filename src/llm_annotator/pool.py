"""Live clients and annotators built from a validated client config.

Everything here acts on the world outside the process: it constructs provider
clients, probes ``/health`` endpoints over HTTP and keeps a background thread
that admits vLLM servers into a running pool.
[`config`][llm_annotator.config] holds the models and their validation and
knows nothing about this module, so the dependency runs one way:
``pipeline`` -> ``pool`` -> ``config``.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llm_annotator.annotator import Annotator, VLLMQueueAnnotator
from llm_annotator.clients.base import Client
from llm_annotator.clients.vllm_online_client import (
    VLLMOnlineClient,
    server_is_healthy,
)
from llm_annotator.config import ClientConfig, _client_class
from llm_annotator.logging_utils import get_logger


LOGGER = get_logger("pool")


def wait_for_servers(
    base_urls: list[str], timeout: float, min_servers: int = 1
) -> list[str]:
    """Block until the requested number of vLLM servers answer ``/health``.

    Args:
        base_urls: vLLM base URLs (each ending in ``/v1``).
        timeout: Maximum number of seconds to wait for the pool as a whole.
        min_servers: Number of ready servers required to continue.

    Returns:
        The base URLs that answered, in the order they did.

    Raises:
        TimeoutError: If fewer than ``min_servers`` are reachable after
            ``timeout``.
    """
    pending = list(dict.fromkeys(base_urls))
    if not 1 <= min_servers <= len(pending):
        raise ValueError(
            f"'min_servers' must be between 1 and {len(pending)}, got"
            f" {min_servers}."
        )

    ready: list[str] = []
    deadline = time.monotonic() + timeout
    while pending:
        remaining = deadline - time.monotonic()
        if remaining < 0:
            break
        with ThreadPoolExecutor(max_workers=len(pending)) as pool:
            results = zip(
                pending,
                pool.map(
                    lambda url: server_is_healthy(
                        url, min(5, max(remaining, 0))
                    ),
                    pending,
                ),
                strict=True,
            )
            newly_ready = []
            for url, is_ready in results:
                if is_ready:
                    ready.append(url)
                    newly_ready.append(url)
        pending = [url for url in pending if url not in newly_ready]
        if len(ready) >= min_servers:
            return ready
        time.sleep(min(5, max(0, deadline - time.monotonic())))

    raise TimeoutError(
        f"Only {len(ready)} of {min_servers} required vLLM server(s) became"
        f" ready within {timeout:g}s."
    )


def build_client(
    client_config: ClientConfig, root: Path
) -> Client | list[Client]:
    """Instantiate the client, or one client per server for a pool.

    Args:
        client_config: The step's effective client configuration.
        root: Directory that relative paths and globs resolve against.

    Returns:
        A single client, or a list of clients when a pool is configured.
    """
    kwargs = dict(client_config.init)
    if client_config.model is not None:
        kwargs["model"] = client_config.model
    if client_config.provider == "vllm_offline":
        engine_kwargs = client_config.engine.as_llm_kwargs()
        # `extra` names arguments the engine block does not, so it can only
        # travel as the client's own passthrough.
        kwargs.update(
            {
                k: v
                for k, v in engine_kwargs.items()
                if k not in client_config.engine.extra
            }
        )
        if client_config.engine.extra:
            kwargs["extra_vllm_kwargs"] = dict(client_config.engine.extra)

    if not client_config.is_pool():
        return _client_class(client_config.provider)(**kwargs)

    # A pool is `vllm_online`-only (enforced in validation), so the server
    # client is named directly here rather than looked up: only it takes the
    # `base_url` that distinguishes one pool member from the next.
    base_urls = client_config.resolve_base_urls(root)
    if client_config.wait_for_servers:
        # The pool source is read once here, so it can name fewer servers than
        # `min_servers` while the rest are still starting. Waiting for more
        # servers than it names could never succeed; the ones that arrive
        # later are admitted by the watcher instead.
        min_ready = min(
            client_config.pool.min_servers, len(dict.fromkeys(base_urls))
        )
        LOGGER.info(
            f"Waiting up to {client_config.wait_for_servers:g}s for"
            f" {min_ready} vLLM server(s) to become ready..."
        )
        base_urls = wait_for_servers(
            base_urls, client_config.wait_for_servers, min_ready
        )

    return [VLLMOnlineClient(base_url=url, **kwargs) for url in base_urls]


def build_annotator(
    client_config: ClientConfig, root: Path, verbose: bool = False
) -> Annotator:
    """Instantiate the annotator that drives a step's client.

    A pool of servers yields a
    [`VLLMQueueAnnotator`][llm_annotator.annotator.VLLMQueueAnnotator], with a
    background thread that admits the servers that are not ready yet;
    everything else yields a plain
    [`Annotator`][llm_annotator.annotator.Annotator].

    Args:
        client_config: The step's effective client configuration.
        root: Directory that relative paths and globs resolve against.
        verbose: Whether the annotator should log progress information.

    Returns:
        The annotator, ready to run.
    """
    one_or_more_clients = build_client(client_config, root)
    if isinstance(one_or_more_clients, list):
        LOGGER.info(
            f"Annotating over {len(one_or_more_clients)} vLLM server(s)."
        )
        expected_servers = _expected_pool_size(client_config, root)
        annotator = VLLMQueueAnnotator(
            clients=one_or_more_clients,
            batch_size=client_config.batch_size,
            queue_size=client_config.queue_size,
            max_concurrent_batches_per_client=(
                client_config.max_concurrent_batches_per_client
            ),
            max_workers=(
                expected_servers
                * client_config.max_concurrent_batches_per_client
            ),
            num_proc=client_config.num_proc,
            verbose=verbose,
        )
        _watch_pool(client_config, root, annotator)
        return annotator
    return Annotator(
        client=one_or_more_clients,
        batch_size=client_config.batch_size,
        num_proc=client_config.num_proc,
        verbose=verbose,
    )


def _expected_pool_size(client_config: ClientConfig, root: Path) -> int:
    """Return how many distinct vLLM servers a pool can grow to.

    Args:
        client_config: The step's effective client configuration.
        root: Directory that relative paths and globs resolve against.

    Returns:
        The number of servers the pool accounts for at full size.
    """
    if client_config.base_urls:
        return client_config.configured_servers()
    try:
        return max(
            client_config.pool.servers,
            len(list(dict.fromkeys(client_config.resolve_base_urls(root)))),
        )
    except ValueError:
        return client_config.pool.servers


def _watch_pool(
    client_config: ClientConfig, root: Path, annotator: VLLMQueueAnnotator
) -> None:
    """Add configured vLLM servers to an active pool once they are ready.

    A server is a candidate whenever the pool does not hold it, so this admits
    a server that starts late as well as one that the annotator evicted and
    that answers ``/health`` again.

    Args:
        client_config: The step's effective client configuration.
        root: Directory that relative paths and globs resolve against.
        annotator: The running annotator whose pool is grown.
    """
    kwargs = dict(client_config.init)
    if client_config.model is not None:
        kwargs["model"] = client_config.model

    def watch() -> None:
        while not annotator.is_shutting_down:
            pooled_urls = annotator.client_base_urls()
            try:
                configured = client_config.resolve_base_urls(root)
            except ValueError:
                # The pool source may not exist yet, or hold nothing.
                configured = []
            candidates = [url for url in configured if url not in pooled_urls]
            with ThreadPoolExecutor(max_workers=len(candidates) or 1) as pool:
                readiness = list(
                    pool.map(lambda url: server_is_healthy(url, 5), candidates)
                )

            for url, is_ready in zip(candidates, readiness, strict=True):
                if annotator.is_shutting_down:
                    return
                if is_ready:
                    annotator.add_client_for_base_url(
                        url,
                        lambda base_url: VLLMOnlineClient(
                            base_url=base_url, **kwargs
                        ),
                    )
            if annotator.wait_for_shutdown(timeout=1):
                return

    threading.Thread(
        target=watch, name="vllm-pool-watcher", daemon=True
    ).start()


__all__ = ["build_annotator", "build_client", "wait_for_servers"]
