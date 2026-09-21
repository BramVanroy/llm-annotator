from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Generator

import pytest

from llm_annotator.clients.vllm_online_client import (
    VLLMOnlineClient,
    VLLMOnlineRuntimeOptions,
    server_is_healthy,
)


pytestmark = [pytest.mark.integration, pytest.mark.slow]

SERVER_STARTUP_TIMEOUT = 900.0
"""Seconds to wait for `vllm serve` to answer /health, weights download
included."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "confident": {"type": "boolean"},
    },
    "required": ["city", "confident"],
}


def free_port() -> int:
    """Return a TCP port that is free at the moment of the call."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture(scope="module")
def vllm_server(test_model_id: str) -> Generator[str, None, None]:
    """Start one `vllm serve` for this module and yield its base URL.

    The server is the only way to exercise the route the pool path uses, so
    this test is skipped rather than faked when no GPU or no `vllm`
    executable is available.
    """
    if shutil.which("vllm") is None:
        pytest.skip("The 'vllm' executable is not on PATH.")
    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("A vLLM server needs a GPU.")
    except ImportError:
        pytest.skip("torch is not installed.")

    port = free_port()
    base_url = f"http://127.0.0.1:{port}/v1"
    env = dict(os.environ, VLLM_LOGGING_LEVEL="WARNING")
    process = subprocess.Popen(
        [
            "vllm",
            "serve",
            test_model_id,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--max-model-len",
            "2048",
            "--max-num-seqs",
            "16",
            "--enforce-eager",
            "--gpu-memory-utilization",
            "0.6",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
    try:
        while not server_is_healthy(base_url, timeout=2.0):
            if process.poll() is not None:
                pytest.fail(
                    f"'vllm serve' exited with code {process.returncode}."
                )
            if time.monotonic() > deadline:
                pytest.fail(
                    "'vllm serve' did not answer /health within"
                    f" {SERVER_STARTUP_TIMEOUT:g}s."
                )
            time.sleep(5)
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=60)


def test_batch_generate_against_a_real_server(
    vllm_server: str, test_model_id: str
) -> None:
    """A schema-constrained batch comes back per sample and in order.

    This covers what the fake SDK cannot: that vLLM accepts the
    ``response_format`` this client sends and constrains the output to the
    schema, and that each of the concurrent requests carries its own usage.
    """
    cities = ["Antwerp", "Ghent", "Bruges", "Leuven", "Mechelen"]
    client = VLLMOnlineClient(
        model=test_model_id, base_url=vllm_server, on_error="raise"
    )
    messages = [
        [
            {
                "role": "user",
                "content": (
                    f"Which Belgian city is this? It is called {city}."
                ),
            }
        ]
        for city in cities
    ]

    responses = client.batch_generate(
        messages=messages,
        options=VLLMOnlineRuntimeOptions(
            max_completion_tokens=64,
            temperature=0.0,
            seed=0,
            json_schema=ANSWER_SCHEMA,
        ),
    )

    assert len(responses) == len(cities)
    for response in responses:
        assert response.error is None
        assert response.num_output_tokens is not None
        assert response.num_output_tokens > 0
        parsed = json.loads(response.text)
        assert set(parsed) == {"city", "confident"}
        assert isinstance(parsed["city"], str)
        assert isinstance(parsed["confident"], bool)


def test_batch_generate_keeps_a_failed_conversation_to_itself(
    vllm_server: str, test_model_id: str
) -> None:
    """One conversation over the context length errors on its own.

    On the batch route vLLM rejected the whole request, so a single oversized
    prompt cost the entire batch.
    """
    client = VLLMOnlineClient(
        model=test_model_id, base_url=vllm_server, on_error="ignore"
    )
    messages = [
        [{"role": "user", "content": "Say hello."}],
        [{"role": "user", "content": "word " * 4000}],
        [{"role": "user", "content": "Say goodbye."}],
    ]

    responses = client.batch_generate(
        messages=messages,
        options=VLLMOnlineRuntimeOptions(
            max_completion_tokens=16, temperature=0.0, seed=0
        ),
    )

    assert len(responses) == 3
    assert responses[0].error is None
    assert responses[2].error is None
    assert responses[1].error is not None
