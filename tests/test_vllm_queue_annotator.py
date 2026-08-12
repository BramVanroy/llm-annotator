from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset

from llm_annotator import VLLMQueueAnnotator
from llm_annotator.annotator import Annotator
from llm_annotator.clients.base import (
    Client,
    Provider,
    ProviderRuntimeOptions,
    Response,
)


class FakeVLLMClient(Client[ProviderRuntimeOptions]):
    """In-process stand-in for a remote vLLM server."""

    provider_type = Provider.VLLM

    def __init__(
        self,
        *,
        base_url: str = "http://worker",
        model: str = "fake-model",
        barrier: threading.Barrier | None = None,
        fail_after_n_batches: int | None = None,
        response_text: str | None = None,
        truncate_responses: int = 0,
    ) -> None:
        super().__init__(model=model, on_error="raise")
        self.base_url = base_url
        self.barrier = barrier
        self.fail_after_n_batches = fail_after_n_batches
        self.response_text = response_text
        self.truncate_responses = truncate_responses
        self.n_batches = 0
        self.n_samples = 0
        self.destroy_called = 0
        self.warm_up_called = 0

    def _process_response(self, response: Any) -> Response:
        raise NotImplementedError

    def _handle_stop_reason(
        self, *, stop_reason: str | None, num_output_tokens: int | None
    ) -> None:
        _ = stop_reason
        _ = num_output_tokens

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        options: ProviderRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> Response:
        _ = options
        _ = gen_kwargs
        return Response(
            text=self.response_text or messages[-1]["content"],
            stop_reason="stop",
            provider=self.provider_type,
            model=self.model,
            num_output_tokens=1,
        )

    def batch_generate(
        self,
        *,
        messages: list[list[dict[str, str]]],
        options: ProviderRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> list[Response]:
        self.n_batches += 1
        self.n_samples += len(messages)

        if (
            self.fail_after_n_batches is not None
            and self.n_batches > self.fail_after_n_batches
        ):
            raise RuntimeError(f"{self.base_url} went down")

        if self.barrier is not None:
            # Times out (BrokenBarrierError) unless the requested number of
            # servers really is busy at the same time.
            self.barrier.wait(timeout=10)

        responses = [
            self.generate(messages=msgs, options=options) for msgs in messages
        ]
        if self.truncate_responses:
            responses = responses[: -self.truncate_responses]
        return responses

    def warm_up(
        self,
        *,
        system_message: str | None = None,
        prompt_prefix: str | None = None,
        options: ProviderRuntimeOptions | None = None,
    ) -> None:
        _ = system_message
        _ = prompt_prefix
        _ = options
        self.warm_up_called += 1

    def destroy(self) -> None:
        self.destroy_called += 1


def _make_dataset(
    n_samples: int, *, task_prefix: str = "", idx_column: str = "idx"
) -> Dataset:
    return Dataset.from_dict(
        {
            idx_column: list(range(n_samples)),
            "text": [f"sample-{idx}" for idx in range(n_samples)],
            f"{task_prefix}messages": [
                [{"role": "user", "content": f"sample-{idx}"}]
                for idx in range(n_samples)
            ],
        }
    )


def _progress_rows(output_dir: Path, task_prefix: str = "") -> list[dict]:
    rows: list[dict] = []
    progress_dir = output_dir / f"{task_prefix}progress_backup"
    for pfin in sorted(progress_dir.glob("*.jsonl")):
        with pfin.open(encoding="utf-8") as fhin:
            rows.extend(json.loads(line) for line in fhin if line.strip())
    return rows


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_requires_at_least_one_client() -> None:
    # Verifies an empty client pool is rejected up front.
    with pytest.raises(ValueError, match="at least one VLLM client"):
        VLLMQueueAnnotator(clients=[])


def test_rejects_non_vllm_clients() -> None:
    # Verifies only vLLM server clients are accepted into the pool.
    class NotVLLM(FakeVLLMClient):
        provider_type = Provider.OPENAI

    with pytest.raises(TypeError, match="only supports VLLM"):
        VLLMQueueAnnotator(clients=[NotVLLM()])


def test_queue_size_defaults_and_floor() -> None:
    # Verifies queue_size defaults to two batches per client and never drops
    # below the number of clients (which would idle servers).
    clients = [FakeVLLMClient(base_url=f"http://w{i}") for i in range(3)]
    assert VLLMQueueAnnotator(clients=clients).queue_size == 6
    assert VLLMQueueAnnotator(clients=clients, queue_size=1).queue_size == 3
    assert VLLMQueueAnnotator(clients=clients, queue_size=10).queue_size == 10

    with pytest.raises(ValueError, match="positive integer"):
        VLLMQueueAnnotator(clients=clients, queue_size=0)


def test_multiprocessing_default_matches_base() -> None:
    # Verifies dataset preprocessing is not silently forced single-process.
    from llm_annotator.annotator import DEFAULT_CPU_COUNT

    annotator = VLLMQueueAnnotator(clients=[FakeVLLMClient()])
    assert annotator.num_proc == DEFAULT_CPU_COUNT


# --------------------------------------------------------------------------
# Coverage of the full dataset
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_samples", "batch_size", "queue_size", "n_clients"),
    [
        (100, 2, 2, 2),  # many more batches than the queue can hold
        (100, 8, 8, 4),
        (7, 8, 4, 2),  # a single partial batch
        (1, 1, 1, 1),
        (13, 5, 3, 3),  # uneven final batch
    ],
)
def test_annotates_every_sample_exactly_once(
    tmp_path: Path,
    n_samples: int,
    batch_size: int,
    queue_size: int,
    n_clients: int,
) -> None:
    # Verifies the bounded queue bounds memory, not the amount of work: the
    # whole dataset is annotated regardless of queue_size.
    dataset = _make_dataset(n_samples)
    clients = [
        FakeVLLMClient(base_url=f"http://w{i}") for i in range(n_clients)
    ]
    annotator = VLLMQueueAnnotator(
        clients=clients, batch_size=batch_size, queue_size=queue_size
    )

    result = annotator.run_annotation(
        output_dir=tmp_path / "out",
        prepared_dataset=dataset,
        keep_idx_column=True,
    )

    assert sorted(result["idx"]) == list(range(n_samples))
    assert result["response"] == [f"sample-{idx}" for idx in range(n_samples)]
    assert sum(client.n_samples for client in clients) == n_samples


def test_results_are_sorted_by_idx(tmp_path: Path) -> None:
    # Verifies out-of-order completion is repaired by the final idx sort.
    dataset = _make_dataset(50)
    annotator = VLLMQueueAnnotator(
        clients=[FakeVLLMClient(base_url=f"http://w{i}") for i in range(4)],
        batch_size=3,
    )

    result = annotator.run_annotation(
        output_dir=tmp_path / "out",
        prepared_dataset=dataset,
        keep_idx_column=True,
    )

    assert result["idx"] == list(range(50))


# --------------------------------------------------------------------------
# Parallelism
# --------------------------------------------------------------------------


def test_clients_run_in_parallel(tmp_path: Path) -> None:
    # Verifies all servers work at the same time: the barrier only clears when
    # `n_clients` batches are in flight simultaneously.
    n_clients = 4
    barrier = threading.Barrier(n_clients)
    clients = [
        FakeVLLMClient(base_url=f"http://w{i}", barrier=barrier)
        for i in range(n_clients)
    ]
    annotator = VLLMQueueAnnotator(clients=clients, batch_size=2)

    result = annotator.run_annotation(
        output_dir=tmp_path / "out",
        prepared_dataset=_make_dataset(16),
        keep_idx_column=True,
    )

    assert len(result) == 16
    assert all(client.n_batches > 0 for client in clients)


def test_queue_size_bounds_in_flight_batches(tmp_path: Path) -> None:
    # Verifies no more than queue_size batches are dispatched before results
    # are consumed.
    lock = threading.Lock()
    state = {"in_flight": 0, "max_in_flight": 0}

    class CountingClient(FakeVLLMClient):
        def batch_generate(self, **kwargs: Any) -> list[Response]:
            with lock:
                state["in_flight"] += 1
                state["max_in_flight"] = max(
                    state["max_in_flight"], state["in_flight"]
                )
            try:
                return super().batch_generate(**kwargs)
            finally:
                with lock:
                    state["in_flight"] -= 1

    clients = [CountingClient(base_url=f"http://w{i}") for i in range(4)]
    annotator = VLLMQueueAnnotator(clients=clients, batch_size=2, queue_size=4)
    annotator.run_annotation(
        output_dir=tmp_path / "out", prepared_dataset=_make_dataset(40)
    )

    assert state["max_in_flight"] <= 4


# --------------------------------------------------------------------------
# Output shape
# --------------------------------------------------------------------------


def test_output_columns_match_the_base_annotator(tmp_path: Path) -> None:
    # Verifies the queue annotator writes exactly the documented column set.
    dataset = _make_dataset(6)

    base = Annotator(client=FakeVLLMClient(), batch_size=2)
    base_result = base.run_annotation(
        output_dir=tmp_path / "base",
        prepared_dataset=dataset,
        keep_idx_column=True,
    )

    queued = VLLMQueueAnnotator(
        clients=[FakeVLLMClient(), FakeVLLMClient()], batch_size=2
    )
    queue_result = queued.run_annotation(
        output_dir=tmp_path / "queue",
        prepared_dataset=dataset,
        keep_idx_column=True,
    )

    assert sorted(queue_result.column_names) == sorted(
        base_result.column_names
    )
    assert queue_result.to_dict() == base_result.to_dict()


def test_task_prefix_namespaces_columns_and_paths(tmp_path: Path) -> None:
    # Verifies task_prefix is honoured with no unprefixed duplicate columns.
    dataset = _make_dataset(6, task_prefix="tp_")
    annotator = VLLMQueueAnnotator(
        clients=[FakeVLLMClient(), FakeVLLMClient()], batch_size=2
    )

    out_dir = tmp_path / "out"
    result = annotator.run_annotation(
        output_dir=out_dir,
        prepared_dataset=dataset,
        task_prefix="tp_",
        keep_idx_column=True,
    )

    assert "tp_response" in result.column_names
    for column in ("response", "finish_reason", "num_tokens", "error"):
        assert column not in result.column_names
    assert (out_dir / "tp_progress_backup").is_dir()


def test_keep_columns_are_written_through(tmp_path: Path) -> None:
    # Verifies source columns can be carried into the annotated output.
    annotator = VLLMQueueAnnotator(clients=[FakeVLLMClient()], batch_size=2)
    result = annotator.run_annotation(
        output_dir=tmp_path / "out",
        prepared_dataset=_make_dataset(4),
        keep_columns="text",
        keep_idx_column=True,
    )

    assert result["text"] == [f"sample-{idx}" for idx in range(4)]


def test_short_client_response_is_rejected(tmp_path: Path) -> None:
    # Verifies a client returning fewer responses than inputs is an error
    # rather than a silent sample drop.
    annotator = VLLMQueueAnnotator(
        clients=[FakeVLLMClient(truncate_responses=1)], batch_size=4
    )

    with pytest.raises(ValueError, match="exactly one response per input"):
        annotator.run_annotation(
            output_dir=tmp_path / "out", prepared_dataset=_make_dataset(8)
        )


# --------------------------------------------------------------------------
# Resumption
# --------------------------------------------------------------------------


def test_resume_returns_previous_and_new_rows(tmp_path: Path) -> None:
    # Verifies a resumed run returns the complete dataset, not just the rows
    # produced by the final invocation.
    out_dir = tmp_path / "out"
    progress_dir = out_dir / "progress_backup"
    progress_dir.mkdir(parents=True)
    with (progress_dir / "resume.jsonl").open("w", encoding="utf-8") as fhout:
        fhout.write(
            json.dumps(
                {
                    "idx": 1,
                    "response": "already-done",
                    "finish_reason": "stop",
                    "num_tokens": 1,
                    "error": None,
                    "error_type": None,
                }
            )
            + "\n"
        )

    annotator = VLLMQueueAnnotator(clients=[FakeVLLMClient()], batch_size=2)
    result = annotator.run_annotation(
        output_dir=out_dir,
        prepared_dataset=_make_dataset(3),
        keep_idx_column=True,
    )

    assert result["idx"] == [0, 1, 2]
    assert result["response"] == ["sample-0", "already-done", "sample-2"]


def test_crash_and_resume_has_no_gaps_or_duplicates(tmp_path: Path) -> None:
    # Verifies a hard mid-run failure loses no idx and duplicates none.
    out_dir = tmp_path / "out"
    dataset = _make_dataset(60)

    crashing = [
        FakeVLLMClient(base_url=f"http://w{i}", fail_after_n_batches=2)
        for i in range(2)
    ]
    crashing_annotator = VLLMQueueAnnotator(
        clients=crashing, batch_size=5, queue_size=2
    )

    with pytest.raises(RuntimeError, match="went down"):
        crashing_annotator.run_annotation(
            output_dir=out_dir, prepared_dataset=dataset, keep_idx_column=True
        )

    assert all(client.destroy_called == 1 for client in crashing)
    partial_idxs = [row["idx"] for row in _progress_rows(out_dir)]
    assert 0 < len(partial_idxs) < 60
    assert len(set(partial_idxs)) == len(partial_idxs)

    healthy_annotator = VLLMQueueAnnotator(
        clients=[FakeVLLMClient(base_url=f"http://w{i}") for i in range(2)],
        batch_size=5,
    )
    result = healthy_annotator.run_annotation(
        output_dir=out_dir, prepared_dataset=dataset, keep_idx_column=True
    )

    assert result["idx"] == list(range(60))
    assert result["response"] == [f"sample-{idx}" for idx in range(60)]

    all_idxs = [row["idx"] for row in _progress_rows(out_dir)]
    assert sorted(all_idxs) == list(range(60))


def test_resume_repairs_a_truncated_progress_line(tmp_path: Path) -> None:
    # Verifies a half-written line from a killed job is dropped and re-annotated
    # instead of breaking the resume.
    out_dir = tmp_path / "out"
    progress_dir = out_dir / "progress_backup"
    progress_dir.mkdir(parents=True)
    pfout = progress_dir / "progress_0.jsonl"
    with pfout.open("w", encoding="utf-8") as fhout:
        fhout.write(
            json.dumps(
                {
                    "idx": 0,
                    "response": "done",
                    "finish_reason": "stop",
                    "num_tokens": 1,
                    "error": None,
                    "error_type": None,
                }
            )
            + "\n"
        )
        fhout.write('{"idx": 1, "response": "half-writ')

    annotator = VLLMQueueAnnotator(clients=[FakeVLLMClient()], batch_size=2)
    result = annotator.run_annotation(
        output_dir=out_dir,
        prepared_dataset=_make_dataset(3),
        keep_idx_column=True,
    )

    assert result["idx"] == [0, 1, 2]
    assert result["response"] == ["done", "sample-1", "sample-2"]


def test_fully_processed_run_short_circuits(tmp_path: Path) -> None:
    # Verifies a completed run is re-assembled without any new inference.
    out_dir = tmp_path / "out"
    dataset = _make_dataset(4)
    annotator = VLLMQueueAnnotator(clients=[FakeVLLMClient()], batch_size=2)
    annotator.run_annotation(
        output_dir=out_dir, prepared_dataset=dataset, keep_idx_column=True
    )

    second_client = FakeVLLMClient()
    result = VLLMQueueAnnotator(
        clients=[second_client], batch_size=2
    ).run_annotation(
        output_dir=out_dir, prepared_dataset=dataset, keep_idx_column=True
    )

    assert second_client.n_batches == 0
    assert result["idx"] == [0, 1, 2, 3]


# --------------------------------------------------------------------------
# Progress backup / Hub cadence
# --------------------------------------------------------------------------


def test_progress_files_are_chunked_and_uploaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Verifies max_samples_per_output_file and upload_every_n_samples are
    # actually honoured (they used to be accepted and ignored).
    uploads: list[Path] = []

    def _fake_push(
        self: Annotator,
        dir_path: Path | str,
        hub_id: str | None = None,
        *,
        task_prefix: str = "",
    ) -> None:
        _ = self, hub_id, task_prefix
        uploads.append(Path(dir_path))

    monkeypatch.setattr(Annotator, "push_progress_to_hub", _fake_push)
    monkeypatch.setattr(
        Annotator,
        "_post_annotate",
        lambda self, **kwargs: Dataset.from_dict({}),
    )

    out_dir = tmp_path / "out"
    annotator = VLLMQueueAnnotator(
        clients=[FakeVLLMClient(), FakeVLLMClient()], batch_size=2
    )
    annotator.run_annotation(
        output_dir=out_dir,
        prepared_dataset=_make_dataset(20),
        hub_id="fake/repo",
        upload_every_n_samples=10,
        max_samples_per_output_file=10,
    )

    jsonl_files = sorted((out_dir / "progress_backup").glob("*.jsonl"))
    assert len(jsonl_files) == 2
    assert len(uploads) >= 2
    assert len(_progress_rows(out_dir)) == 20


# --------------------------------------------------------------------------
# Inherited entry points, warm-up and cleanup
# --------------------------------------------------------------------------


def test_annotate_dataset_entry_point(tmp_path: Path) -> None:
    # Verifies the inherited prepare_data + run_annotation wrapper works on the
    # subclass (its signature used to be incompatible).
    clients = [FakeVLLMClient(base_url=f"http://w{i}") for i in range(2)]
    annotator = VLLMQueueAnnotator(clients=clients, batch_size=2)

    result = annotator.annotate_dataset(
        output_dir=tmp_path / "out",
        prompt_template="Classify: {text}",
        dataset=Dataset.from_dict({"text": [f"t{i}" for i in range(6)]}),
        keep_idx_column=True,
    )

    assert result["idx"] == list(range(6))
    assert result["response"] == [f"Classify: t{i}" for i in range(6)]
    assert all(client.warm_up_called == 1 for client in clients)


def test_generate_dataset_entry_point(tmp_path: Path) -> None:
    # Verifies the synthetic-prompt entry point works on the subclass.
    annotator = VLLMQueueAnnotator(
        clients=[FakeVLLMClient(), FakeVLLMClient()], batch_size=2
    )

    result = annotator.generate_dataset(
        output_dir=tmp_path / "out",
        prompts=[
            f"Write about {topic}" for topic in ("cats", "dogs", "birds")
        ],
        keep_idx_column=True,
    )

    assert result["idx"] == [0, 1, 2]
    assert result["response"] == [
        "Write about cats",
        "Write about dogs",
        "Write about birds",
    ]


def test_context_manager_destroys_every_client() -> None:
    # Verifies cleanup covers the whole pool, not just clients[0].
    clients = [FakeVLLMClient(base_url=f"http://w{i}") for i in range(3)]
    with VLLMQueueAnnotator(clients=clients) as annotator:
        assert annotator.client is clients[0]

    assert [client.destroy_called for client in clients] == [1, 1, 1]


def test_destroy_continues_after_a_failing_client() -> None:
    # Verifies one broken client cannot leak the others' resources.
    class BrokenClient(FakeVLLMClient):
        def destroy(self) -> None:
            super().destroy()
            raise RuntimeError("cannot close")

    broken = BrokenClient(base_url="http://broken")
    healthy = FakeVLLMClient(base_url="http://healthy")
    annotator = VLLMQueueAnnotator(clients=[broken, healthy])

    with pytest.raises(RuntimeError, match="cannot close"):
        annotator.destroy()

    assert healthy.destroy_called == 1


def test_retries_invalid_samples(tmp_path: Path) -> None:
    # Verifies the inherited invalid-output retry loop runs per worker.
    seen: dict[str, int] = {"n": 0}

    def _validate(sample: dict[str, Any]) -> bool:
        seen["n"] += 1
        # Only the first response of every sample is rejected.
        return seen["n"] > 4

    annotator = VLLMQueueAnnotator(clients=[FakeVLLMClient()], batch_size=4)
    result = annotator.run_annotation(
        output_dir=tmp_path / "out",
        prepared_dataset=_make_dataset(4),
        validate_fn=_validate,
        num_retries_invalid=2,
        keep_idx_column=True,
    )

    assert result["valid"] == [True, True, True, True]
    assert seen["n"] == 8
