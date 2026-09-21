"""Tests for the SLURM job scripts under `slurm/`.

The scripts are driven with a fake `llm-annotate` on `PATH` that records the
arguments it was called with, so the wait loop and the flags it builds can be
checked without a scheduler, a GPU or a model. Nothing here needs Slurm, and
nothing here may reach it: `sbatch`, `scancel` and `squeue` are shadowed by
stubs, so a submitter that stopped honouring `--dry-run` fails the test rather
than queueing jobs on whatever machine the suite happens to run on.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
SLURM_DIR = REPO_ROOT / "slurm"
VENV_PATH = Path(sys.prefix)

pytestmark = pytest.mark.skipif(
    not Path("/bin/bash").exists(), reason="the SLURM scripts need bash"
)


def _fake_bin(tmp_path: Path) -> Path:
    """Create a bin directory with the fakes every script run needs.

    Args:
        tmp_path: Directory to create the fake binaries in.

    Returns:
        The directory to prepend to ``PATH``.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)

    annotate = bin_dir / "llm-annotate"
    annotate.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > "$ARGS_FILE"\n', encoding="utf-8"
    )
    annotate.chmod(0o755)

    # The wait loop sleeps ten seconds between polls, which a timeout test
    # would otherwise have to sit through.
    sleeper = bin_dir / "sleep"
    sleeper.write_text("#!/bin/bash\nexec /bin/sleep 0.2\n", encoding="utf-8")
    sleeper.chmod(0o755)

    # `pool_alive` shells out to squeue; FAKE_SQUEUE_OUTPUT lets a test say
    # what it reports for the array without a scheduler. Unset, it behaves
    # like a live array so tests that never set SERVER_JOB_ID are unaffected
    # (pool_alive returns early for them and never calls this at all).
    squeue = bin_dir / "squeue"
    squeue.write_text(
        "#!/bin/bash\nprintf '%s' \"${FAKE_SQUEUE_OUTPUT-RUNNING}\"\n",
        encoding="utf-8",
    )
    squeue.chmod(0o755)

    # The suite runs on login nodes too, where a real sbatch is on PATH and a
    # submitter that stopped honouring --dry-run would queue jobs for real.
    # These stubs make that a failed test instead.
    for name in ("sbatch", "scancel"):
        stub = bin_dir / name
        stub.write_text(
            f'#!/bin/bash\necho "{name} was called from a test" >&2\nexit 1\n',
            encoding="utf-8",
        )
        stub.chmod(0o755)

    return bin_dir


def _run_annotate(
    tmp_path: Path, pool_dir: Path, **overrides: str
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run `slurm/vllm_annotate.sh` against a fake pool directory.

    Args:
        tmp_path: Directory for the fake binaries, logs and argument record.
        pool_dir: Pool directory the client watches.
        **overrides: Environment variables set on top of the defaults.

    Returns:
        The finished process and the arguments the fake CLI was called with.
    """
    args_file = tmp_path / "args.txt"
    env = {
        "PATH": f"{_fake_bin(tmp_path)}:{os.environ['PATH']}",
        "REPO_ROOT": str(REPO_ROOT),
        "CLUSTER_ENV": str(tmp_path / "absent.env"),
        "LOG_DIR": str(tmp_path / "logs"),
        "VENV_PATH": str(tmp_path / "absent-venv"),
        "ANNOTATE_CONFIG": str(tmp_path / "pipeline.yaml"),
        "STEP_NAME": "write",
        "POOL_DIR": str(pool_dir),
        "NUM_SERVERS": "4",
        "POOL_WAIT": "60",
        "CANCEL_SERVERS_ON_EXIT": "0",
        "ARGS_FILE": str(args_file),
    }
    env.update(overrides)

    process = subprocess.run(
        ["/bin/bash", str(SLURM_DIR / "vllm_annotate.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    recorded = (
        args_file.read_text(encoding="utf-8").splitlines()
        if args_file.exists()
        else []
    )
    return process, recorded


def _publish(pool_dir: Path, count: int) -> None:
    """Write `count` server URL files into a pool directory."""
    pool_dir.mkdir(parents=True, exist_ok=True)
    for task in range(count):
        (pool_dir / f"{task}.url").write_text(
            f"http://node{task:02d}:8000/v1\n", encoding="utf-8"
        )


def test_annotate_starts_on_a_partial_pool(tmp_path: Path) -> None:
    """Two of four servers is enough when MIN_SERVERS is two."""
    pool_dir = tmp_path / "pool"
    _publish(pool_dir, 2)

    process, recorded = _run_annotate(tmp_path, pool_dir, MIN_SERVERS="2")

    assert process.returncode == 0, process.stderr
    assert "starting at 2 of 4 server(s)" in process.stdout
    assert recorded == [
        str(tmp_path / "pipeline.yaml"),
        "--steps",
        "write",
        "--url-glob",
        f"{pool_dir}/*.url",
    ]


def test_annotate_passes_the_pool_glob_not_a_snapshot(
    tmp_path: Path,
) -> None:
    """The client gets the directory itself, so late servers still join."""
    pool_dir = tmp_path / "pool"
    _publish(pool_dir, 4)

    process, recorded = _run_annotate(tmp_path, pool_dir, MIN_SERVERS="4")

    assert process.returncode == 0, process.stderr
    assert "--hosts-file" not in recorded
    assert not (pool_dir / "hosts.txt").exists()


def test_annotate_waits_for_the_whole_pool_by_default(
    tmp_path: Path,
) -> None:
    """Without MIN_SERVERS the script waits for NUM_SERVERS servers."""
    pool_dir = tmp_path / "pool"
    _publish(pool_dir, 1)

    process, recorded = _run_annotate(
        tmp_path, pool_dir, NUM_SERVERS="3", POOL_WAIT="0"
    )

    assert process.returncode == 0, process.stderr
    assert "starting at 3 of 3 server(s)" in process.stdout
    assert "Waited 0s for 3 server(s), 1 showed up." in process.stdout
    assert "--url-glob" in recorded


def test_annotate_fails_when_no_server_registers(tmp_path: Path) -> None:
    """An empty pool directory after the timeout is an error, not a run."""
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()

    process, recorded = _run_annotate(
        tmp_path, pool_dir, MIN_SERVERS="1", POOL_WAIT="0"
    )

    assert process.returncode == 1
    assert "No server registered" in process.stderr
    assert recorded == []


def test_annotate_stops_waiting_when_the_server_array_is_gone(
    tmp_path: Path,
) -> None:
    """An empty pool stops waiting once squeue reports no array left."""
    pool_dir = tmp_path / "pool"
    _publish(pool_dir, 1)

    started = time.monotonic()
    process, recorded = _run_annotate(
        tmp_path,
        pool_dir,
        MIN_SERVERS="3",
        SERVER_JOB_ID="12345",
        POOL_WAIT="600",
        FAKE_SQUEUE_OUTPUT="",
    )
    elapsed = time.monotonic() - started

    assert process.returncode == 0, process.stderr
    assert elapsed < 30, "should stop well short of POOL_WAIT, not sit it out"
    assert "has no element left in the queue" in process.stdout
    assert "--url-glob" in recorded


def test_annotate_forwards_config_overrides(tmp_path: Path) -> None:
    """ANNOTATE_SET becomes one --set per line, values kept whole."""
    pool_dir = tmp_path / "pool"
    _publish(pool_dir, 1)

    process, recorded = _run_annotate(
        tmp_path,
        pool_dir,
        MIN_SERVERS="1",
        ANNOTATE_SET=(
            "dataset.max_num_samples=50000\n"
            "dataset.shuffle_seed=42\n"
            "steps.0.prompt=two words"
        ),
    )

    assert process.returncode == 0, process.stderr
    assert recorded[-6:] == [
        "--set",
        "dataset.max_num_samples=50000",
        "--set",
        "dataset.shuffle_seed=42",
        "--set",
        "steps.0.prompt=two words",
    ]


def _write_pool_config(
    tmp_path: Path,
    pool: dict[str, int],
    name: str = "write",
    model: str = "Qwen/Qwen3-8B",
) -> Path:
    """Write a one-step pooled-vLLM config and return its path.

    Args:
        tmp_path: Directory to write the config into.
        pool: The step's ``client.pool`` block.
        name: Name of the single step.
        model: The step's ``client.model``.

    Returns:
        Path of the config file.
    """
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "out"),
                "dataset": {"name": "stanfordnlp/imdb", "split": "test"},
                "steps": [
                    {
                        "name": name,
                        "prompt": "x {text}",
                        "client": {
                            "provider": "vllm_online",
                            "model": model,
                            "pool": pool,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _write_two_step_config(tmp_path: Path) -> Path:
    """Write a pooled-vLLM step followed by a hosted step."""
    config_path = tmp_path / "two-step.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "out"),
                "dataset": {"name": "stanfordnlp/imdb", "split": "test"},
                "steps": [
                    {
                        "name": "write",
                        "prompt": "x {text}",
                        "client": {
                            "provider": "vllm_online",
                            "model": "Qwen/Qwen3-8B",
                            "pool": {"servers": 2, "min_servers": 1},
                        },
                    },
                    {
                        "name": "rate",
                        "prompt": "y {write_response}",
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


def _run_submit(
    tmp_path: Path,
    config_path: Path,
    dry_run: bool = True,
    extra: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run `slurm/submit_pipeline.sh` over one config without a scheduler.

    `SBATCH_CMD` points at the stub in the test bin, so even the runs that are
    not dry queue nothing. The submitter reads the config through the installed
    CLI, which it looks for under ``VENV_PATH`` before ``PATH``, so the fake
    `llm-annotate` beside that stub is not what answers `--describe-steps`.

    Args:
        tmp_path: Directory for the fake binaries and the log directory.
        config_path: Pipeline config to submit.
        dry_run: Whether to pass ``--dry-run``.
        extra: Further arguments for the submitter.
        env: Environment variables set on top of the defaults.

    Returns:
        The finished process.
    """
    bin_dir = _fake_bin(tmp_path)
    command = ["/bin/bash", str(SLURM_DIR / "submit_pipeline.sh")]
    if dry_run:
        command.append("--dry-run")
    command.extend(extra or [])
    command.append(str(config_path))

    environment = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SBATCH_CMD": str(bin_dir / "sbatch"),
        "REPO_ROOT": str(REPO_ROOT),
        "CLUSTER_ENV": str(tmp_path / "absent.env"),
        "LOG_DIR": str(tmp_path / "logs"),
        "VENV_PATH": str(VENV_PATH),
        "HOME": os.environ.get("HOME", str(tmp_path)),
    }
    environment.update(env or {})

    return subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.mark.skipif(
    not (VENV_PATH / "bin" / "llm-annotate").exists(),
    reason="the submitter reads the config through the installed CLI",
)
def test_submit_pipeline_exports_min_servers(tmp_path: Path) -> None:
    """The submitter hands the step's own readiness threshold to its client."""
    config_path = _write_pool_config(
        tmp_path, {"servers": 4, "min_servers": 2}
    )

    process = _run_submit(tmp_path, config_path)

    assert process.returncode == 0, process.stderr
    assert "client starts at 2 ready server(s)" in process.stdout
    assert "NUM_SERVERS=4,MIN_SERVERS=2" in process.stderr


@pytest.mark.skipif(
    not (VENV_PATH / "bin" / "llm-annotate").exists(),
    reason="the submitter reads the config through the installed CLI",
)
def test_submit_pipeline_prints_the_pool_concurrency(tmp_path: Path) -> None:
    """A pool's request load is visible before any GPU is allocated."""
    config_path = _write_pool_config(tmp_path, {"servers": 4})

    process = _run_submit(tmp_path, config_path)

    assert process.returncode == 0, process.stderr
    # 4 concurrent batches per server times the default batch size of 256,
    # over four servers, with four batches queued per batch slot.
    assert (
        "up to 1024 requests per server, 4096 over the pool,"
        " queue of 64 batches" in process.stdout
    )


@pytest.mark.skipif(
    not (VENV_PATH / "bin" / "llm-annotate").exists(),
    reason="the submitter reads the config through the installed CLI",
)
def test_submit_pipeline_pool_dependency_is_or_joined(
    tmp_path: Path,
) -> None:
    """The client is released by any one server, not the whole array."""
    config_path = _write_pool_config(
        tmp_path, {"servers": 4, "min_servers": 2}
    )

    process = _run_submit(tmp_path, config_path)

    assert process.returncode == 0, process.stderr
    assert (
        "--dependency=after:<job-1>_1?after:<job-1>_2?"
        "after:<job-1>_3?after:<job-1>_4" in process.stderr
    )
    assert "--kill-on-invalid-dep=yes" in process.stderr


@pytest.mark.skipif(
    not (VENV_PATH / "bin" / "llm-annotate").exists(),
    reason="the submitter reads the config through the installed CLI",
)
def test_submit_pipeline_stops_when_a_submit_is_refused(
    tmp_path: Path,
) -> None:
    """A refused sbatch ends the run instead of chaining onto a missing id."""
    config_path = _write_pool_config(
        tmp_path, {"servers": 2, "min_servers": 1}
    )

    process = _run_submit(tmp_path, config_path, dry_run=False)

    assert process.returncode == 1
    assert "could not queue the servers of step 'write'" in process.stderr
    assert "Submitted" not in process.stdout


@pytest.mark.skipif(
    not (VENV_PATH / "bin" / "llm-annotate").exists(),
    reason="the submitter reads the config through the installed CLI",
)
def test_submit_pipeline_rejects_a_set_without_a_value(
    tmp_path: Path,
) -> None:
    """A --set that is not KEY=VALUE fails on the login node."""
    config_path = _write_pool_config(
        tmp_path, {"servers": 2, "min_servers": 1}
    )

    process = _run_submit(
        tmp_path, config_path, extra=["--set", "dataset.max_num_samples"]
    )

    assert process.returncode == 1
    assert "--set needs KEY=VALUE" in process.stderr


@pytest.mark.skipif(
    not (VENV_PATH / "bin" / "llm-annotate").exists(),
    reason="the submitter reads the config through the installed CLI",
)
def test_submit_pipeline_reports_its_config_overrides(
    tmp_path: Path,
) -> None:
    """Overrides travel in the environment, so the submitter prints them."""
    config_path = _write_pool_config(
        tmp_path, {"servers": 2, "min_servers": 1}
    )

    process = _run_submit(
        tmp_path,
        config_path,
        extra=["--set", "dataset.max_num_samples=50000"],
    )

    assert process.returncode == 0, process.stderr
    assert "dataset.max_num_samples=50000" in process.stdout


REQUIRES_CLI = pytest.mark.skipif(
    not (VENV_PATH / "bin" / "llm-annotate").exists(),
    reason="the submitter reads the config through the installed CLI",
)


@REQUIRES_CLI
def test_submit_pipeline_queues_a_cleanup_job(tmp_path: Path) -> None:
    """A pool step gets a job that cancels its array however it ended."""
    config_path = _write_pool_config(
        tmp_path, {"servers": 4, "min_servers": 2}
    )

    process = _run_submit(tmp_path, config_path)

    assert process.returncode == 0, process.stderr
    cleanup = [
        line
        for line in process.stderr.splitlines()
        if "--job-name=cancel-write" in line
    ]
    assert len(cleanup) == 1
    assert "--dependency=afterany:<job-2>" in cleanup[0]
    assert "--kill-on-invalid-dep=yes" in cleanup[0]
    assert "scancel" in cleanup[0] and "job-1" in cleanup[0]
    assert "cleanup: <job-3> cancels <job-1>" in process.stdout


@REQUIRES_CLI
def test_submit_pipeline_skips_the_cleanup_job_when_asked(
    tmp_path: Path,
) -> None:
    """CANCEL_SERVERS_ON_EXIT=0 keeps the servers, so nothing cancels them."""
    config_path = _write_pool_config(
        tmp_path, {"servers": 4, "min_servers": 2}
    )

    process = _run_submit(
        tmp_path, config_path, env={"CANCEL_SERVERS_ON_EXIT": "0"}
    )

    assert process.returncode == 0, process.stderr
    assert "cancel-write" not in process.stderr


@REQUIRES_CLI
def test_submit_pipeline_chains_resubmits(tmp_path: Path) -> None:
    """Each further attempt waits for the previous one to have failed."""
    config_path = _write_two_step_config(tmp_path)

    process = _run_submit(
        tmp_path, config_path, extra=["--max-resubmits", "1"]
    )

    assert process.returncode == 0, process.stderr
    arrays = [
        line
        for line in process.stderr.splitlines()
        if "--job-name=vllm-write" in line
    ]
    assert len(arrays) == 2
    assert "afternotok" not in arrays[0]
    assert "--dependency=afternotok:<job-2>" in arrays[1]
    assert "--kill-on-invalid-dep=yes" in arrays[1]
    assert "attempt 2 of 2" in process.stdout

    # The next step needs any one attempt of the previous one to have
    # succeeded, which Slurm writes with '?' between the alternatives.
    rate = [
        line
        for line in process.stderr.splitlines()
        if "--job-name=annotate-rate" in line
    ]
    assert "--dependency=afterok:<job-2>?afterok:<job-5>" in rate[0]
    assert "--dependency=afternotok:<job-7>" in rate[1]


@REQUIRES_CLI
def test_submit_pipeline_throttles_the_server_array(tmp_path: Path) -> None:
    """MAX_CONCURRENT_SERVERS becomes the %N of the array specification."""
    config_path = _write_pool_config(
        tmp_path, {"servers": 4, "min_servers": 2}
    )

    plain = _run_submit(tmp_path, config_path)
    throttled = _run_submit(
        tmp_path, config_path, env={"MAX_CONCURRENT_SERVERS": "2"}
    )

    assert plain.returncode == 0, plain.stderr
    assert "--array=1-4 " in plain.stderr
    assert throttled.returncode == 0, throttled.stderr
    assert "--array=1-4%2 " in throttled.stderr


@REQUIRES_CLI
def test_submit_pipeline_rejects_a_throttle_below_min_servers(
    tmp_path: Path,
) -> None:
    """A throttle the client's threshold cannot survive fails at submit."""
    config_path = _write_pool_config(
        tmp_path, {"servers": 4, "min_servers": 2}
    )

    process = _run_submit(
        tmp_path, config_path, env={"MAX_CONCURRENT_SERVERS": "1"}
    )

    assert process.returncode == 1
    assert "MAX_CONCURRENT_SERVERS=1" in process.stderr
    assert "never reached" in process.stderr


@REQUIRES_CLI
def test_submit_pipeline_downloads_the_model_first(tmp_path: Path) -> None:
    """MODEL_DOWNLOAD=1 puts one fetch job in front of the whole chain."""
    config_path = _write_pool_config(
        tmp_path, {"servers": 4, "min_servers": 2}
    )

    process = _run_submit(tmp_path, config_path, env={"MODEL_DOWNLOAD": "1"})

    assert process.returncode == 0, process.stderr
    download = [
        line
        for line in process.stderr.splitlines()
        if "--job-name=download-Qwen3-8B" in line
    ]
    assert len(download) == 1
    assert "vllm_download_model" in download[0]
    assert "Qwen/Qwen3-8B" in download[0]
    array = next(
        line
        for line in process.stderr.splitlines()
        if "--job-name=vllm-write" in line
    )
    assert "--dependency=afterok:<job-1>" in array


@REQUIRES_CLI
def test_submit_pipeline_leaves_a_local_model_alone(tmp_path: Path) -> None:
    """A model that is a directory on this machine is not downloaded."""
    local_model = tmp_path / "my-model"
    local_model.mkdir()
    config_path = _write_pool_config(
        tmp_path, {"servers": 2, "min_servers": 1}, model=str(local_model)
    )

    process = _run_submit(tmp_path, config_path, env={"MODEL_DOWNLOAD": "1"})

    assert process.returncode == 0, process.stderr
    assert "--job-name=download-" not in process.stderr
    assert "is a local directory" in process.stdout


@REQUIRES_CLI
def test_submit_pipeline_survives_an_awkward_step_name(
    tmp_path: Path,
) -> None:
    """A step name with a space and a quote reaches sbatch unbroken."""
    config_path = _write_pool_config(
        tmp_path,
        {"servers": 2, "min_servers": 1},
        name='wri te"x',
        model="org/a model",
    )

    process = _run_submit(tmp_path, config_path)

    assert process.returncode == 0, process.stderr
    assert "Step 1 'wri te\"x'" in process.stdout
    assert "serving org/a model" in process.stdout
    assert '--job-name=annotate-wri\\ te\\"x' in process.stderr


def test_annotate_logs_the_thread_limit(tmp_path: Path) -> None:
    """The client names the process limit its threads are drawn from."""
    pool_dir = tmp_path / "pool"
    _publish(pool_dir, 1)

    process, _ = _run_annotate(tmp_path, pool_dir, MIN_SERVERS="1")

    assert process.returncode == 0, process.stderr
    assert "Thread limit (ulimit -u):" in process.stdout


def _run_server(
    tmp_path: Path, **overrides: str
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run `slurm/vllm_server.sh` against a stubbed `vllm`.

    The stub records every invocation in a file and exits with the message the
    test gave it, so the script's retry decision can be driven without a GPU.

    Args:
        tmp_path: Directory for the fake binaries, the pool and the record.
        **overrides: Environment variables set on top of the defaults.

    Returns:
        The finished process and one entry per `vllm` invocation.
    """
    bin_dir = _fake_bin(tmp_path)
    calls_file = tmp_path / "vllm_calls.txt"

    annotate = bin_dir / "llm-annotate"
    annotate.write_text(
        "#!/bin/bash\n"
        'if [[ -n "${ANNOTATE_FAILS:-}" ]]; then\n'
        '  echo "error: cfg.yaml: boom" >&2\n'
        "  exit 2\n"
        "fi\n"
        'printf "%s\\n" Qwen/Qwen3-8B\n',
        encoding="utf-8",
    )
    annotate.chmod(0o755)

    vllm = bin_dir / "vllm"
    vllm.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$*" >> "$VLLM_CALLS"\n'
        'printf "%s\\n" "$VLLM_MESSAGE"\n'
        "exit 1\n",
        encoding="utf-8",
    )
    vllm.chmod(0o755)

    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REPO_ROOT": str(REPO_ROOT),
        "CLUSTER_ENV": str(tmp_path / "absent.env"),
        "LOG_DIR": str(tmp_path / "logs"),
        "VENV_PATH": str(tmp_path / "absent-venv"),
        "ANNOTATE_CONFIG": str(tmp_path / "pipeline.yaml"),
        "STEP_NAME": "write",
        "POOL_DIR": str(tmp_path / "pool"),
        "VLLM_PORT": "39117",
        "READY_TIMEOUT": "1",
        "PORT_RETRIES": "2",
        "VLLM_CALLS": str(calls_file),
        "VLLM_MESSAGE": "OSError: [Errno 98] Address already in use",
        "HOME": os.environ.get("HOME", str(tmp_path)),
    }
    env.update(overrides)

    process = subprocess.run(
        ["/bin/bash", str(SLURM_DIR / "vllm_server.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    calls = (
        calls_file.read_text(encoding="utf-8").splitlines()
        if calls_file.exists()
        else []
    )
    return process, calls


def test_server_retries_on_a_taken_port(tmp_path: Path) -> None:
    """A bind error before /health moves the server to the next port."""
    process, calls = _run_server(tmp_path)

    assert process.returncode == 1
    assert "Server never became ready" in process.stderr
    assert "retrying on the next free port" in process.stdout
    # PORT_RETRIES retries on top of the first try.
    assert len(calls) == 3
    ports = [call.split("--port ")[1] for call in calls]
    assert len(set(ports)) == 3


def test_server_does_not_retry_on_another_failure(tmp_path: Path) -> None:
    """An early exit that is not a bind error fails the task at once."""
    process, calls = _run_server(
        tmp_path, VLLM_MESSAGE="ValueError: unsupported quantization"
    )

    assert process.returncode == 1
    assert "retrying on the next free port" not in process.stdout
    assert len(calls) == 1


def test_server_stops_when_the_serve_args_fail(tmp_path: Path) -> None:
    """A config that does not load ends the job before vLLM is started."""
    process, calls = _run_server(tmp_path, ANNOTATE_FAILS="1")

    assert process.returncode != 0
    assert "error: cfg.yaml: boom" in process.stderr
    assert "Could not read serving arguments" in process.stderr
    assert calls == []


def test_server_publishes_the_configured_host(tmp_path: Path) -> None:
    """SERVER_HOST_CMD decides the address the pool file would carry."""
    process, _ = _run_server(tmp_path, SERVER_HOST_CMD="echo node42")

    assert "Serving on http://node42:" in process.stdout
