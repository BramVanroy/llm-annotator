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
    bin_dir.mkdir()

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


def _write_pool_config(tmp_path: Path, pool: dict[str, int]) -> Path:
    """Write a one-step pooled-vLLM config and return its path."""
    config_path = tmp_path / "pipeline.yaml"
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
                            "pool": pool,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _run_submit(
    tmp_path: Path, config_path: Path, dry_run: bool = True
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

    Returns:
        The finished process.
    """
    bin_dir = _fake_bin(tmp_path)
    command = ["/bin/bash", str(SLURM_DIR / "submit_pipeline.sh")]
    if dry_run:
        command.append("--dry-run")
    command.append(str(config_path))

    return subprocess.run(
        command,
        env={
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "SBATCH_CMD": str(bin_dir / "sbatch"),
            "REPO_ROOT": str(REPO_ROOT),
            "CLUSTER_ENV": str(tmp_path / "absent.env"),
            "LOG_DIR": str(tmp_path / "logs"),
            "VENV_PATH": str(VENV_PATH),
            "HOME": os.environ.get("HOME", str(tmp_path)),
        },
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
        "--dependency=after:<job-id>_1?after:<job-id>_2?"
        "after:<job-id>_3?after:<job-id>_4" in process.stderr
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
