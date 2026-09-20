"""Tests for the SLURM job scripts under `slurm/`.

The scripts are driven with a fake `llm-annotate` on `PATH` that records the
arguments it was called with, so the wait loop and the flags it builds can be
checked without a scheduler, a GPU or a model.
"""

from __future__ import annotations

import os
import subprocess
import sys
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
    """Create a bin directory with a recording `llm-annotate` and a fast sleep.

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


@pytest.mark.skipif(
    not (VENV_PATH / "bin" / "llm-annotate").exists(),
    reason="the submitter reads the config through the installed CLI",
)
def test_submit_pipeline_exports_min_servers(tmp_path: Path) -> None:
    """The submitter hands the step's own readiness threshold to its client."""
    config_path = _write_pool_config(
        tmp_path, {"servers": 4, "min_servers": 2}
    )

    process = subprocess.run(
        [
            "/bin/bash",
            str(SLURM_DIR / "submit_pipeline.sh"),
            "--dry-run",
            str(config_path),
        ],
        env={
            "PATH": os.environ["PATH"],
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

    assert process.returncode == 0, process.stderr
    assert "client starts at 2 ready server(s)" in process.stdout
    assert "NUM_SERVERS=4,MIN_SERVERS=2" in process.stderr
