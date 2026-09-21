"""Every pipeline config in the documentation has to load."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_annotator.config import load_pipeline_config


REPO_ROOT = Path(__file__).parent.parent

DOC_FILES = [
    REPO_ROOT / "README.md",
    # Included verbatim into the docs site as docs/slurm.md.
    REPO_ROOT / "slurm" / "README.md",
    *sorted(REPO_ROOT.glob("docs/**/*.md")),
]

FENCE_START = re.compile(r"^```+\s*yaml\b")
FENCE_END = re.compile(r"^```+\s*$")

FILE_KEYS = (
    "prompt_file",
    "system_prompt_file",
    "output_schema_file",
    "prompts",
)


def yaml_blocks() -> list[tuple[Path, int, str]]:
    """Collect the fenced yaml blocks of the documentation.

    Returns:
        The file, the line the fence opens on and the block's text, for every
        block that looks like a whole pipeline config.
    """
    blocks = []
    for path in DOC_FILES:
        lines = path.read_text(encoding="utf-8").splitlines()
        index = 0
        while index < len(lines):
            if not FENCE_START.match(lines[index]):
                index += 1
                continue
            opened_at = index + 1
            index += 1
            body: list[str] = []
            while index < len(lines) and not FENCE_END.match(lines[index]):
                body.append(lines[index])
                index += 1
            text = "\n".join(body)
            try:
                data = yaml.safe_load(text)
            except yaml.YAMLError:
                data = None
            if isinstance(data, dict) and {"output_dir", "steps"} <= set(data):
                blocks.append((path, opened_at, text))
    return blocks


def _write_placeholders(data: dict[str, Any], root: Path) -> None:
    """Create the files that a config block refers to.

    Args:
        data: The decoded config block.
        root: Directory the config file was written to.
    """
    for step in data.get("steps") or []:
        if not isinstance(step, dict):
            continue
        for key in FILE_KEYS:
            value = step.get(key)
            if not isinstance(value, str):
                continue
            pfout = root / value
            pfout.parent.mkdir(parents=True, exist_ok=True)
            pfout.write_text(
                "{}" if pfout.suffix == ".json" else "Prompt {text}",
                encoding="utf-8",
            )


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(text, id=f"{path.relative_to(REPO_ROOT)}:{line}")
        for path, line, text in yaml_blocks()
    ],
)
def test_documented_config_loads(tmp_path: Path, text: str) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(text, encoding="utf-8")
    _write_placeholders(yaml.safe_load(text), tmp_path)

    load_pipeline_config(config_path)


def test_every_doc_file_is_searched() -> None:
    # A file that moved or was renamed would otherwise silently drop out.
    assert len(DOC_FILES) >= 7
    assert yaml_blocks()
