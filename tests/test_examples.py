"""Smoke-test every example script.

Two levels of checking are applied:

1. **Syntax check** (all ``*.py`` files under ``examples/``): the source is
   parsed with :func:`ast.parse`.  This is instantaneous and never requires
   any third-party package to be installed.

2. **Import check** (annotator example scripts only: each script is loaded as a module with
   :mod:`importlib`.  This validates that all top-level imports from
   ``llm_annotator`` resolve correctly.  ``main()`` is never called, so no
   GPU, model, or network access is needed.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
ALL_EXAMPLE_SCRIPTS = sorted(EXAMPLES_DIR.rglob("*.py"))


def test_prepare_seed_parses_json_personas(tmp_path: Path) -> None:
    """Grouped JSON persona files should be accepted and selected by key."""
    personas_path = tmp_path / "personas_nl.json"
    personas_path.write_text(
        json.dumps(
            {
                "social": [
                    {"code": "alpha", "description": "A"},
                    {"code": "beta", "description": "B"},
                ],
                "professional": [
                    {"code": "gamma", "description": "C"},
                    {"code": "delta", "description": "D"},
                ],
            }
        ),
        encoding="utf-8",
    )

    script_path = EXAMPLES_DIR / "gpt-nl-e" / "_shared" / "prepare_seed.py"
    spec = importlib.util.spec_from_file_location(
        "prepare_seed_test_module", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["prepare_seed_test_module"] = module
    spec.loader.exec_module(module)

    assert module.parse_personas(personas_path, "social") == [
        "alpha",
        "beta",
    ]
    assert module.parse_personas(personas_path, "professional") == [
        "gamma",
        "delta",
    ]


def test_prepare_seed_normalizes_and_splits_sentences() -> None:
    """Normalization should clean common unicode junk and sentence splitting should work."""
    script_path = EXAMPLES_DIR / "gpt-nl-e" / "_shared" / "prepare_seed.py"
    spec = importlib.util.spec_from_file_location(
        "prepare_seed_sentence_test_module", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["prepare_seed_sentence_test_module"] = module
    spec.loader.exec_module(module)

    assert module.normalize("A\u200bB") == "ab"
    assert module.split_sentences("Hallo wereld. Dit is tweede zin!") == [
        "Hallo wereld.",
        "Dit is tweede zin!",
    ]


@pytest.mark.parametrize(
    "script",
    ALL_EXAMPLE_SCRIPTS,
    ids=[p.relative_to(EXAMPLES_DIR).as_posix() for p in ALL_EXAMPLE_SCRIPTS],
)
def test_example_syntax(script: Path) -> None:
    """Assert that *script* is valid Python (syntax check only)."""
    source = script.read_text(encoding="utf-8")
    ast.parse(source, filename=str(script))


@pytest.mark.parametrize(
    "script",
    ALL_EXAMPLE_SCRIPTS,
    ids=[p.relative_to(EXAMPLES_DIR).as_posix() for p in ALL_EXAMPLE_SCRIPTS],
)
def test_example_imports(script: Path) -> None:
    """Import *script* as a module and assert it loads without raising.

    ``main()`` is never invoked, so no GPU or network access is required.
    """
    module_name = f"_example_{script.stem}_{script.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None, (
        f"Could not create module spec for {script}"
    )
    module = importlib.util.module_from_spec(spec)
    # Register so relative imports inside the script (if any) can resolve.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
