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
import sys
from pathlib import Path

import pytest


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
ALL_EXAMPLE_SCRIPTS = sorted(EXAMPLES_DIR.rglob("*.py"))
ANNOTATOR_EXAMPLE_SCRIPTS = [p for p in ALL_EXAMPLE_SCRIPTS]


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
    ANNOTATOR_EXAMPLE_SCRIPTS,
    ids=[p.relative_to(EXAMPLES_DIR).as_posix() for p in ANNOTATOR_EXAMPLE_SCRIPTS],
)
def test_example_imports(script: Path) -> None:
    """Import *script* as a module and assert it loads without raising.

    ``main()`` is never invoked, so no GPU or network access is required.
    """
    module_name = f"_example_{script.stem}_{script.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None, f"Could not create module spec for {script}"
    module = importlib.util.module_from_spec(spec)
    # Register so relative imports inside the script (if any) can resolve.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    finally:
        sys.modules.pop(module_name, None)
