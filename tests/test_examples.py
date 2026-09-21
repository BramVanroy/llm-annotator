"""Smoke-test every example script and case study.

Two levels of checking are applied:

1. **Syntax check** (all ``*.py`` files under ``examples/`` and
   ``case-studies/``): the source is parsed with :func:`ast.parse`. This is
   instantaneous and never requires any third-party package to be installed.

2. **Import check**: each script is loaded as a module with :mod:`importlib`.
   This validates that all top-level imports resolve correctly. ``main()`` is
   never called, so no GPU, model, or network access is needed.

Every pipeline config under ``examples/`` and ``case-studies/`` is validated
as well, so neither can keep a setting the config layer rejects.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

from llm_annotator.config import load_config_file, load_pipeline_config


REPO_ROOT = Path(__file__).parent.parent
EXAMPLE_ROOTS = {
    "examples": REPO_ROOT / "examples",
    "case-studies": REPO_ROOT / "case-studies",
}


def _test_id(path: Path) -> str:
    """Readable test id, as ``<root name>/<path relative to that root>``."""
    for root_name, root_dir in EXAMPLE_ROOTS.items():
        if root_dir in path.parents:
            return f"{root_name}/{path.relative_to(root_dir).as_posix()}"
    raise ValueError(f"{path} is not under a known example root")


ALL_EXAMPLE_SCRIPTS = sorted(
    path
    for root_dir in EXAMPLE_ROOTS.values()
    for path in root_dir.rglob("*.py")
)


def _is_pipeline_config(path: Path) -> bool:
    """Whether *path* is a pipeline config rather than a schema or a prompt."""
    try:
        data = load_config_file(path)
    except Exception:
        return False
    return isinstance(data, dict) and "steps" in data


ALL_EXAMPLE_CONFIGS = sorted(
    path
    for root_dir in EXAMPLE_ROOTS.values()
    for pattern in ("*.yaml", "*.yml", "*.json")
    for path in root_dir.rglob(pattern)
    if _is_pipeline_config(path)
)


@pytest.mark.parametrize(
    "script",
    ALL_EXAMPLE_SCRIPTS,
    ids=[_test_id(p) for p in ALL_EXAMPLE_SCRIPTS],
)
def test_example_syntax(script: Path) -> None:
    """Assert that *script* is valid Python (syntax check only)."""
    source = script.read_text(encoding="utf-8")
    ast.parse(source, filename=str(script))


@pytest.mark.parametrize(
    "script",
    ALL_EXAMPLE_SCRIPTS,
    ids=[_test_id(p) for p in ALL_EXAMPLE_SCRIPTS],
)
def test_example_imports(script: Path) -> None:
    """Import *script* as a module and assert it loads without raising.

    ``main()`` is never invoked, so no GPU or network access is required. The
    script's own directory is put on ``sys.path`` for the duration of the
    import, since some example scripts import a sibling module (for example
    ``propella.py`` importing ``propella_schema``) the way they would when run
    directly with ``python examples/<name>/<script>.py``, which puts the
    script directory on ``sys.path`` too.
    """
    module_name = f"_example_{script.stem}_{script.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None, (
        f"Could not create module spec for {script}"
    )
    module = importlib.util.module_from_spec(spec)
    script_dir = str(script.parent)
    sys.path.insert(0, script_dir)
    # Register so relative imports inside the script (if any) can resolve.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        sys.path.remove(script_dir)
        # Drop any sibling module the script imported, so a same-named module
        # in another example directory is not resolved from this one.
        for name in list(sys.modules):
            module_path = getattr(sys.modules[name], "__file__", None)
            if module_path and module_path.startswith(script_dir):
                del sys.modules[name]


@pytest.mark.parametrize(
    "config",
    ALL_EXAMPLE_CONFIGS,
    ids=[_test_id(p) for p in ALL_EXAMPLE_CONFIGS],
)
def test_example_config_validates(config: Path) -> None:
    """Assert that *config* passes the pipeline config validation."""
    load_pipeline_config(config)
