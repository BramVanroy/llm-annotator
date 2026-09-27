"""Checks that the FlashInfer wheel pins stay compatible with each other.

FlashInfer refuses to import when `flashinfer-python`, `flashinfer-cubin`
and `flashinfer-jit-cache` do not carry the same version, so the three pins
have to move together whenever vLLM (and with it its `flashinfer-python`
requirement) is bumped.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent

FLASHINFER_PACKAGES = (
    "flashinfer-python",
    "flashinfer-cubin",
    "flashinfer-jit-cache",
)


def _strip_local_label(version: str) -> str:
    """Drop a local version label such as ``+cu130``.

    Example:
        >>> _strip_local_label("0.6.18+cu130")
        '0.6.18'
    """
    return version.split("+", 1)[0]


def _locked_flashinfer_versions() -> dict[str, str]:
    with open(REPO_ROOT / "uv.lock", "rb") as f:
        lock = tomllib.load(f)
    packages = {
        package["name"]: package["version"]
        for package in lock["package"]
        if "version" in package
    }
    return {name: packages[name] for name in FLASHINFER_PACKAGES}


def test_locked_flashinfer_versions_match():
    """The three locked FlashInfer packages agree on one version."""
    versions = _locked_flashinfer_versions()
    stripped = {name: _strip_local_label(v) for name, v in versions.items()}
    distinct = set(stripped.values())
    assert len(distinct) == 1, (
        "flashinfer-python, flashinfer-cubin and flashinfer-jit-cache "
        f"must share one version in uv.lock, got {versions}"
    )


def test_vllm_kernels_group_pins_match_locked_flashinfer_python():
    """The `vllm-kernels` group pins the version vLLM's own lock resolved."""
    locked_python_version = _strip_local_label(
        _locked_flashinfer_versions()["flashinfer-python"]
    )
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    group = pyproject["dependency-groups"]["vllm-kernels"]
    pins = {
        name: requirement.split("==", 1)[1]
        for name in ("flashinfer-cubin", "flashinfer-jit-cache")
        for requirement in group
        if requirement.startswith(f"{name}==")
    }
    assert set(pins) == {"flashinfer-cubin", "flashinfer-jit-cache"}, (
        f"expected an == pin for both kernel wheels in the vllm-kernels "
        f"group, got {group}"
    )
    for name, pin in pins.items():
        assert _strip_local_label(pin) == locked_python_version, (
            f"{name}=={pin} in pyproject.toml no longer matches the locked "
            f"flashinfer-python=={locked_python_version}"
        )
