"""Checks that keep the prose documentation in step with the code."""

from __future__ import annotations

import ast
import importlib
import re
from dataclasses import dataclass
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parent.parent
SRC_ROOT = REPO_ROOT / "src"
TROUBLESHOOTING = REPO_ROOT / "docs" / "troubleshooting.md"

DOC_FILES = [
    REPO_ROOT / "README.md",
    *sorted(REPO_ROOT.glob("docs/**/*.md")),
]

# Put this directly above a fenced python block that is deliberately not a
# whole module, so that it is not parsed.
FRAGMENT_MARKER = "<!-- docs-test: fragment -->"

FENCE = re.compile(r"^(?P<ticks>```+)\s*(?P<language>[A-Za-z0-9_+-]*)")
PLACEHOLDER = re.compile(r"<[^>]*>")


@dataclass(frozen=True)
class Block:
    """One fenced code block of a documentation file."""

    path: Path
    line: int
    language: str
    text: str
    is_fragment: bool

    @property
    def id(self) -> str:
        """The block's `<file>:<line>` identifier."""
        return f"{self.path.relative_to(REPO_ROOT)}:{self.line}"


def fenced_blocks(language: str, paths: list[Path]) -> list[Block]:
    """Collect the fenced blocks of one language.

    Args:
        language: The language written after the opening fence.
        paths: The Markdown files to read.

    Returns:
        One block per fence, in file order.
    """
    blocks: list[Block] = []
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        index = 0
        while index < len(lines):
            match = FENCE.match(lines[index])
            if not match or match.group("language") != language:
                index += 1
                continue
            preceding = [line for line in lines[:index] if line.strip()]
            opened_at = index + 1
            closing = match.group("ticks")
            index += 1
            body: list[str] = []
            while index < len(lines) and lines[index].rstrip() != closing:
                body.append(lines[index])
                index += 1
            index += 1
            blocks.append(
                Block(
                    path=path,
                    line=opened_at,
                    language=language,
                    text="\n".join(body),
                    is_fragment=bool(preceding)
                    and preceding[-1].strip() == FRAGMENT_MARKER,
                )
            )
    return blocks


def parsed_python_blocks() -> list[tuple[Block, ast.Module]]:
    """Parse the python blocks that are whole modules.

    Returns:
        Every block that parses, with its syntax tree.
    """
    parsed = []
    for block in fenced_blocks("python", DOC_FILES):
        if block.is_fragment:
            continue
        try:
            parsed.append((block, ast.parse(block.text)))
        except SyntaxError:
            continue
    return parsed


def imported_names(tree: ast.Module) -> list[tuple[str, str]]:
    """Find what a block imports from the package.

    Args:
        tree: The block's syntax tree.

    Returns:
        One `(module, name)` pair per imported name.
    """
    return [
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.split(".")[0] == "llm_annotator"
        for alias in node.names
    ]


def normalise(text: str) -> str:
    """Collapse every run of whitespace into one space.

    Args:
        text: The text to normalise.

    Returns:
        The text with single spaces and no leading or trailing whitespace.
    """
    return " ".join(text.split())


def source_messages() -> str:
    """Render every string literal of `src/` as one searchable haystack.

    An f-string becomes its literal parts with `<>` in place of each
    interpolation, so that a message quoted in the documentation with
    `<placeholder>` in the same spots matches it piece by piece.

    Returns:
        The normalised literals, one per line.
    """
    messages = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                messages.append(node.value)
            elif isinstance(node, ast.JoinedStr):
                messages.append(
                    "".join(
                        part.value
                        if isinstance(part, ast.Constant)
                        and isinstance(part.value, str)
                        else "<>"
                        for part in node.values
                    )
                )
    return "\n".join(normalise(message) for message in messages)


PYTHON_BLOCKS = fenced_blocks("python", DOC_FILES)
PARSED_BLOCKS = parsed_python_blocks()
IMPORTING_BLOCKS = [
    (block, names)
    for block, tree in PARSED_BLOCKS
    if (names := imported_names(tree))
]
QUOTED_MESSAGES = fenced_blocks("text", [TROUBLESHOOTING])


@pytest.mark.parametrize(
    "block",
    [pytest.param(block, id=block.id) for block in PYTHON_BLOCKS],
)
def test_documented_python_parses(block: Block) -> None:
    if block.is_fragment:
        return
    try:
        ast.parse(block.text)
    except SyntaxError as exc:
        pytest.fail(
            f"{block.id} does not parse: {exc}. Write it as a whole module,"
            f" or put '{FRAGMENT_MARKER}' on the line directly above the"
            " fence when the block is deliberately a fragment."
        )


@pytest.mark.parametrize(
    ("block", "names"),
    [
        pytest.param(block, names, id=block.id)
        for block, names in IMPORTING_BLOCKS
    ],
)
def test_documented_imports_resolve(
    block: Block, names: list[tuple[str, str]]
) -> None:
    for module_name, name in names:
        module = importlib.import_module(module_name)
        assert hasattr(module, name), (
            f"{block.id} imports '{name}' from '{module_name}', which does"
            " not define it."
        )


@pytest.mark.parametrize(
    "block",
    [pytest.param(block, id=block.id) for block in QUOTED_MESSAGES],
)
def test_quoted_error_text_exists(block: Block) -> None:
    haystack = source_messages()
    for fragment in PLACEHOLDER.split(normalise(block.text)):
        fragment = fragment.strip()
        if not fragment:
            continue
        assert fragment in haystack, (
            f"{block.id} quotes text that no string in src/ holds:"
            f" {fragment!r}. Copy the message from the source, and write"
            " <placeholder> where it interpolates a value."
        )


def test_every_doc_file_is_searched() -> None:
    # A page that moved or was renamed would otherwise silently drop out.
    assert len(DOC_FILES) >= 9
    assert len(PYTHON_BLOCKS) >= 10
    assert len(QUOTED_MESSAGES) >= 40
