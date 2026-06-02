"""MkDocs hook: render doctest Examples sections as clean, syntax-highlighted Python.

Docstrings use ``>>>`` prompts so that ``pytest --doctest-modules`` can run them.
This hook post-processes the rendered HTML: it finds the plain-text blocks that
mkdocstrings generates for ``Examples::`` sections, strips the ``>>>`` / ``...``
prompts, then re-highlights the resulting code with Pygments as Python: giving
readers a clean, copyable, syntax-coloured block.

It also rewrites API source links so versioned docs point at the Git tag used
to build that release instead of always pointing at ``main``.

Doctests in source files are NOT modified: only the rendered HTML differs.
"""

from __future__ import annotations

import html as html_lib
import os
import re
from collections.abc import Mapping
from urllib.parse import quote

from pygments import highlight as pyg_highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer


_lexer = PythonLexer()
_formatter = HtmlFormatter(nowrap=True)
_SOURCE_LINK_PATTERN = re.compile(
    r'(?P<prefix><a\b[^>]*\bdata-source-link="github"[^>]*\bhref=")'
    r'(?P<source>[^"]+)'
    r'(?P<suffix>"[^>]*>)'
)
_PYCON_BLOCK_PATTERN = re.compile(
    r'<div class="language-pycon highlight"><pre[^>]*><span></span><code>'
    r"(?P<code>.*?)"
    r"</code></pre></div>",
    flags=re.DOTALL,
)
_TEXT_BLOCK_PATTERN = re.compile(
    r'<div class="language-text highlight"><pre[^>]*><span></span><code>'
    r"(?P<code>.*?)"
    r"</code></pre></div>",
    flags=re.DOTALL,
)


def _strip_doctest_prompts(code: str) -> str:
    """
    - Strip ``>>> `` and ``... `` prompts portions of examples in docstrings;
    - convert output lines to ``# comments``.

    Does smart handling of output in the examples so that it gets written to a comment in the documentation.
    """
    lines = code.split("\n")
    result: list[str] = []
    expect_output = False

    for line in lines:
        if line.startswith(">>> "):
            result.append(line[4:])
            expect_output = True
        elif line == ">>>":
            result.append("")
            expect_output = False
        elif line.startswith("... "):
            result.append(line[4:])
        elif line == "...":
            result.append("")
        elif expect_output and line != "":
            result.append("# " + line)
            expect_output = False
        else:
            result.append(line)
            if line.strip():
                expect_output = False

    return "\n".join(result).strip()


def _extract_code_text(raw_code: str) -> str:
    """Extract plain code text from rendered HTML code blocks."""
    without_tags = re.sub(r"<[^>]+>", "", raw_code)
    return html_lib.unescape(without_tags)


def _render_python_block(code: str) -> str:
    """Render code as a highlighted Python block."""
    inner = pyg_highlight(code, _lexer, _formatter)
    return (
        '<div class="language-python highlight">'
        "<pre><span></span><code>" + inner + "</code></pre></div>"
    )


def _rewrite_doctest_block(match: re.Match[str]) -> str:
    """Rewrite a doctest-like code block into plain Python-highlighted code."""
    plain_code = _extract_code_text(match.group("code"))
    if ">>>" not in plain_code:
        return match.group(0)

    clean = _strip_doctest_prompts(plain_code)
    return _render_python_block(clean)


def _get_docs_source_ref() -> str:
    """Return the Git ref that source links should point to."""
    git_ref = os.environ.get("DOCS_SOURCE_REF", "main").strip()
    return git_ref or "main"


def _get_repo_url(config: object | None) -> str | None:
    """Return the repository URL from MkDocs config when available."""
    if config is None:
        return None

    if isinstance(config, Mapping):
        repo_url = config.get("repo_url")
    else:
        repo_url = getattr(config, "repo_url", None)
        if repo_url is None:
            try:
                repo_url = config["repo_url"]  # type: ignore[index]
            except (KeyError, TypeError):
                repo_url = None

    if not repo_url:
        return None

    normalised_repo_url = str(repo_url).rstrip("/")
    if normalised_repo_url.endswith(".git"):
        normalised_repo_url = normalised_repo_url[:-4]
    return normalised_repo_url


def _build_github_source_url(
    repo_url: str,
    git_ref: str,
    source_target: str,
) -> str:
    """Build a GitHub source URL from a repo URL and relative source target."""
    filepath, hash_sep, fragment = html_lib.unescape(source_target).partition(
        "#"
    )
    quoted_ref = quote(git_ref, safe="/")
    quoted_filepath = quote(filepath, safe="/")
    url = f"{repo_url}/blob/{quoted_ref}/{quoted_filepath}"
    if hash_sep:
        url += f"#{fragment}"
    return url


def _rewrite_source_links(
    html: str,
    repo_url: str,
    git_ref: str,
) -> str:
    """Rewrite API source links to the requested GitHub ref."""

    def transform_link(match: re.Match[str]) -> str:
        source_target = match.group("source")
        source_url = _build_github_source_url(
            repo_url=repo_url,
            git_ref=git_ref,
            source_target=source_target,
        )
        return (
            f"{match.group('prefix')}{html_lib.escape(source_url, quote=True)}"
            f"{match.group('suffix')}"
        )

    return _SOURCE_LINK_PATTERN.sub(transform_link, html)


def on_page_content(html: str, **kwargs) -> str:
    """Post-process rendered page HTML for doctests and GitHub source links."""

    html = _PYCON_BLOCK_PATTERN.sub(_rewrite_doctest_block, html)
    html = _TEXT_BLOCK_PATTERN.sub(_rewrite_doctest_block, html)

    repo_url = _get_repo_url(kwargs.get("config"))
    if repo_url is None:
        return html

    return _rewrite_source_links(
        html=html,
        repo_url=repo_url,
        git_ref=_get_docs_source_ref(),
    )
