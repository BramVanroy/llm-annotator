"""MkDocs hook: render doctest Examples sections as clean, syntax-highlighted Python.

Docstrings use ``>>>`` prompts so that ``pytest --doctest-modules`` can run them.
This hook post-processes the rendered HTML: it finds the plain-text blocks that
mkdocstrings generates for ``Examples::`` sections, strips the ``>>>`` / ``...``
prompts, then re-highlights the resulting code with Pygments as Python: giving
readers a clean, copyable, syntax-coloured block.

It also rewrites API source links so versioned docs point at the Git tag used
to build that release instead of always pointing at ``main``. The templates in
``docs/overrides/python/material/`` emit a repo-relative path; the ref to
prepend comes from the ``DOCS_SOURCE_REF`` environment variable, which
``.github/workflows/docs.yml`` sets to the release tag.

Doctests in source files are NOT modified: only the rendered HTML differs.
"""

from __future__ import annotations

import html as html_lib
import os
import re
from urllib.parse import quote

from pygments import highlight as pyg_highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer


_lexer = PythonLexer()
_formatter = HtmlFormatter(nowrap=True)

# Anchors emitted by the custom mkdocstrings templates, carrying a path that is
# relative to the repository root.
_SOURCE_LINK_PATTERN = re.compile(
    r'(?P<prefix><a\b[^>]*\bdata-source-link="github"[^>]*\bhref=")'
    r'(?P<source>[^"]+)'
    r'(?P<suffix>"[^>]*>)'
)

# mkdocstrings renders an Examples section as either a pycon or a plain text
# block, depending on how the docstring was written.
_DOCTEST_BLOCK_PATTERN = re.compile(
    r'<div class="language-(?:pycon|text) highlight">'
    r"<pre[^>]*><span></span><code>"
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


def _rewrite_doctest_block(match: re.Match[str]) -> str:
    """Rewrite a doctest-like code block into plain Python-highlighted code."""
    plain_code = html_lib.unescape(re.sub(r"<[^>]+>", "", match.group("code")))
    if ">>>" not in plain_code:
        return match.group(0)

    inner = pyg_highlight(
        _strip_doctest_prompts(plain_code), _lexer, _formatter
    )
    return (
        '<div class="language-python highlight">'
        "<pre><span></span><code>" + inner + "</code></pre></div>"
    )


def _rewrite_source_links(html: str, repo_url: str) -> str:
    """Point API source links at the Git ref the docs were built from."""
    git_ref = os.environ.get("DOCS_SOURCE_REF", "").strip() or "main"

    def transform_link(match: re.Match[str]) -> str:
        filepath, hash_sep, fragment = html_lib.unescape(
            match.group("source")
        ).partition("#")
        url = (
            f"{repo_url}/blob/{quote(git_ref, safe='/')}"
            f"/{quote(filepath, safe='/')}"
        )
        if hash_sep:
            url += f"#{fragment}"
        return (
            f"{match.group('prefix')}{html_lib.escape(url, quote=True)}"
            f"{match.group('suffix')}"
        )

    return _SOURCE_LINK_PATTERN.sub(transform_link, html)


def on_page_content(html: str, page, config, files) -> str:
    """Post-process rendered page HTML for doctests and GitHub source links."""
    html = _DOCTEST_BLOCK_PATTERN.sub(_rewrite_doctest_block, html)

    repo_url = str(config.repo_url or "").rstrip("/").removesuffix(".git")
    if not repo_url:
        return html

    return _rewrite_source_links(html, repo_url)
