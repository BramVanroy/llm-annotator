# Contributing to llm-annotator

The short version of the rules on this page, written for a coding agent and
useful for a first contribution too, is in [AGENTS.md](AGENTS.md).

## Development setup

Everything runs through [uv](https://docs.astral.sh/uv/). The project needs
Python 3.12, 3.13 or 3.14.

```bash
git clone https://github.com/BramVanroy/llm-annotator.git
cd llm-annotator
uv sync --dev
pre-commit install
pre-commit install --hook-type pre-push
```

`uv sync --dev` installs the package with every provider extra (`vllm`,
`openai`, `anthropic`) plus the development and documentation tools. The two
`pre-commit install` calls are separate because the checks run at two different
stages, see [Pre-commit hooks](#pre-commit-hooks).

## Make targets

| Target | What it runs |
| --- | --- |
| `make style` | `ruff check --fix` and `ruff format` over `src/`, `tests/`, `examples/`, `case-studies/`, `scripts/` |
| `make quality` | `interrogate` (docstring coverage) plus `ruff check` and `ruff format --check` |
| `make typecheck` | `mypy` over `src/`, `tests/` and `scripts/` |
| `make test` | the fast suite, `pytest -m "not slow"`, with coverage |
| `make test-slow` | `pytest -m "slow"`: loads real models through vLLM |
| `make test-integration` | `pytest -m "integration"`: network and Hugging Face Hub |
| `make test-all` | every marker |
| `make test-matrix` | the fast suite once per CI Python version |
| `make serve-docs` | `mkdocs serve` with live reload |
| `make build-docs` | `mkdocs build --strict`, what the CI docs job runs |

`make test-fast` is an alias of `make test` that CI calls by that name.

## Tests

Markers are strict (`--strict-markers` in `pyproject.toml`), and there are two:

- `slow`: loads a real model. Seven tests carry it.
- `integration`: talks to the network or the Hugging Face Hub, and needs an
  authenticated account.

A single test or file:

```bash
uv run pytest tests/test_annotator.py::test_prepare_data_writes_a_record
uv run pytest tests/no_network_clients/ -m "not slow"
```

`pyproject.toml` puts `--doctest-modules` in `addopts` and lists both `tests`
and `src` in `testpaths`, so every docstring example in `src/` is executed as a
doctest. An example that needs a GPU, a model or the network has to carry
`# doctest: +SKIP`.

Provider SDKs are faked in `tests/no_network_clients/` by injecting modules
into `sys.modules` (the `fake_openai_module` and `fake_anthropic_module`
fixtures in `conftest.py`). The `block_network` fixture makes
`socket.socket.connect` fail, so a test that reaches the network fails rather
than hangs. Hub cleanup in `conftest.py` is opt-in through
`LLM_ANNOTATOR_ALLOW_NETWORK_TESTS=1`; a default run stays offline.

### The Python version matrix

`make test` only exercises the interpreter in `.venv`. Some breakage is
version-specific and therefore invisible there: `@dataclass(slots=True)` plus a
zero-argument `super()` raises `TypeError` on 3.12 and works from 3.13 on.
`make test-matrix` runs the fast suite against every interpreter the CI matrix
covers. It reads the version list out of `.github/workflows/ci.yml`, so adding a
version to CI adds it here, and it creates one venv per version under
`.venvs/py<X.Y>` (gitignored, hardlinked from uv's cache).

To run pytest in one of those venvs afterwards without uv rebuilding the
environment:

```bash
UV_PROJECT_ENVIRONMENT=.venvs/py3.12 uv run --no-sync --python 3.12 pytest -m "not slow and not integration"
```

`--no-sync` is what keeps uv from re-resolving, and `UV_PROJECT_ENVIRONMENT`
belongs on that one command line rather than exported into the shell, where it
would also redirect `make quality`, `make typecheck` and `make build-docs`.

### The slow suite

The seven `slow` tests load `HuggingFaceTB/SmolLM2-135M-Instruct` through vLLM
and need one GPU. They pass in a single process on an H100 with vLLM 0.29.0.
The vLLM engine fixtures are module-scoped rather than session-scoped, so an
engine is released before the next module builds its own.

```bash
make test-slow
```

A machine without a GPU skips them. A machine with a GPU on which the engine
fails to start reports a failure rather than a skip, so a broken engine is not
mistaken for a missing one.

Serving needs the prebuilt FlashInfer kernels from the `vllm-kernels`
dependency group:

```bash
uv sync --extra vllm --group vllm-kernels
```

Without them vLLM JIT-compiles its kernels at start-up, which needs `nvcc` on
the node and races between servers that share `~/.cache/flashinfer`. The wheels
are about 2.5 GB, which is why they sit in a group rather than in the `vllm`
extra that CI installs. `docs/provider-info.md` has the details.

## Code style

- Ruff's `line-length` is 79. The formatter wraps code at that width, so
  `make style` reflows anything longer that it can reflow.
- `E501` is in `[tool.ruff.lint] ignore`, so a long line is not a lint error.
  A string, a URL or a comment that the formatter cannot break is therefore
  accepted as it is. Do not reformat such a line by hand to chase the number.
- isort has `lines-after-imports = 2` and `known-first-party = ["llm_annotator"]`.
  Imports are top level; a provider SDK is the exception and is imported inside
  `__init__` or inside the method that uses it, so the package imports without
  the optional extras installed.
- Google-style docstrings everywhere in `src/`. `interrogate` fails
  `make quality` if a public object has none. Tests, examples and docs are
  excluded from that check.
- mypy runs with `check_untyped_defs`, `warn_return_any`, `warn_unused_ignores`
  and `no_implicit_optional`.
- Logging goes through `llm_annotator.logging_utils.get_logger(...)`, never
  `print`.

### Docstrings

mkdocstrings renders docstrings as Markdown, so a Sphinx role survives as
literal `:class:` text on the site. Cross-reference with the mkdocstrings form
``[`Name`][full.dotted.path]`` instead. Two pre-commit hooks enforce that and
the related rule that an attribute needs a real docstring rather than a `#:`
comment.

```python
def my_function(param1: str, param2: int = 10) -> bool:
    """Return whether param1 is longer than param2.

    Args:
        param1: The text to measure.
        param2: The length to compare against.

    Returns:
        True when param1 is longer than param2.

    Examples:
        >>> my_function("test", 5)
        False
    """
    return len(param1) > param2
```

## Documentation

The site is built with [MkDocs](https://www.mkdocs.org/) and the Material
theme. `mkdocs.yml` holds the theme, the extensions and the `nav` tree.

| Path | Holds |
| --- | --- |
| `docs/index.md` | home page: what the library is, install, quickstart, where to go next |
| `docs/choosing-a-provider.md`, `docs/pipeline.md`, `docs/python-api.md`, `docs/growing-a-run.md`, `docs/troubleshooting.md`, `docs/provider-info.md`, `docs/migration.md` | the prose guides |
| `docs/slurm.md` | one line that includes `slurm/README.md` verbatim |
| `docs/api/*.md` | one stub per module, each holding only `::: llm_annotator.<module>` |
| `docs/hooks.py` | build hook: renders doctest examples as plain Python and points API source links at the release tag |
| `docs/overrides/` | mkdocstrings template overrides |

Two rules when a page is added or a public symbol appears:

- A new page has to be listed in `mkdocs.yml`'s `nav`, or
  `mkdocs build --strict` fails on the omitted file.
- A new public symbol belongs in `src/llm_annotator/__init__.py` (with an
  explicit `as` alias and an `__all__` entry) and on the matching `docs/api/`
  stub page.

The install section and the quickstart examples live in `README.md` only.
`docs/index.md` includes them with `pymdownx.snippets` section markers
(`<!-- --8<-- [start:name] -->`), the same mechanism `docs/slurm.md` uses for
`slurm/README.md`. A file that is rendered in both places carries no relative
Markdown links, since no relative path resolves correctly from the repository
root and from inside `docs/` at the same time. Put those links in the page that
includes the snippet.

Build and preview:

```bash
make build-docs   # mkdocs build --strict
make serve-docs   # http://127.0.0.1:8000
```

`make serve-docs-versioned` is only needed to exercise the mike version
selector; it writes to a throwaway local branch.

Two test modules keep the documentation honest, and both run in the fast suite:

- `tests/test_docs_configs.py` loads every fenced `yaml` block of `README.md`,
  `docs/**/*.md` and `slurm/README.md` that looks like a whole pipeline config.
- `tests/test_docs.py` parses every fenced `python` block, resolves the names
  each one imports from `llm_annotator`, and checks that every error text
  quoted in `docs/troubleshooting.md` still occurs in `src/`.

## Pre-commit hooks

The commit-stage hooks run ruff lint, ruff format, mypy, a set of file hygiene
checks and two pygrep rules about docstrings.

The pre-push hook runs `make test-matrix` when the push touches a `.py` file.
It runs at a different stage, so it needs its own install line
(`pre-commit install --hook-type pre-push`) once per clone. `pre-commit run
--all-files`, which CI uses, skips it.

If a hook fails, fix what it reports, `git add` the fixes and commit again.

## Pull requests

1. Branch from `main`.
2. Keep the documentation in step with the change.
3. Run `make style`, `make quality`, `make typecheck` and `make test`.
4. Run `make build-docs` when a page or a docstring changed.
5. Open the pull request with a description of what changed and why.

CI runs pre-commit and `make typecheck`, `mkdocs build --strict`, and the fast
suite on 3.12, 3.13 and 3.14. The slow suite runs on pushes to `main` only.
A separate workflow checks the links in the Markdown files with
[lychee](https://lychee.cli.rs/).

## Releases

Documentation is published with [mike](https://github.com/jimporter/mike) from
`.github/workflows/docs.yml` when a release tag is pushed, and lands on
<https://bramvanroy.github.io/llm-annotator/>.

## Questions

Open an issue for a bug or a feature request, or start a discussion for
anything else.
