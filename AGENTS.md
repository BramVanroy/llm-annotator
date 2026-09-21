# AGENTS.md

Working rules for a coding agent (or a new contributor) in this repository. They
hold for every tool and every model. [CONTRIBUTING.md](CONTRIBUTING.md) has the
contributor workflow in more words, and the `docs/` directory is the user
documentation.

## Commands

Everything runs through `uv` (Python 3.12 to 3.14).

```sh
uv sync --dev          # all extras (vllm, openai, anthropic) plus the dev tools
make style             # ruff check --fix, then ruff format
make quality           # interrogate (docstring coverage), ruff check, ruff format --check
make typecheck         # mypy over src/ and tests/
make test              # fast suite: pytest -m "not slow", with coverage
make test-slow         # pytest -m "slow": needs one GPU
make test-integration  # pytest -m "integration": needs an authenticated Hugging Face account
make build-docs        # mkdocs build --strict, the same as the CI docs job
```

One test file or one test:

```sh
uv run pytest tests/test_annotator.py::test_name --no-cov
```

CI runs pre-commit (ruff, mypy), `make typecheck`, the fast suite on Python 3.12,
3.13 and 3.14, `mkdocs build --strict` and a Markdown link check. A pull request
merges when all of them pass.

## The virtual environments

This is the part that breaks most easily, so read it before you run a test.

- `.venv` is the default environment of `uv run`. `.venvs/py<X.Y>` holds one
  environment per CI Python version, built by `make test-matrix`.
- `uv run` syncs the environment that it uses. Combined with `--python` or with
  `UV_PROJECT_ENVIRONMENT`, a sync can rebuild an environment for another Python
  version, which removes vLLM from it and breaks every job that shares it.
- To run the fast suite against one CI version without a rebuild, use exactly
  this form, with the variable on the command line and never exported:

  ```sh
  UV_PROJECT_ENVIRONMENT=.venvs/py3.12 uv run --no-sync --python 3.12 pytest -m "not slow and not integration"
  ```

- Never `export UV_PROJECT_ENVIRONMENT`, and never run a `make` target or a bare
  `uv run` or `uv sync` with it set. `make quality`, `make typecheck` and
  `make build-docs` run without it.
- Repair a broken matrix environment with
  `UV_PROJECT_ENVIRONMENT=.venvs/py3.12 uv sync --locked --group dev --python 3.12`.
- The package is installed in editable mode from this checkout. In a second
  git worktree, prefix every Python command with `PYTHONPATH=<worktree>/src`,
  and check once that `python -c "import llm_annotator; print(llm_annotator.__file__)"`
  prints a path inside that worktree. Without the prefix the tests import the
  code of the main checkout.
- Do not edit `pyproject.toml` and do not add a dependency without the
  maintainer. If a change needs one, stop and say which line.

## Architecture in short

Three layers, each in few files under `src/llm_annotator/`:

- Clients (`clients/`): one class per provider (`OpenAIClient`, `ClaudeClient`,
  `VLLMOnlineClient`, `VLLMOfflineClient`), each with a frozen runtime options
  dataclass. The only contact with the layer above is
  `batch_generate(messages, options, gen_kwargs) -> list[Response]`.
- Annotator (`annotator.py`): loading, prompt templates, the JSONL progress
  files, resumption, Hub backup, the final dataset. `VLLMQueueAnnotator` runs
  the same workload over a pool of vLLM servers. `hub.py` restores a progress
  backup from the Hub.
- Config and CLI (`config.py`, `pool.py`, `pipeline.py`): pydantic models, the
  objects built from them, and `run_pipeline` plus the `llm-annotate` command.
  The dependency runs one way: `pipeline` -> `pool` -> `config`.

`slurm/` is a job submitter that uses four CLI flags of the package
(`--describe-steps`, `--serve-args`, `--hosts-file`, `--url-glob`) and nothing
else of its internals. Cluster names, partitions and core counts belong in the
cluster file (`slurm/cluster.env`), never in a script.

Contracts that other code relies on. A change that touches one of them needs a
test that exercises it:

- Client errors: a client returns exactly one `Response` per input, in input
  order. With `on_error` set to `"warn"` or `"ignore"` a failed request is a
  `Response` with `error` and `error_type`; with `"raise"` it is a
  `ProviderError`. A malformed request payload is a `ValueError` whatever
  `on_error` says.
- Resumption: `prepare_data` adds `idx_column`, results are appended to
  `<output_dir>/<task_prefix>progress_backup/*.jsonl`, and a restart skips the
  ids that those files hold (`_get_skip_idxs`). A test that resumes a run has to
  cover any change near this path, `drop_jsonl_rows`, `SelectionRecord` or
  `_check_reuse`.
- Artifact names: every file, directory and Hub branch of a task starts with its
  `task_prefix`, and `overwrite=True` removes only the artifacts of its own
  task, each by name. The final dataset in the root of `output_dir` is the one
  artifact that tasks share.
- Pipeline steps: `run_pipeline` removes a stale `output/` directory before the
  step runs, and a step whose recorded settings differ from the config is never
  loaded from its snapshot.
- Imports: provider SDKs (`openai`, `anthropic`, `vllm`) are imported inside
  methods, never at module level, so that the package imports without the
  extras.

## Code conventions

- Modern typing (`list[str]`, `X | None`). mypy runs with `check_untyped_defs`,
  `warn_return_any` and `warn_unused_ignores`.
- Signatures name every argument. No `**kwargs` pass-through and no options
  object to shorten a long signature. The long signatures of `prepare_data`,
  `run_annotation`, `annotate_dataset` and `generate_dataset` are intended.
- Never pass an `argparse.Namespace` on. Unpack it at the call site into
  explicit keyword arguments.
- Top-level imports only, with the provider SDK exception above.
- No one-line wrapper functions unless they are functionally needed.
- Google-style docstrings in `src/`: one short paragraph, then `Args`,
  `Returns`, `Raises`, and an example where behaviour is not obvious.
  `interrogate` fails the build on an undocumented public object. Every
  docstring example in `src/` runs as a doctest, so an example that needs a GPU,
  a model or the network carries `# doctest: +SKIP`.
- Comments explain a decision that the code cannot show. No comments that
  restate the next line, no banner comments, and no comments about what the code
  used to do ("no longer", "now", "previously").
- Logging goes through `llm_annotator.logging_utils.get_logger(...)`. No
  `print` in `src/`.
- Inside a `@dataclass(slots=True)`, call a base method as `Base.method(self)`.
  Zero-argument `super()` raises `TypeError` there on Python 3.12.
- The formatter wraps code at 79 columns with double quotes. `E501` is ignored,
  so a long string or comment is accepted.
- New public names go into `src/llm_annotator/__init__.py` (explicit `as` alias
  and `__all__`) only when a user of the documented API types them. Every module
  has a stub page under `docs/api/` that is listed in the `mkdocs.yml` nav.

## Writing rules

They hold for documentation, docstrings, comments, error messages, commit
messages and pull request text.

- Simple words and short sentences. State the mechanical fact. Do not sell,
  hedge or pad, and do not write filler ("Let's dive in").
- Do not personify code or data.
- Prefer a "that" clause over a chain of participles.
- No list of three items that exists for rhythm.
- No trailing "-ing" phrase for effect ("..., showing its importance").
- No "it is not X but Y" construction.
- Give an example where it clarifies behaviour.
- No emojis. No dashes as punctuation (use parentheses or a new sentence). `-`
  for list items. No bold list headers (`- Name: text`). Sentence case for
  headings. Straight quotes, `...` and `->`.
- A factual statement needs a source that the reader can check: a file, a
  function, a measurement with its conditions.

## Tests

- Markers are strict, and only `slow` and `integration` exist. A new test is
  fast and offline unless it cannot be.
- Provider SDKs are faked through `sys.modules` (`fake_openai_module`,
  `fake_anthropic_module` in `tests/conftest.py`, used by
  `tests/no_network_clients/`). The Hub is faked by monkeypatching the
  `huggingface_hub` functions that a module imports. Threads are tested with
  `threading.Event`, never with sleeps.
- Every behaviour change comes with a test of the new behaviour, and a bug fix
  comes with a test that fails without the fix.
- Coverage must not drop: not in total, and not in a module that the change
  touches. Compare against the coverage that CI reports for `main`. No
  `# pragma: no cover` to reach a number, and no test without an assertion.
- The slow suite needs one GPU. Its engine fixtures are module scoped, so that
  an engine releases the GPU before the next module starts one, and with a GPU
  present a start-up failure fails the test and does not skip it. A green run
  with skipped tests on a GPU machine is a failed run.
- `slurm/` is tested without a scheduler: `--dry-run` prints every `sbatch`
  line, and `tests/test_slurm_scripts.py` runs the scripts against stub
  commands. Keep `--dry-run` complete when you add a job.

## Documentation

- `README.md` holds the install section and the quickstart examples once.
  `docs/index.md` and `docs/python-api.md` include them through snippet
  sections, and `docs/slurm.md` includes `slurm/README.md` in full. A file that
  renders on GitHub and inside `docs/` carries no relative Markdown links.
- Tests keep the docs true: every full config block of the README and the docs
  loads (`tests/test_docs_configs.py`), every fenced `python` block parses and
  its imports resolve, and every error text in a `text` fence of
  `docs/troubleshooting.md` occurs in `src/` (`tests/test_docs.py`). A changed
  error message therefore needs a changed troubleshooting entry.
- Check every argument name, default, path, flag and error text that you write
  against the source. The common defect is a sentence that describes behaviour
  that the code does not have.
- One home per topic, links elsewhere.

## Changes in behaviour

- There is no backward-compatibility code: no reader for an old file format, no
  deprecated alias, no argument that is accepted and ignored. Break cleanly,
  raise an error that names the problem, and add an entry to
  `docs/migration.md` that says what to write now.
- Two error messages in `src/` name a heading of `docs/migration.md`. If you
  rename such a heading, change the message too.
- Performance work starts with a measurement. Measure the current cost under
  stated conditions, report the number, and change code only when the gain is
  material for a real run. A speed-up that adds a correctness condition to the
  resume path needs a large gain to be worth it. Record a measurement that led
  to "no change" where the next person will look for it.

## Git and pull requests

- Work on a branch, one pull request per topic, never directly on `main`.
  Do not tag and do not publish a release.
- Commit after every finished item, so that an interrupted session loses little.
  Stage whole files before a commit: the pre-commit hooks stash unstaged changes
  and can rewrite a partially staged file.
- A commit message has an imperative subject and a short body where it helps,
  and it reads like a person wrote it. No tool attribution, no co-author
  trailer for a tool, and no mention of an assistant, in commits and in pull
  request text.
- A pull request body says what changed (`- Name: text` items), what a user has
  to migrate, and what deserves a second look, including anything that was not
  run (a slow test without a GPU, a script without a scheduler).

## Before you hand work back

1. The fast suite passes, in the form shown under "The virtual environments".
2. `make quality`, `make typecheck` and `make build-docs` pass.
3. Coverage is not lower than on `main`.
4. You have read your own full diff once, and every statement in the docs and
   the docstrings matches the code.
5. The working tree is clean, and no background process of yours still runs.
6. Your summary states what was done, what was not run, and what you decided
   where the task left a choice. Say plainly when something failed.
