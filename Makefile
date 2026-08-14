.PHONY: quality style style-check style-fix test test-fast test-slow test-integration test-all typecheck serve-docs serve-docs-versioned build-docs

PACKAGE = src/llm_annotator

quality:
	uv run interrogate -vv
	$(MAKE) style-check

style-check:
	uv run ruff check $(PACKAGE) tests/ examples/ scripts/
	uv run ruff format --check $(PACKAGE) tests/ examples/ scripts/

# Explicit manual repo-wide cleanup; not intended for normal commit hooks.
style-fix:
	uv run ruff check $(PACKAGE) tests/ examples/ scripts/ --fix
	uv run ruff format $(PACKAGE) tests/ examples/ scripts/

style: style-fix

# Coverage flags live in [tool.pytest.ini_options] addopts, so the recipes below
# only need to select which marker subset to run.
test:
	uv run pytest -m "not slow"

# Alias kept because CI calls it by this name.
test-fast: test

test-slow:
	uv run pytest -m "slow"

test-integration:
	uv run pytest -m "integration"

test-all:
	uv run pytest

typecheck:
	uv run mypy $(PACKAGE) tests/ scripts/

# Everyday local preview: live reload, no git branches involved.
serve-docs:
	uv run mkdocs serve

# What CI runs on release. Fails on broken cross-references and nav entries.
build-docs:
	uv run mkdocs build --strict

# Only needed to exercise the mike version selector and the "outdated version"
# banner locally. Writes to a throwaway local branch; never push it.
DOCS_BRANCH ?= tmp-gh-pages
DOCS_VERSION ?= 0.0.0
DOCS_ALIAS ?= latest
DOCS_ADDR ?= 127.0.0.1:8000
DOCS_SOURCE_REF ?= main

serve-docs-versioned:
	DOCS_SOURCE_REF=$(DOCS_SOURCE_REF) uv run mike deploy --branch $(DOCS_BRANCH) --update-aliases $(DOCS_VERSION) $(DOCS_ALIAS)
	uv run mike set-default --branch $(DOCS_BRANCH) $(DOCS_ALIAS)
	uv run mike serve -b $(DOCS_BRANCH) -a $(DOCS_ADDR)
