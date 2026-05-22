.PHONY: quality style test test-fast test-slow test-integration test-all typecheck serve-docs

PACKAGE = src/llm_annotator

quality:
	uv run interrogate -vv
	uv run ruff check $(PACKAGE) tests/ examples/
	uv run ruff format --check $(PACKAGE) tests/ examples/

style:
	uv run ruff check $(PACKAGE) tests/ examples/ --fix
	uv run ruff format $(PACKAGE) tests/ examples/

test:
	uv run pytest -m "not slow" --cov=$(PACKAGE) --cov-report=term-missing --cov-report=xml

test-fast:
	uv run pytest -m "not slow" --cov=$(PACKAGE) --cov-report=term-missing --cov-report=xml

test-slow:
	uv run pytest -m "slow" --cov=$(PACKAGE) --cov-report=term-missing --cov-report=xml

test-integration:
	uv run pytest -m "integration" --cov=$(PACKAGE) --cov-report=term-missing --cov-report=xml

test-all:
	uv run pytest --cov=$(PACKAGE) --cov-report=term-missing --cov-report=xml

typecheck:
	uv run mypy $(PACKAGE) tests/ examples/

# Only used for doc-testing, not for remote doc deployment
DOCS_BRANCH ?= tmp-gh-pages
DOCS_VERSION ?= 0.0.0
DOCS_ALIAS ?= latest
DOCS_ADDR ?= 127.0.0.1:8000
DOCS_SOURCE_REF ?= main

serve-docs:
	DOCS_SOURCE_REF=$(DOCS_SOURCE_REF) uv run mike deploy --branch $(DOCS_BRANCH) --update-aliases $(DOCS_VERSION) $(DOCS_ALIAS)
	uv run mike set-default --branch $(DOCS_BRANCH) $(DOCS_ALIAS)
	uv run mike serve -b $(DOCS_BRANCH) -a $(DOCS_ADDR)

