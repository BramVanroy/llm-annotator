quality:
	ruff check src/llm_annotator tests/ examples/
	ruff format --check src/llm_annotator tests/ examples/

style:
	ruff check src/llm_annotator tests/ examples/ --fix
	ruff format src/llm_annotator tests/ examples/

setup:
	uv sync --dev --group docs
	pre-commit install --hook-type pre-push

docs:
	@cd docs && $(MAKE) html

docs-serve:
	@cd docs && python -m http.server -d _build/html 8000

.PHONY: quality style setup docs docs-serve
