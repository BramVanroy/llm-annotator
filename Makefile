quality:
	ruff check src/llm_annotator tests/ examples/
	ruff format --check src/llm_annotator tests/ examples/

style:
	ruff check src/llm_annotator tests/ examples/ --fix
	ruff format src/llm_annotator tests/ examples/

setup:
	uv sync --dev
	pre-commit install --hook-type pre-push
