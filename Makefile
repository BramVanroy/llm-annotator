quality:
	ruff check src/llm_annotator tests/
	ruff format --check src/llm_annotator tests/

style:
	ruff check src/llm_annotator tests/ --fix
	ruff format src/llm_annotator tests/
