quality:
	ruff check src/llm_annotator scripts/
	ruff format --check src/llm_annotator scripts/

style:
	ruff check src/llm_annotator scripts/ --fix
	ruff format src/llm_annotator scripts/
