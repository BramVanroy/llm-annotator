"""Run an annotation pipeline from a JSON or YAML config file.

This is a thin wrapper so the pipeline can be started from a checkout without
installing the package's console script::

    python scripts/annotate.py examples/pipeline-qa/config.yaml

An installed llm-annotator exposes the very same entry point as
``llm-annotate``. All of the logic lives in :mod:`llm_annotator.pipeline`, which
is where to look (and where to add tests) when changing behaviour.
"""

from llm_annotator.pipeline import main


if __name__ == "__main__":
    main()
