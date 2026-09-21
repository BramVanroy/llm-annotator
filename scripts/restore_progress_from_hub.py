"""Restore a run's Hub progress backup from a checkout that is not installed.

This is a thin wrapper so the restore can be run from a checkout without
installing the package's console script::

    python scripts/restore_progress_from_hub.py \
        --hub-id user/my-dataset --output-dir outputs/run

An installed llm-annotator exposes the very same entry point as
``llm-annotate-restore``. All of the logic lives in
:func:`llm_annotator.hub.restore_progress_from_hub`, which is where to look
(and where to add tests) when changing behaviour.
"""

from llm_annotator.hub import main


if __name__ == "__main__":
    main()
