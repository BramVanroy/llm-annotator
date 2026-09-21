"""Sequential executor and CLI for config-driven annotation pipelines.

[`run_pipeline`][llm_annotator.pipeline.run_pipeline] walks the steps of a
[`PipelineConfig`][llm_annotator.config.PipelineConfig] in order, handing the
dataset each step produces to the next one. Nothing here re-implements
annotation: a step is one ordinary
[`annotate_dataset`][llm_annotator.annotator.Annotator.annotate_dataset] call,
so prompt templating, JSONL progress checkpoints, resumption, retries and Hub
backups behave exactly as they do when the library is used directly.

Two things make a *pipeline* more than a loop:

* Every step owns a subdirectory of ``output_dir`` and a ``task_prefix``, so
  its internal columns and artifacts cannot collide with another step's.
* A finished step writes its result to ``<step-dir>/output``. On a re-run that
  snapshot is loaded and the step is skipped, so a pipeline that dies in step
  three does not repeat steps one and two.
* The ``idx_column`` of the first step is the id of a row in every later step.
  A re-run with a higher ``dataset.max_num_samples`` resumes every step and
  sends only the new rows to the model.

[`main`][llm_annotator.pipeline.main] is the ``llm-annotate`` console entry
point.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import yaml
from datasets import Dataset

from llm_annotator.annotator import (
    _COMPONENT_CHANGES,
    PROGRESS_DS_LOCAL_SUBDIR,
    Annotator,
    SelectionRecord,
    VLLMQueueAnnotator,
    _preparation_components,
    _reuse_error,
    _schema_component,
    is_retried_error,
)
from llm_annotator.config import (
    ClientConfig,
    PipelineConfig,
    StepConfig,
    load_config_file,
    load_pipeline_config,
)
from llm_annotator.logging_utils import configure_logging, get_logger
from llm_annotator.utils import dataset_signature, drop_jsonl_rows


LOGGER = get_logger("pipeline")

STEP_OUTPUT_SUBDIR = "output"
"""Name of the subdirectory holding a finished step's dataset.

Its presence marks a step as done for the selection in the step's
[`SelectionRecord`][llm_annotator.annotator.SelectionRecord].
"""

STEP_ANNOTATE_SUBDIR = "annotate"
"""Name of the subdirectory handed to the annotator as its ``output_dir``.

It is one level below the step directory because ``_post_annotate`` writes a
dataset into the root of whatever ``output_dir`` it is given.
"""


def _is_complete(snapshot_dir: Path) -> bool:
    """Check whether a step already wrote its result snapshot.

    Args:
        snapshot_dir: The step's ``output`` directory.

    Returns:
        ``True`` when the directory exists and holds a saved dataset.
    """
    return snapshot_dir.is_dir() and (snapshot_dir / "state.json").is_file()


def _step_components(
    config: PipelineConfig,
    step: StepConfig,
    dataset: Dataset | None,
    is_first: bool,
    upstream_token: str | None,
) -> dict[str, str]:
    """Describe the settings that a step's finished rows depend on.

    The values must match what
    [`prepare_data`][llm_annotator.annotator.Annotator.prepare_data] records
    for the same step, so this mirrors the arguments that ``_run_step``
    passes. ``upstream`` is the pipeline's own addition: the columns a step
    reads are only the ones its input held when it ran.

    Args:
        config: The pipeline configuration.
        step: The step to describe.
        dataset: The step's in-memory input, or ``None`` for a source that the
            annotator loads itself.
        is_first: Whether this is the pipeline's first step.
        upstream_token: Token of the run of the step that produced the input.

    Returns:
        One short string per setting, keyed by the setting's name.
    """
    root = config.config_dir
    source = config.dataset if is_first else None
    template = (
        _generate_template(step, root)
        if step.type == "generate"
        else step.resolved_prompt(root)
    )
    components = _preparation_components(
        prompt_template=template or "",
        system_message=step.resolved_system_prompt(root),
        sort_by_length=step.sort_by_length,
        idx_column=config.idx_column,
        reuse_idx_column=not is_first,
        shuffle_seed=source.shuffle_seed if source else None,
        preprocess_fn=None,
        source_signature=(
            dataset_signature(dataset) if dataset is not None else None
        ),
    )
    components["output_schema"] = _schema_component(
        step.resolved_output_schema(root)
    )
    # A step whose own run is not recorded (the first step, or one that
    # finished under a version that kept no record) leaves the key out, so
    # that an unknown token is not read as a changed one.
    if upstream_token is not None:
        components["upstream"] = upstream_token
    return components


def _record_step_run(
    annotate_dir: Path,
    task_prefix: str,
    token: str,
    upstream_token: str | None,
) -> str | None:
    """Add a finished step's own token and its input's token to its record.

    Args:
        annotate_dir: The directory the annotator wrote its record to.
        task_prefix: The step's task prefix.
        token: The token of this run of the step.
        upstream_token: The token of the step that produced the input.

    Returns:
        The token that was recorded, or ``None`` when the step has no record
        to attach it to, so that the next step compares against nothing.
    """
    record = SelectionRecord.read(annotate_dir, task_prefix)
    if record is None:
        return None
    components = {**record.components, "run_token": token}
    if upstream_token is not None:
        components["upstream"] = upstream_token
    dataclasses.replace(record, components=components).write(
        annotate_dir, task_prefix
    )
    return token


def _conflict_causes(
    config: PipelineConfig, index: int, conflicting: list[str]
) -> list[str]:
    """Name the changes that keep a step from resuming, for its error.

    Args:
        config: The pipeline configuration.
        index: Index of the step that cannot resume.
        conflicting: Names of the components that differ from its record.

    Returns:
        One phrase per component, in the order they were given.
    """
    causes = []
    for name in conflicting:
        if name == "upstream":
            causes.append(
                f"step '{config.steps[index - 1].name}' was annotated again"
                " from scratch"
            )
        else:
            causes.append(_COMPONENT_CHANGES[name])
    return causes


def _step_remedy(config: PipelineConfig, index: int) -> str:
    """Name the command that annotates a step and the ones after it again.

    Args:
        config: The pipeline configuration.
        index: Index of the step that cannot resume.

    Returns:
        A sentence for [`_reuse_error`][llm_annotator.annotator._reuse_error].
    """
    names = " ".join(step.name for step in config.steps[index:])
    return (
        "Restore the old value(s), use a new 'output_dir', or re-run with"
        f" '--steps {names} --overwrite' to annotate that step and the ones"
        " that read it again from scratch."
    )


def _load_input_dataset(config: PipelineConfig) -> Dataset | None:
    """Load the pipeline's source dataset when it lives on disk.

    A Hub id or builder name is *not* loaded here: it is forwarded to
    [`annotate_dataset`][llm_annotator.annotator.Annotator.annotate_dataset],
    which knows how to verify splits and apply ``max_num_samples``.

    Args:
        config: The pipeline configuration.

    Returns:
        The dataset when ``dataset.path`` was given, else ``None``.
    """
    if config.dataset is None or config.dataset.path is None:
        return None

    path = config.dataset.path
    if not path.is_absolute():
        path = (config.config_dir / path).resolve()
    LOGGER.info(f"Loading source dataset from disk at '{path}'.")
    return Dataset.load_from_disk(str(path))


def _generate_template(step: StepConfig, root: Path) -> str:
    """Resolve the prompt template of a ``generate`` step.

    Args:
        step: The generate step.
        root: Directory that relative config paths resolve against.

    Returns:
        The template that renders each entry of ``prompts``.

    Raises:
        ValueError: If an explicit template does not contain ``{prompt}``.
    """
    template = step.resolved_prompt(root)
    if template is None:
        return "{prompt}"
    if "{prompt}" not in template:
        raise ValueError(
            f"Step '{step.name}': the template of a 'generate' step must"
            " contain the '{prompt}' placeholder, which is filled in with each"
            " entry of 'prompts'."
        )
    return template


def _generate_dataset(step: StepConfig, root: Path) -> Dataset:
    """Build the synthetic prompt dataset for a ``generate`` step.

    This mirrors what
    [`generate_dataset`][llm_annotator.annotator.Annotator.generate_dataset] does
    internally, but routes through ``annotate_dataset`` instead so a generate
    step also gets ``system_prompt``, ``sort_by_length`` and the per-step
    column handling that the rest of the pipeline offers.

    Args:
        step: The generate step.
        root: Directory that relative config paths resolve against.

    Returns:
        The one-column prompt dataset.
    """
    prompts = step.resolved_prompts(root)
    LOGGER.info(f"Step '{step.name}': {len(prompts):,} prompt(s) to generate.")
    return Dataset.from_dict({"prompt": prompts})


def _postprocess_step(
    dataset: Dataset,
    step: StepConfig,
    task_prefix: str,
    num_proc: int | None,
) -> Dataset:
    """Apply a step's column bookkeeping to the dataset it produced.

    Order matters: invalid rows are dropped first so the retained columns are
    only judged on rows that survive, then the rendered prompts are pruned, and
    only then are columns renamed and removed -- that way ``rename`` and
    ``drop_columns`` refer to the names the model actually produced.

    Args:
        dataset: The dataset returned by the annotator.
        step: The step configuration.
        task_prefix: The step's resolved task prefix.
        num_proc: Number of processes for the filter operation.

    Returns:
        The cleaned-up dataset.

    Raises:
        ValueError: If every row was invalid, or if ``rename``/``drop_columns``
            names a column that does not exist.
    """
    if step.filter_invalid:
        valid_column = f"{task_prefix}valid_fields"
        if valid_column not in dataset.column_names:
            raise ValueError(
                f"Step '{step.name}': cannot filter invalid samples because"
                f" column '{valid_column}' is missing. This column only exists"
                " when an output schema is configured."
            )
        num_before = len(dataset)
        dataset = dataset.filter(
            lambda is_valid: bool(is_valid),
            input_columns=[valid_column],
            num_proc=num_proc,
            desc="Filtering invalid samples",
        )
        num_dropped = num_before - len(dataset)
        if not len(dataset):
            raise ValueError(
                f"Step '{step.name}': all {num_before:,} samples failed schema"
                " validation, so there is nothing left for the next step. A"
                " common cause is a 'max_completion_tokens' too low for the"
                " schema."
            )
        if num_dropped:
            LOGGER.warning(
                f"Step '{step.name}': dropped {num_dropped:,} invalid"
                f" sample(s), {len(dataset):,} remaining."
            )

    if not step.keep_messages:
        messages_column = f"{task_prefix}messages"
        if messages_column in dataset.column_names:
            dataset = dataset.remove_columns([messages_column])

    if step.rename:
        missing = sorted(set(step.rename) - set(dataset.column_names))
        if missing:
            raise ValueError(
                f"Step '{step.name}': 'rename' refers to column(s) {missing}"
                f" that the step did not produce. Available columns:"
                f" {sorted(dataset.column_names)}."
            )
        clashes = sorted(
            new
            for old, new in step.rename.items()
            if new in dataset.column_names and new != old
        )
        if clashes:
            raise ValueError(
                f"Step '{step.name}': 'rename' targets {clashes} which already"
                " exist in the dataset. Pick different names or drop the"
                " existing columns first."
            )
        dataset = dataset.rename_columns(step.rename)

    if step.drop_columns:
        missing = sorted(set(step.drop_columns) - set(dataset.column_names))
        if missing:
            raise ValueError(
                f"Step '{step.name}': 'drop_columns' refers to column(s)"
                f" {missing} that do not exist. Available columns:"
                f" {sorted(dataset.column_names)}."
            )
        dataset = dataset.remove_columns(list(step.drop_columns))

    return dataset


def _run_step(
    *,
    annotator: Annotator,
    config: PipelineConfig,
    step: StepConfig,
    client_config: ClientConfig,
    dataset: Dataset | None,
    is_first: bool,
    step_dir: Path,
) -> Dataset:
    """Run one annotation step and return the dataset it produced.

    Args:
        annotator: The annotator to run the step on.
        config: The pipeline configuration.
        step: The step to run.
        client_config: The step's effective client configuration.
        dataset: The incoming dataset (the prompt dataset for a ``generate``
            step), or ``None`` when the first step should load it from a Hub id
            or builder name itself.
        is_first: Whether this is the pipeline's first step.
        step_dir: Directory holding this step's artifacts.

    Returns:
        The annotated dataset, before column bookkeeping.
    """
    root = config.config_dir
    task_prefix = step.resolved_task_prefix()
    output_schema = step.resolved_output_schema(root)
    options = client_config.build_options(output_schema)

    kwargs: dict[str, Any] = {
        "output_dir": step_dir / STEP_ANNOTATE_SUBDIR,
        "task_prefix": task_prefix,
        "idx_column": config.idx_column,
        # Every input column must survive, otherwise later steps could not
        # reference what earlier steps produced.
        "keep_columns": True,
        "options": options,
        "gen_kwargs": client_config.gen_kwargs or None,
        "output_schema": output_schema,
        "system_message": step.resolved_system_prompt(root),
        "sort_by_length": step.sort_by_length,
        "num_retries_invalid": step.num_retries_invalid,
        "max_samples_per_output_file": step.max_samples_per_output_file,
        "max_consecutive_failed_batches": step.max_consecutive_failed_batches,
        "upload_every_n_samples": step.upload_every_n_samples,
        "hub_id": step.hub_id,
        "overwrite": config.overwrite,
        "force_data_preparation": step.force_data_preparation,
        # The ids of the first step identify a row in every later step. Ids
        # that are numbered again by position change when the input grows,
        # and the progress files of a later step would then name other rows.
        "keep_idx_column": True,
        "reuse_idx_column": not is_first,
    }

    if step.type == "generate":
        kwargs["dataset"] = dataset
        kwargs["prompt_template"] = _generate_template(step, root)
    else:
        kwargs["prompt_template"] = step.resolved_prompt(root)
        if dataset is not None:
            kwargs["dataset"] = dataset
        else:
            source = config.dataset
            assert source is not None  # guaranteed by PipelineConfig
            kwargs["dataset_name"] = source.name
            kwargs["dataset_config"] = source.config
            kwargs["dataset_split"] = source.split
            kwargs["data_dir"] = source.data_dir
            kwargs["data_files"] = source.data_files

    if is_first and config.dataset is not None:
        kwargs["max_num_samples"] = config.dataset.max_num_samples
        kwargs["shuffle_seed"] = config.dataset.shuffle_seed

    return annotator.annotate_dataset(**kwargs)


def _check_unrecorded_run(
    config: PipelineConfig, will_overwrite: bool
) -> None:
    """Refuse to grow a run whose finished steps carry no sample ids.

    A first step that finished under a version without selection records
    numbered its rows by position and dropped the ids afterwards. The later
    steps of such a run cannot be resumed on a larger input, because their
    progress files would name other rows than before. The previous cap and
    seed come from ``pipeline.json``, so this check has to run before that
    file is replaced.

    Args:
        config: The pipeline configuration.
        will_overwrite: Whether this run deletes the first step's directory.

    Raises:
        ValueError: If the first step finished without a record and the
            config now asks for another cap or seed.
    """
    first_dir = config.step_dir(0)
    record = SelectionRecord.read(
        first_dir / STEP_ANNOTATE_SUBDIR,
        config.steps[0].resolved_task_prefix(),
    )
    if (
        will_overwrite
        or record is not None
        or not _is_complete(first_dir / STEP_OUTPUT_SUBDIR)
    ):
        return

    try:
        previous = json.loads(
            (config.output_dir / "pipeline.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return

    before = previous.get("dataset") or {}
    now = config.dataset.model_dump() if config.dataset else {}
    changed = [
        key
        for key in ("max_num_samples", "shuffle_seed")
        if before.get(key) != now.get(key)
    ]
    if changed:
        raise ValueError(
            f"Step '{config.steps[0].name}' finished under a version of"
            " llm-annotator that did not record its sample selection, and"
            f" {changed} changed since. Such a run cannot grow, because its"
            " later steps did not keep a stable sample id. Restore the old"
            " value(s), use a new 'output_dir', or pass --overwrite."
        )


def _resolve_selection(
    config: PipelineConfig, selected: Sequence[str] | None
) -> range:
    """Turn a set of step names into the contiguous index range to run.

    Args:
        config: The pipeline configuration.
        selected: Step names to run, or ``None`` for all of them.

    Returns:
        The indices to execute, as a contiguous range.

    Raises:
        ValueError: If a name is unknown, or if the selection has a hole in it.
    """
    if not selected:
        return range(len(config.steps))

    positions = {step.name: index for index, step in enumerate(config.steps)}
    unknown = [name for name in selected if name not in positions]
    if unknown:
        raise ValueError(
            f"Unknown step(s) {sorted(unknown)}. This pipeline defines"
            f" {list(positions)}."
        )

    indices = sorted(positions[name] for name in selected)
    # A hole would hand the step after it a dataset that never got the columns
    # the skipped step produces, so the prompt would reference nothing.
    if indices != list(range(indices[0], indices[-1] + 1)):
        missing = [
            config.steps[i].name
            for i in range(indices[0], indices[-1] + 1)
            if i not in indices
        ]
        raise ValueError(
            f"Steps must be selected contiguously; {missing} would be skipped"
            " in the middle, and later steps read the columns they produce."
        )
    return range(indices[0], indices[-1] + 1)


def run_pipeline(
    config: PipelineConfig,
    selected: Sequence[str] | None = None,
    retry_errors: bool | Sequence[str] = False,
) -> Dataset:
    """Run a pipeline, or part of one, and return the resulting dataset.

    Steps share one live client whenever their provider, model and constructor
    settings match, so a pipeline that uses the same local model twice loads it
    only once. The client is always released before returning, including on
    failure.

    Passing ``selected`` runs only those steps. Earlier steps must already have
    finished: their saved output is loaded as the input, which is what lets a
    scheduler run one step per job while keeping a single config file as the
    source of truth.

    Args:
        config: The validated pipeline configuration.
        selected: Names of the steps to run. ``None`` runs all of them. The
            names must form a contiguous run of the pipeline.
        retry_errors: Annotate rows again that finished with an error.
            ``True`` redoes every errored row of the selected steps, a
            sequence redoes only those error types, e.g. ``["ConnectError"]``.
            A row that is redone in one step is also redone in every selected
            step after it, because those steps read what it produces.

    Returns:
        The dataset produced by the last step that ran.

    Raises:
        ValueError: If the selection is unknown or non-contiguous, or if a step
            before it has not run yet.

    Examples:
        >>> from llm_annotator import load_pipeline_config, run_pipeline
        >>> config = load_pipeline_config("config.yaml")  # doctest: +SKIP
        >>> dataset = run_pipeline(config)  # doctest: +SKIP
        >>> only = run_pipeline(config, selected=["judge"])  # doctest: +SKIP
    """
    chosen = _resolve_selection(config, selected)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _check_unrecorded_run(config, config.overwrite and 0 in chosen)
    snapshot = config.output_dir / "pipeline.json"
    snapshot.write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )

    dataset: Dataset | None = _load_input_dataset(config)
    annotator: Annotator | None = None
    active_client_key: str | None = None
    runs_last_step = chosen.stop >= len(config.steps)
    retried_idxs: set[Any] = set()
    upstream_token: str | None = None

    try:
        for index, step in enumerate(config.steps):
            if index >= chosen.stop:
                break

            step_dir = config.step_dir(index)
            step_output = step_dir / STEP_OUTPUT_SUBDIR
            task_prefix = step.resolved_task_prefix()
            label = f"Step {index + 1}/{len(config.steps)} '{step.name}'"

            # Only wipe what this run is actually going to redo; a step outside
            # the selection is a dependency, not something to throw away.
            if config.overwrite and index in chosen and step_dir.is_dir():
                LOGGER.info(f"{label}: removing '{step_dir}' (overwrite).")
                shutil.rmtree(step_dir, ignore_errors=True)

            if step.type == "generate":
                dataset = _generate_dataset(step, config.config_dir)

            if retry_errors and index in chosen:
                retried_rows = drop_jsonl_rows(
                    step_dir
                    / STEP_ANNOTATE_SUBDIR
                    / f"{task_prefix}{PROGRESS_DS_LOCAL_SUBDIR}",
                    lambda row: (
                        row.get(config.idx_column) in retried_idxs
                        or is_retried_error(
                            row,
                            retry_errors=retry_errors,
                            task_prefix=task_prefix,
                        )
                    ),
                )
                retried_idxs.update(
                    row[config.idx_column] for row in retried_rows
                )
                LOGGER.info(
                    f"{label}: {len(retried_rows):,} sample(s) are annotated"
                    " again (retry_errors)."
                )

            annotate_dir = step_dir / STEP_ANNOTATE_SUBDIR
            progress_dir = (
                annotate_dir / f"{task_prefix}{PROGRESS_DS_LOCAL_SUBDIR}"
            )
            record = SelectionRecord.read(annotate_dir, task_prefix)
            changed: list[str] = []
            is_outdated = bool(retried_idxs)
            if record is not None:
                changed = record.changed_components(
                    _step_components(
                        config, step, dataset, index == 0, upstream_token
                    )
                )
                source = config.dataset if index == 0 else None
                cap = source.max_num_samples if source else None
                is_outdated = (
                    is_outdated
                    or bool(changed)
                    or record.max_num_samples != cap
                )

            if _is_complete(step_output) and not is_outdated:
                LOGGER.info(
                    f"{label}: already finished, loading its result from"
                    f" '{step_output}'."
                )
                dataset = Dataset.load_from_disk(str(step_output))
                upstream_token = (
                    record.components.get("run_token") if record else None
                )
                continue

            if index not in chosen:
                reason = (
                    "finished with other settings than the ones that are"
                    " requested now"
                    if _is_complete(step_output)
                    else "not run yet"
                )
                raise ValueError(
                    f"Step '{step.name}' has {reason}, so there is no input"
                    f" for '{config.steps[chosen.start].name}'. Run it first,"
                    " or select it too."
                )

            # A changed input is what growth looks like from a later step, so
            # it is resumed. Every other change gives the finished rows
            # another meaning, and the step's own result is still on disk, so
            # this is refused before anything is removed.
            conflicting = [name for name in changed if name != "dataset"]
            if (
                conflicting
                and not config.overwrite
                and any(progress_dir.glob("*.jsonl"))
            ):
                raise _reuse_error(
                    progress_dir,
                    _conflict_causes(config, index, conflicting),
                    _step_remedy(config, index),
                )

            if _is_complete(step_output):
                if not retried_idxs:
                    LOGGER.info(
                        f"{label}: its input or its sample selection changed"
                        " since it finished. Resuming it; rows in its"
                        " progress files are not sent to the model again."
                    )
                # Before the step runs: the annotator replaces the selection
                # record, and a crash after that would leave an old result
                # that passes for a finished one.
                shutil.rmtree(step_output, ignore_errors=True)

            # A step that starts with nothing in its progress files answers
            # every row again, so the steps that read it must not keep the
            # rows they annotated on the previous answers.
            token = record.components.get("run_token") if record else None
            if token is None or not any(progress_dir.glob("*.jsonl")):
                token = uuid4().hex

            client_config = config.step_client(step)
            client_key = client_config.cache_key()
            if annotator is None or client_key != active_client_key:
                if annotator is not None:
                    annotator.destroy()
                annotator = client_config.build_annotator(
                    config.config_dir, verbose=config.verbose
                )
                active_client_key = client_key
            else:
                # Same underlying client, but batching is per step and cheap
                # to change without rebuilding anything. The queue settings
                # are in the same category and deliberately absent from
                # `cache_key`, so they have to be refreshed here too or the
                # step would silently run with the previous step's values.
                annotator.batch_size = client_config.batch_size
                annotator.num_proc = client_config.num_proc
                if isinstance(annotator, VLLMQueueAnnotator):
                    annotator.set_max_concurrent_batches_per_client(
                        client_config.max_concurrent_batches_per_client
                    )
                    annotator.set_queue_size(client_config.queue_size)

            LOGGER.info(
                f"{label}: {step.type} with"
                f" '{client_config.provider}' model"
                f" '{client_config.model or 'served-default'}'."
            )
            dataset = _run_step(
                annotator=annotator,
                config=config,
                step=step,
                client_config=client_config,
                dataset=dataset,
                is_first=index == 0,
                step_dir=step_dir,
            )
            dataset = _postprocess_step(
                dataset, step, task_prefix, annotator.num_proc
            )
            upstream_token = _record_step_run(
                annotate_dir, task_prefix, token, upstream_token
            )

            dataset.save_to_disk(str(step_output))
            LOGGER.info(
                f"{label}: done, {len(dataset):,} sample(s) saved to"
                f" '{step_output}'."
            )
    finally:
        if annotator is not None:
            annotator.destroy()

    assert dataset is not None  # at least one step always runs

    # Only the run that finishes the pipeline may publish it. A per-step run
    # that stopped early has a partial dataset, which must not land in `final`
    # or on the Hub as though it were the finished article.
    if not runs_last_step:
        LOGGER.info(
            f"Stopped after step '{config.steps[chosen.stop - 1].name}';"
            f" {len(dataset):,} sample(s) carried forward. Run the remaining"
            " step(s) to finish the pipeline."
        )
        return dataset

    if config.idx_column in dataset.column_names:
        dataset = dataset.remove_columns([config.idx_column])

    final_dir = config.output_dir / "final"
    dataset.save_to_disk(str(final_dir))
    LOGGER.info(
        f"Pipeline finished: {len(dataset):,} sample(s) in '{final_dir}'."
    )

    if config.hub_id:
        LOGGER.info(f"Pushing the final dataset to '{config.hub_id}'.")
        dataset.push_to_hub(config.hub_id, private=True)

    return dataset


def _pool_source_override(
    config_path: Path,
    hosts_file: Path | None,
    url_glob: str | None,
    selected: Sequence[str] | None,
) -> dict[str, dict[str, Any]] | None:
    """Work out which step a ``--hosts-file`` or ``--url-glob`` belongs to.

    The pool source is attached to the selected steps that run on vLLM, never
    to the pipeline as a whole: a step on a hosted provider must not inherit a
    pool it cannot use, and a set of servers only ever serves one model.

    Args:
        config_path: Path to the config file, loaded once to see the providers.
        hosts_file: The path given to ``--hosts-file``, or ``None``.
        url_glob: The pattern given to ``--url-glob``, or ``None``.
        selected: Step names being run, or ``None`` for all of them.

    Returns:
        A per-step client override mapping, or ``None`` when neither flag was
        given.

    Raises:
        ValueError: If both flags are given, if no selected step runs on vLLM,
            or if the selected vLLM steps disagree about which model they want,
            since one pool of servers can only serve one of them.
    """
    if hosts_file is not None and url_glob is not None:
        raise ValueError(
            "--hosts-file and --url-glob both name the pool's servers; give"
            " one of them."
        )
    if hosts_file is not None:
        # A command-line path means what the shell means by it. Config paths
        # resolve against the config file instead, so pin it down before it is
        # handed over as though it had been written in the config.
        source = {"hosts_file": str(hosts_file.expanduser().resolve())}
        flag = "--hosts-file"
    elif url_glob is not None:
        source = {"url_glob": str(Path(url_glob).expanduser().absolute())}
        flag = "--url-glob"
    else:
        return None

    probe = load_pipeline_config(config_path)
    targets = {}
    for index in _resolve_selection(probe, selected):
        step = probe.steps[index]
        client = probe.step_client(step)
        if client.provider == "vllm_online":
            targets[step.name] = client.model

    if not targets:
        raise ValueError(
            f"{flag} points at vLLM servers, but none of the steps being"
            f" run ({selected or 'all'}) uses provider 'vllm_online'."
        )

    models = {model for model in targets.values() if model is not None}
    if len(models) > 1:
        raise ValueError(
            f"Steps {sorted(targets)} want different models {sorted(models)},"
            " but one set of vLLM servers serves a single model. Run them as"
            " separate --steps invocations, each against its own servers."
        )

    return {name: dict(source) for name in targets}


def _serve_args(config: PipelineConfig, step_name: str) -> list[str]:
    """Build the ``vllm serve`` argument list for one step.

    This is what lets a server job read its own serving profile out of the
    config instead of being handed one through the environment: a job submitter
    cannot carry a JSON value like ``--speculative-config`` through
    ``sbatch --export``, and a pipeline whose steps use different models needs
    a different profile per step anyway.

    ``--host`` and ``--port`` are deliberately absent. The port is probed on the
    node, because two servers of the same pool can land on one machine.

    Args:
        config: The loaded pipeline configuration.
        step_name: Name of the step whose servers are being started.

    Returns:
        Arguments to pass to ``vllm serve``, starting with the model.

    Raises:
        ValueError: If the config has no such step, or if that step needs no
            servers started for it.
    """
    for step in config.steps:
        if step.name == step_name:
            break
    else:
        raise ValueError(
            f"Config defines no step '{step_name}'. It has"
            f" {[s.name for s in config.steps]}."
        )

    client = config.step_client(step)
    kind = client.kind()
    if kind != "vllm_pool":
        detail = {
            "api": "runs on a hosted provider",
            "vllm_offline": "loads the model in-process",
            "vllm_online": "points at servers that already exist",
        }[kind]
        raise ValueError(
            f"Step '{step_name}' {detail}, so there is no vLLM server to start"
            " for it."
        )
    if client.model is None:
        raise ValueError(
            f"Step '{step_name}' names no 'model', so there is nothing to"
            " serve. A client can ask a running server what it serves, but"
            " nothing can ask a server that does not exist yet."
        )

    return [
        client.model,
        "--served-model-name",
        client.model,
        *client.engine.as_serve_args(),
    ]


_DATASET_FLAG_KEYS = ("dataset.max_num_samples", "dataset.shuffle_seed")


def _parse_set_override(assignment: str) -> tuple[str, Any]:
    """Split one ``--set key=value`` argument into a config key and a value.

    The value is read as YAML, the same way the config file itself is, so
    ``2000`` is an integer, ``true`` a boolean, ``[a, b]`` a list and anything
    else the string it looks like.

    Args:
        assignment: The argument as typed, e.g.
            ``dataset.max_num_samples=2000``.

    Returns:
        The config key and the decoded value.

    Raises:
        ValueError: If the argument has no ``=``, or an empty key or path
            segment.

    Examples:
        >>> _parse_set_override("dataset.max_num_samples=2000")
        ('dataset.max_num_samples', 2000)
        >>> _parse_set_override("client.model=gpt-4o-mini")
        ('client.model', 'gpt-4o-mini')
    """
    key, separator, raw = assignment.partition("=")
    key = key.strip()
    if not separator or not all(key.split(".")):
        raise ValueError(
            f"--set expects 'key=value', got '{assignment}'. Use a dotted key"
            " to reach a nested value, as in"
            " --set dataset.max_num_samples=2000."
        )

    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw
    return key, value


def _cli_overrides(
    config_path: Path,
    output_dir: str | None = None,
    hub_id: str | None = None,
    log_level: str | None = None,
    overwrite: bool | None = None,
    max_num_samples: int | None = None,
    shuffle_seed: int | None = None,
    settings: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Collect the config overrides typed on the command line.

    The named flags are shorthands for keys that ``--set`` can also reach, so
    they end up in the same mapping and a key given twice is an error rather
    than a silent winner.

    Args:
        config_path: Path to the config file, read only to check that the two
            dataset flags have a block to apply to.
        output_dir: Value of ``--output-dir``, resolved against the current
            directory because it is typed at the shell.
        hub_id: Value of ``--hub-id``.
        log_level: Value of ``--log-level``.
        overwrite: ``True`` when ``--overwrite`` was passed.
        max_num_samples: Value of ``--max-num-samples``.
        shuffle_seed: Value of ``--shuffle-seed``.
        settings: Raw ``--set key=value`` arguments.

    Returns:
        Overrides keyed the way
        [`load_pipeline_config`][llm_annotator.config.load_pipeline_config]
        expects them.

    Raises:
        ValueError: If a ``--set`` argument is malformed, if a key is given
            twice, or if a dataset flag is used on a config that has no
            ``dataset`` block.
    """
    overrides: dict[str, Any] = {}
    named: tuple[tuple[str, Any], ...] = (
        (
            "output_dir",
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else None,
        ),
        ("hub_id", hub_id),
        ("log_level", log_level),
        ("overwrite", overwrite),
        ("dataset.max_num_samples", max_num_samples),
        ("dataset.shuffle_seed", shuffle_seed),
    )
    for key, value in named:
        if value is not None:
            overrides[key] = value

    # Without this, the flags would create a 'dataset' block on a pipeline that
    # generates its own data, and the run would fail on a validation error that
    # says nothing about the flag that caused it.
    if set(_DATASET_FLAG_KEYS) & set(overrides):
        if "dataset" not in load_config_file(config_path):
            raise ValueError(
                f"Config '{config_path}' has no 'dataset' block, so"
                " --max-num-samples and --shuffle-seed have nothing to apply"
                " to. A pipeline whose first step generates its own data sizes"
                " it with 'num_samples' on that step:"
                " --set steps.0.num_samples=N."
            )

    for assignment in settings or []:
        key, value = _parse_set_override(assignment)
        if key in overrides:
            raise ValueError(
                f"'{key}' is set twice on the command line. Give it once,"
                " either through its own flag or through --set."
            )
        overrides[key] = value

    return overrides


def main(args: list[str] | None = None) -> None:
    """Run an annotation pipeline described by a JSON or YAML config file.

    Args:
        args: Optional argument list; defaults to ``sys.argv``.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="llm-annotate",
        description=(
            "Run a single- or multi-step LLM annotation pipeline from a JSON"
            " or YAML config file. Paths inside the config resolve relative"
            " to the config file itself."
        ),
    )
    parser.add_argument(
        "config", type=Path, help="Path to a JSON or YAML config file."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the config's 'output_dir'. Resolved against the"
        " current directory, not the config file.",
    )
    parser.add_argument(
        "--hub-id",
        default=None,
        help="Override the Hub dataset id for the final push.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override the log level (DEBUG/INFO/WARNING/ERROR/CRITICAL).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Delete the directories of the selected steps, including every"
        " finished generation in them, and run those steps from scratch. Not"
        " needed for a higher 'max_num_samples': a plain re-run annotates only"
        " the new rows.",
    )
    parser.add_argument(
        "--max-num-samples",
        type=int,
        default=None,
        help="Override 'dataset.max_num_samples'. Raising it on a finished"
        " run and re-running annotates only the rows that are new.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Override 'dataset.shuffle_seed', the seed the source is"
        " shuffled with before it is capped.",
    )
    parser.add_argument(
        "--set",
        dest="settings",
        metavar="KEY=VALUE",
        action="append",
        default=None,
        help="Override any config key; repeat for more than one. A dotted key"
        " reaches a nested value and an integer segment indexes a list, as in"
        " --set client.options.temperature=0.2 or"
        " --set steps.0.client.batch_size=8. The value is read as YAML.",
    )
    parser.add_argument(
        "--steps",
        default=None,
        help="Comma-separated names of the steps to run, which must be"
        " contiguous. Earlier steps must already have finished; their saved"
        " output is used as the input. Defaults to the whole pipeline.",
    )
    parser.add_argument(
        "--retry-errors",
        nargs="*",
        metavar="ERROR_TYPE",
        default=None,
        help="Annotate the rows again that finished with an error, in the"
        " selected steps and in the steps that read them. Without a value"
        " every errored row is redone; with values only those error types,"
        " as in --retry-errors ConnectError APITimeoutError. The end-of-run"
        " summary lists the error types of a run.",
    )
    parser.add_argument(
        "--hosts-file",
        type=Path,
        default=None,
        help="File with one vLLM server base URL per line, applied to the"
        " selected step that runs on vLLM. Use it to point a step at servers"
        " whose addresses are only known at run time.",
    )
    parser.add_argument(
        "--url-glob",
        default=None,
        help="Glob matching files that each hold one vLLM server base URL,"
        " applied to the same steps as --hosts-file. Unlike a file of URLs it"
        " is re-read while the run continues, so servers that become ready"
        " later join the pool.",
    )
    parser.add_argument(
        "--serve-args",
        metavar="STEP",
        default=None,
        help="Print the 'vllm serve' arguments for one step's servers, one per"
        " line, and exit. This is how a job submitter starts servers whose"
        " profile lives in the config.",
    )
    parser.add_argument(
        "--describe-steps",
        action="store_true",
        help="Print one JSON object per step describing what it needs to run"
        " (kind, provider, model, pool size) and how much work it keeps in"
        " flight (batch_size, max_concurrent_batches_per_client, the"
        " effective queue_size, and the max_requests_per_server that a vLLM"
        " server's --max-num-seqs has to cover), then exit without"
        " annotating anything.",
    )
    parsed = parser.parse_args(args)

    selected = (
        [name.strip() for name in parsed.steps.split(",") if name.strip()]
        if parsed.steps
        else None
    )

    overrides = _cli_overrides(
        config_path=parsed.config,
        output_dir=parsed.output_dir,
        hub_id=parsed.hub_id,
        log_level=parsed.log_level,
        overwrite=parsed.overwrite,
        max_num_samples=parsed.max_num_samples,
        shuffle_seed=parsed.shuffle_seed,
        settings=parsed.settings,
    )

    config = load_pipeline_config(
        parsed.config,
        overrides=overrides,
        step_client_overrides=_pool_source_override(
            parsed.config, parsed.hosts_file, parsed.url_glob, selected
        ),
    )
    configure_logging(level=config.log_level)
    if overrides:
        applied = ", ".join(f"{k}={v}" for k, v in overrides.items())
        LOGGER.info(f"Config overrides from the command line: {applied}.")

    if parsed.describe_steps:
        for described in config.describe_steps():
            print(json.dumps(described))
        return

    if parsed.serve_args:
        for arg in _serve_args(config, parsed.serve_args):
            print(arg)
        return

    run_pipeline(
        config,
        selected=selected,
        retry_errors=(
            False
            if parsed.retry_errors is None
            else parsed.retry_errors or True
        ),
    )


__all__ = ["main", "run_pipeline"]


if __name__ == "__main__":
    main()
