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

[`main`][llm_annotator.pipeline.main] is the ``llm-annotate`` console entry
point.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from datasets import Dataset

from llm_annotator.annotator import Annotator, VLLMQueueAnnotator
from llm_annotator.config import (
    ClientConfig,
    PipelineConfig,
    StepConfig,
    load_pipeline_config,
)
from llm_annotator.logging_utils import configure_logging, get_logger


LOGGER = get_logger("pipeline")

STEP_OUTPUT_SUBDIR = "output"
"""Name of the subdirectory holding a finished step's dataset.

Its presence is what marks a step as done.
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


def _generate_dataset(step: StepConfig, root: Path) -> tuple[Dataset, str]:
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
        The one-column prompt dataset and the prompt template to render it
        with.

    Raises:
        ValueError: If an explicit template does not contain ``{prompt}``.
    """
    prompts = step.resolved_prompts(root)
    template = step.resolved_prompt(root)
    if template is None:
        template = "{prompt}"
    elif "{prompt}" not in template:
        raise ValueError(
            f"Step '{step.name}': the template of a 'generate' step must"
            " contain the '{prompt}' placeholder, which is filled in with each"
            " entry of 'prompts'."
        )

    LOGGER.info(f"Step '{step.name}': generating {len(prompts):,} sample(s).")
    return Dataset.from_dict({"prompt": prompts}), template


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
        dataset: The incoming dataset, or ``None`` when the first step should
            load it from a Hub id or builder name itself.
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
    }

    if step.type == "generate":
        prompt_dataset, template = _generate_dataset(step, root)
        kwargs["dataset"] = prompt_dataset
        kwargs["prompt_template"] = template
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

    if is_first and config.dataset is not None:
        kwargs["max_num_samples"] = config.dataset.max_num_samples
        kwargs["shuffle_seed"] = config.dataset.shuffle_seed

    return annotator.annotate_dataset(**kwargs)


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
    config: PipelineConfig, selected: Sequence[str] | None = None
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
    snapshot = config.output_dir / "pipeline.json"
    snapshot.write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )

    dataset: Dataset | None = _load_input_dataset(config)
    annotator: Annotator | None = None
    active_client_key: str | None = None
    runs_last_step = chosen.stop >= len(config.steps)

    try:
        for index, step in enumerate(config.steps):
            if index >= chosen.stop:
                break

            step_dir = config.step_dir(index)
            step_output = step_dir / STEP_OUTPUT_SUBDIR
            label = f"Step {index + 1}/{len(config.steps)} '{step.name}'"

            # Only wipe what this run is actually going to redo; a step outside
            # the selection is a dependency, not something to throw away.
            if config.overwrite and index in chosen and step_dir.is_dir():
                LOGGER.info(f"{label}: removing '{step_dir}' (overwrite).")
                shutil.rmtree(step_dir, ignore_errors=True)

            if _is_complete(step_output):
                LOGGER.info(
                    f"{label}: already finished, loading its result from"
                    f" '{step_output}'."
                )
                dataset = Dataset.load_from_disk(str(step_output))
                continue

            if index not in chosen:
                raise ValueError(
                    f"Step '{step.name}' has not run yet, so there is no input"
                    f" for '{config.steps[chosen.start].name}'. Run it first,"
                    " or select it too."
                )

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
                # to change without rebuilding anything. `queue_size` is in
                # the same category and is deliberately absent from
                # `cache_key`, so it has to be refreshed here too or the step
                # would silently run with the previous step's value.
                annotator.batch_size = client_config.batch_size
                annotator.num_proc = client_config.num_proc
                if isinstance(annotator, VLLMQueueAnnotator):
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
                dataset,
                step,
                step.resolved_task_prefix(),
                annotator.num_proc,
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

    final_dir = config.output_dir / "final"
    dataset.save_to_disk(str(final_dir))
    LOGGER.info(
        f"Pipeline finished: {len(dataset):,} sample(s) in '{final_dir}'."
    )

    if config.hub_id:
        LOGGER.info(f"Pushing the final dataset to '{config.hub_id}'.")
        dataset.push_to_hub(config.hub_id, private=True)

    return dataset


def _hosts_file_override(
    config_path: Path,
    hosts_file: Path | None,
    selected: Sequence[str] | None,
) -> dict[str, dict[str, Any]] | None:
    """Work out which step a ``--hosts-file`` argument belongs to.

    The file is attached to the selected steps that run on vLLM, never to the
    pipeline as a whole: a step on a hosted provider must not inherit a pool it
    cannot use, and a set of servers only ever serves one model.

    Args:
        config_path: Path to the config file, loaded once to see the providers.
        hosts_file: The path given on the command line, or ``None``.
        selected: Step names being run, or ``None`` for all of them.

    Returns:
        A per-step client override mapping, or ``None`` when no hosts file was
        given.

    Raises:
        ValueError: If no selected step runs on vLLM, or if the selected vLLM
            steps disagree about which model they want, since one pool of
            servers can only serve one of them.
    """
    if hosts_file is None:
        return None

    # A command-line path means what the shell means by it. Config paths
    # resolve against the config file instead, so pin it down before it is
    # handed over as though it had been written in the config.
    resolved = str(hosts_file.expanduser().resolve())

    probe = load_pipeline_config(config_path)
    targets = {}
    for index in _resolve_selection(probe, selected):
        step = probe.steps[index]
        client = probe.step_client(step)
        if client.provider == "vllm_online":
            targets[step.name] = client.model

    if not targets:
        raise ValueError(
            "--hosts-file points at vLLM servers, but none of the steps being"
            f" run ({selected or 'all'}) uses provider 'vllm_online'."
        )

    models = {model for model in targets.values() if model is not None}
    if len(models) > 1:
        raise ValueError(
            f"Steps {sorted(targets)} want different models {sorted(models)},"
            " but one set of vLLM servers serves a single model. Run them as"
            " separate --steps invocations, each against its own servers."
        )

    return {name: {"hosts_file": resolved} for name in targets}


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
        help="Discard existing step directories instead of resuming them.",
    )
    parser.add_argument(
        "--steps",
        default=None,
        help="Comma-separated names of the steps to run, which must be"
        " contiguous. Earlier steps must already have finished; their saved"
        " output is used as the input. Defaults to the whole pipeline.",
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
        " (kind, provider, model, pool size) and exit without annotating"
        " anything.",
    )
    parsed = parser.parse_args(args)

    selected = (
        [name.strip() for name in parsed.steps.split(",") if name.strip()]
        if parsed.steps
        else None
    )

    output_dir_override = (
        Path(parsed.output_dir).expanduser().resolve()
        if parsed.output_dir is not None
        else None
    )
    overrides = {
        key: value
        for key, value in (
            ("output_dir", output_dir_override),
            ("hub_id", parsed.hub_id),
            ("log_level", parsed.log_level),
            ("overwrite", parsed.overwrite),
        )
        if value is not None
    }

    config = load_pipeline_config(
        parsed.config,
        overrides=overrides,
        step_client_overrides=_hosts_file_override(
            parsed.config, parsed.hosts_file, selected
        ),
    )
    configure_logging(level=config.log_level)

    if parsed.describe_steps:
        for described in config.describe_steps():
            print(json.dumps(described))
        return

    if parsed.serve_args:
        for arg in _serve_args(config, parsed.serve_args):
            print(arg)
        return

    run_pipeline(config, selected=selected)


__all__ = ["main", "run_pipeline"]


if __name__ == "__main__":
    main()
