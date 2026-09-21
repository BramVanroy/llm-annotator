# LLM Annotator

LLM Annotator runs a large language model over a dataset and writes the answers
back as columns. A run is resumable: every finished sample is appended to a
JSONL progress file, so a job that crashes, times out or is preempted continues
where it stopped instead of starting over.

Four providers share one interface:

- `VLLMOfflineClient` for in-process vLLM (`vllm_offline`).
- `VLLMOnlineClient` for a vLLM server, one or many (`vllm_online`).
- `OpenAIClient` for OpenAI-compatible APIs (`openai`).
- `ClaudeClient` for the Anthropic API (`claude`).

[Choosing a provider](choosing-a-provider.md) says which one fits the hardware
you have. [Provider setup](provider-info.md) lists the extras, the
authentication variables and the vLLM tuning knobs.

## Install

--8<-- "README.md:install"

The online vLLM client speaks the OpenAI protocol, so it takes the `openai`
extra rather than the much heavier `vllm` one, which is only needed where the
model weights are loaded. See [Provider setup](provider-info.md).

## Quickstart

Describe the run in one YAML (or JSON) file and start it. No Python needed.

--8<-- "README.md:config-quickstart"

A config can hold several steps that run in order, each annotating what the
previous one produced. [Annotating from a config file](pipeline.md) is the key
reference, and `examples/pipeline-qa/` in the repository is a complete two-step
example.

The same run from Python:

--8<-- "README.md:one-step"

## Where to go next

- [Choosing a provider](choosing-a-provider.md): one GPU, many GPUs or none.
- [Annotating from a config file](pipeline.md): every config key, multi-step
  pipelines, the `llm-annotate` command line.
- [Python API guide](python-api.md): the staged `prepare_data` plus
  `run_annotation` workflow, several tasks in one output directory, errors and
  retries, a pool of vLLM servers.
- [Growing a run](growing-a-run.md): resume a run, raise the sample cap, edit a
  prompt mid-run.
- [SLURM](slurm.md): submit a config as one job chain per step.
- [Troubleshooting](troubleshooting.md): what an error message means and what
  to do about it.
- [Migrating from 0.16](migration.md): everything that changed since the last
  release.

The "User API" section of the navigation is generated from the docstrings of
what the guides above use; "Internals" covers the modules those build on.
