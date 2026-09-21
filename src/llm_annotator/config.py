"""Declarative configuration for the config-driven annotation pipeline.

As much as possible is validated at config validation time but functional elements
like preprocess/postprocess/validation functions are not configurable here. If you
need such functionality, you need to write your own Python script that uses the
library's API directly.
"""

from __future__ import annotations

import dataclasses
import glob
import inspect
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any, Literal, get_args

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from llm_annotator.annotator import (
    DEFAULT_CPU_COUNT,
    QUEUE_BATCHES_PER_SLOT,
    Annotator,
    VLLMQueueAnnotator,
)
from llm_annotator.clients.base import Client, ProviderRuntimeOptions
from llm_annotator.clients.vllm_online_client import (
    VLLMOnlineClient,
    server_is_healthy,
)
from llm_annotator.logging_utils import get_logger


LOGGER = get_logger("config")

ProviderName = Literal["openai", "claude", "vllm_online", "vllm_offline"]
StepType = Literal["annotate", "generate"]

StepKind = Literal["vllm_pool", "vllm_online", "vllm_offline", "api"]
"""What a step needs in order to run, as reported by ``--describe-steps``."""

DEFAULT_MAX_CONCURRENT_BATCHES = 4
"""Simultaneous batch requests per vLLM server unless a step says otherwise."""


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Read a JSON or YAML config file into a plain dictionary.

    The format is chosen from the file suffix: ``.json`` is parsed as JSON,
    ``.yaml`` and ``.yml`` as YAML. YAML is a superset of JSON, so an unknown
    suffix is parsed as YAML.

    Args:
        path: Path to the config file.

    Returns:
        The decoded mapping.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file does not decode to a mapping.
    """
    pfin = Path(path).expanduser()
    if not pfin.is_file():
        raise FileNotFoundError(f"Config file '{pfin}' does not exist.")

    text = pfin.read_text(encoding="utf-8")
    if pfin.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ValueError(
            f"Config file '{pfin}' must contain a mapping at the top level,"
            f" got {type(data).__name__}."
        )
    return data


def _list_position(values: list[Any], segment: str, key: str) -> int:
    """Turn one segment of an override key into an index into a config list.

    Args:
        values: The list the segment addresses.
        segment: The segment as written in the override key.
        key: The whole override key, for the error message.

    Returns:
        The index the segment names.

    Raises:
        ValueError: If the segment is not an integer, or names a position the
            list does not have.
    """
    try:
        position = int(segment)
    except ValueError:
        raise ValueError(
            f"Override '{key}': '{segment}' is not a list index. A list in a"
            " config is addressed by position, as in 'steps.0.name'."
        ) from None
    if not -len(values) <= position < len(values):
        raise ValueError(
            f"Override '{key}': position {position} is out of range for a"
            f" list of {len(values)}."
        )
    return position


def _apply_override(data: dict[str, Any], key: str, value: Any) -> None:
    """Set one possibly dotted key inside a decoded config mapping.

    A dot descends into a nested block, and an integer segment indexes a list,
    so ``dataset.max_num_samples`` and ``steps.0.client.batch_size`` both name
    a single value. A missing intermediate block is created.

    Args:
        data: The decoded config mapping, modified in place.
        key: The key to set, with dots between path segments.
        value: The value to store.

    Raises:
        ValueError: If a segment runs into a value that is neither a mapping
            nor a list, or into a list position that does not exist.

    Examples:
        >>> data = {"dataset": {"name": "imdb"}, "steps": [{"name": "rate"}]}
        >>> _apply_override(data, "dataset.max_num_samples", 2000)
        >>> _apply_override(data, "steps.0.client.batch_size", 8)
        >>> data["dataset"]["max_num_samples"], data["steps"][0]["client"]
        (2000, {'batch_size': 8})
    """
    *parents, leaf = key.split(".")
    node: Any = data
    for segment in parents:
        if isinstance(node, list):
            node = node[_list_position(node, segment, key)]
        elif isinstance(node, dict):
            if node.get(segment) is None:
                node[segment] = {}
            node = node[segment]
        else:
            raise ValueError(
                f"Override '{key}': '{segment}' cannot be set inside a"
                f" {type(node).__name__}."
            )

    if isinstance(node, list):
        node[_list_position(node, leaf, key)] = value
    elif isinstance(node, dict):
        node[leaf] = value
    else:
        raise ValueError(
            f"Override '{key}': '{leaf}' cannot be set inside a"
            f" {type(node).__name__}."
        )


def _options_class(provider: ProviderName) -> type[ProviderRuntimeOptions]:
    """Get the runtime-options dataclass belonging to a provider.

    Provider SDKs are imported lazily inside the client modules, so importing
    the options classes here does not pull in ``openai``, ``anthropic`` or
    ``vllm``.

    Args:
        provider: Canonical provider name.

    Returns:
        The matching ``*RuntimeOptions`` dataclass.

    Raises:
        ValueError: If the provider is unknown.
    """
    if provider == "openai":
        from llm_annotator.clients.openai_client import OpenAIRuntimeOptions

        return OpenAIRuntimeOptions
    if provider == "claude":
        from llm_annotator.clients.claude_client import ClaudeRuntimeOptions

        return ClaudeRuntimeOptions
    if provider == "vllm_online":
        from llm_annotator.clients.vllm_online_client import (
            VLLMOnlineRuntimeOptions,
        )

        return VLLMOnlineRuntimeOptions
    if provider == "vllm_offline":
        from llm_annotator.clients.vllm_offline_client import (
            VLLMOfflineRuntimeOptions,
        )

        return VLLMOfflineRuntimeOptions
    raise ValueError(f"Unknown provider '{provider}'.")


def _provider_bound_keys(provider: ProviderName) -> set[str]:
    """Name the client keys that a step must not inherit across a switch.

    These are the keys whose value only makes sense for the provider it was
    written for: ``init`` names constructor arguments, ``options`` names fields
    of a runtime-options dataclass, and the rest describe a vLLM engine or a
    pool of vLLM servers.

    Args:
        provider: The provider the step switched to.

    Returns:
        The names of the keys that provider cannot take over.

    Examples:
        >>> sorted(_provider_bound_keys("vllm_offline"))
        ['base_urls', 'hosts_file', 'init', 'max_concurrent_batches_per_client', 'options', 'pool', 'queue_size', 'url_glob', 'wait_for_servers']
        >>> sorted(_provider_bound_keys("vllm_online"))
        ['init', 'options']
    """
    keys = {"init", "options"}
    if not provider.startswith("vllm"):
        keys.add("engine")
    if provider != "vllm_online":
        keys |= {
            "base_urls",
            "hosts_file",
            "url_glob",
            "pool",
            "queue_size",
            "max_concurrent_batches_per_client",
            "wait_for_servers",
        }
    return keys


def _client_class(provider: ProviderName) -> type[Client[Any]]:
    """Get the client class belonging to a provider.

    Args:
        provider: Canonical provider name.

    Returns:
        The matching [`Client`][llm_annotator.clients.base.Client] subclass.

    Raises:
        ImportError: If the provider's optional extra is not installed, with a
            message naming the extra to install.
        ValueError: If the provider is unknown.
    """
    extras = {
        "openai": "openai",
        "claude": "anthropic",
        "vllm_online": "openai",
        "vllm_offline": "vllm",
    }
    try:
        if provider == "openai":
            from llm_annotator.clients.openai_client import OpenAIClient

            return OpenAIClient
        elif provider == "claude":
            from llm_annotator.clients.claude_client import ClaudeClient

            return ClaudeClient
        elif provider == "vllm_online":
            from llm_annotator.clients.vllm_online_client import (
                VLLMOnlineClient,
            )

            return VLLMOnlineClient
        elif provider == "vllm_offline":
            from llm_annotator.clients.vllm_offline_client import (
                VLLMOfflineClient,
            )

            return VLLMOfflineClient
    except ImportError as exc:
        raise ImportError(
            f"Provider '{provider}' needs its optional dependency. Install it"
            f" with `uv sync --extra {extras[provider]}`."
        ) from exc
    raise ValueError(f"Unknown provider '{provider}'.")


def wait_for_servers(
    base_urls: list[str], timeout: float, min_servers: int = 1
) -> list[str]:
    """Block until the requested number of vLLM servers answer ``/health``.

    Args:
        base_urls: vLLM base URLs (each ending in ``/v1``).
        timeout: Maximum number of seconds to wait for the pool as a whole.
        min_servers: Number of ready servers required to continue.

    Raises:
        TimeoutError: If fewer than ``min_servers`` are reachable after
            ``timeout``.
    """
    pending = list(dict.fromkeys(base_urls))
    if not 1 <= min_servers <= len(pending):
        raise ValueError(
            f"'min_servers' must be between 1 and {len(pending)}, got"
            f" {min_servers}."
        )

    ready: list[str] = []
    deadline = time.monotonic() + timeout
    while pending:
        remaining = deadline - time.monotonic()
        if remaining < 0:
            break
        with ThreadPoolExecutor(max_workers=len(pending)) as pool:
            results = zip(
                pending,
                pool.map(
                    lambda url: server_is_healthy(
                        url, min(5, max(remaining, 0))
                    ),
                    pending,
                ),
                strict=True,
            )
            newly_ready = []
            for url, is_ready in results:
                if is_ready:
                    ready.append(url)
                    newly_ready.append(url)
        pending = [url for url in pending if url not in newly_ready]
        if len(ready) >= min_servers:
            return ready
        time.sleep(min(5, max(0, deadline - time.monotonic())))

    raise TimeoutError(
        f"Only {len(ready)} of {min_servers} required vLLM server(s) became"
        f" ready within {timeout:g}s."
    )


class _StrictBase(BaseModel):
    """Base model that rejects unknown keys so config typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class DatasetConfig(_StrictBase):
    """Source dataset for the first step of a pipeline.

    Exactly one of ``name`` or ``path`` is used: ``name`` is handed to
    ``datasets.load_dataset`` (a Hub id or a builder name, e.g. ``"json"`` to
    load a local directory of JSON Lines files via ``data_dir`` or
    ``data_files``), ``path`` loads a dataset previously written with
    ``save_to_disk``. ``data_dir`` and ``data_files`` only apply to ``name``.

    Attributes:
        name: Hub dataset id or builder name.
        path: Local directory holding a ``save_to_disk`` dataset.
        config: Dataset configuration name.
        split: Split to load. Required when the dataset has several splits.
        data_dir: Data directory for local/loader datasets.
        data_files: Specific file(s) for local/loader datasets, as a single
            path, a list of paths, or a mapping of split name to path(s).
        max_num_samples: Truncate the dataset to this many samples. A re-run
            with a higher value annotates only the new samples, as long as
            ``shuffle_seed`` and the source stay the same.
        shuffle_seed: Shuffle the dataset with this seed before truncating.
    """

    name: str | None = None
    path: Path | None = None
    config: str | None = None
    split: str | None = None
    data_dir: str | None = None
    data_files: str | list[str] | dict[str, str | list[str]] | None = None
    max_num_samples: int | None = None
    shuffle_seed: int | None = None

    @model_validator(mode="after")
    def _one_source(self) -> "DatasetConfig":
        """Require exactly one of ``name`` and ``path``."""
        if (self.name is None) == (self.path is None):
            raise ValueError(
                "Provide exactly one of 'name' or 'path' in the dataset block."
            )
        if self.path is not None and (
            self.data_dir is not None or self.data_files is not None
        ):
            raise ValueError(
                "'data_dir' and 'data_files' only apply to 'name', not 'path'."
            )
        return self


class EngineConfig(_StrictBase):
    """How one vLLM engine is built, for either vLLM provider.

    The same field names mean the same thing whether the model is loaded in
    process or served by ``vllm serve``; only the transport differs. A
    ``vllm_offline`` step turns this block into ``vllm.LLM`` keyword arguments,
    and a ``vllm_online`` step whose servers still have to be started turns it
    into ``vllm serve`` flags, which is what
    ``llm-annotate --serve-args <step>`` prints for a job submitter.

    That shared spelling is the point: it is why a step states its GPU count
    once, in ``tensor_parallel_size``, instead of once for the allocation and
    once for vLLM.

    Attributes:
        tensor_parallel_size: GPUs one engine shards its weights over. For a
            served step this is also how many GPUs its job asks for.
        max_model_len: Maximum total sequence length, prompt plus completion.
        gpu_memory_utilization: Fraction of each GPU's memory vLLM may claim.
        max_num_seqs: Sequences the engine runs concurrently.
        max_num_batched_tokens: Token budget of a single forward pass.
        enforce_eager: Disable CUDA graphs and run eagerly.
        quantization: Quantization method, e.g. ``"fp8"`` or ``"awq"``.
        enable_prefix_caching: Reuse the KV cache of a shared prompt prefix.
        enable_chunked_prefill: Split long prefills to bound peak memory.
        reasoning_parser: Name of the vLLM reasoning parser that separates a
            thinking model's trace from its answer, e.g. ``"qwen3"`` or
            ``"deepseek_r1"``. Set it and the step writes ``{prefix}reasoning``
            instead of leaving the trace inline in ``{prefix}response``. A
            served step renders it as ``--reasoning-parser``; an offline step
            hands it to the client, which parses the trace itself with vLLM's
            own parser, since ``vllm.LLM`` does no such splitting.
        speculative_config: vLLM speculative-decoding configuration.
        extra: Any other vLLM engine argument, by its Python name. Rendered as
            ``--kebab-case`` for ``vllm serve`` and passed verbatim to
            ``vllm.LLM``, so one spelling covers both.
    """

    tensor_parallel_size: int = Field(default=1, ge=1)
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    enforce_eager: bool | None = None
    quantization: str | None = None
    enable_prefix_caching: bool | None = None
    enable_chunked_prefill: bool | None = None
    reasoning_parser: str | None = None
    speculative_config: dict[str, Any] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def as_llm_kwargs(self) -> dict[str, Any]:
        """Build the keyword arguments for an in-process ``vllm.LLM``.

        Unset fields are dropped rather than passed as ``None``, so vLLM's own
        defaults apply to anything the config does not mention.
        ``reasoning_parser`` rides along here because
        [`build_client`][llm_annotator.config.ClientConfig.build_client] feeds
        this dict to the offline client's constructor, which keeps it rather
        than forwarding it: ``vllm.LLM`` does not take it.

        Returns:
            Keyword arguments, with ``extra`` merged in.

        Examples:
            >>> EngineConfig(max_model_len=4096).as_llm_kwargs()
            {'tensor_parallel_size': 1, 'max_model_len': 4096}
        """
        kwargs = self.model_dump(exclude={"extra"}, exclude_none=True)
        kwargs.update(self.extra)
        return kwargs

    def as_serve_args(self) -> list[str]:
        """Build the ``vllm serve`` flags for one server.

        Returned as an argument list rather than a string so a value may
        contain spaces; ``--speculative-config`` takes a JSON object, and
        shell word-splitting cannot carry one.

        Returns:
            Flags for ``vllm serve``, without the model or ``--host``/``--port``.

        Examples:
            >>> EngineConfig(max_model_len=4096).as_serve_args()
            ['--tensor-parallel-size', '1', '--max-model-len', '4096']
            >>> EngineConfig(enforce_eager=True).as_serve_args()
            ['--tensor-parallel-size', '1', '--enforce-eager']
        """
        args: list[str] = []
        for name, value in self.as_llm_kwargs().items():
            flag = f"--{name.replace('_', '-')}"
            if isinstance(value, bool):
                # vLLM's argparse spells a disabled boolean `--no-<flag>`.
                args.append(
                    flag if value else f"--no-{name.replace('_', '-')}"
                )
            elif isinstance(value, (dict, list)):
                args.extend([flag, json.dumps(value, separators=(",", ":"))])
            else:
                args.extend([flag, str(value)])
        return args


class PoolConfig(_StrictBase):
    """How many vLLM servers a step wants started for it.

    How big each one is lives in
    [`EngineConfig`][llm_annotator.config.EngineConfig] instead, because that
    is a property of the engine rather than of the pool, and a ``vllm_offline``
    step needs it without wanting a pool at all.

    Both keys are reported by ``llm-annotate --describe-steps``, so a job
    submitter can size the servers it starts and knows how many of them the
    step needs before it can begin. ``min_servers`` is what a pooled run acts
    on while it runs: it waits for that many servers to answer ``/health``,
    and the servers that become ready later join the pool. ``servers`` is how
    many the step counts on at full size, which is what the floor on
    ``queue_size`` and the reported concurrency are computed from. A step that
    talks to servers someone else started (``base_urls``, ``hosts_file``,
    ``url_glob``) needs neither key, since ``base_urls`` already counts them.

    Attributes:
        servers: Number of vLLM server processes to run for this step.
        min_servers: Number of ready servers required before annotation begins.
            Defaults to one, so a step starts on the first server that answers
            and grows as the rest arrive.
    """

    servers: int = Field(default=1, ge=1)
    min_servers: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _minimum_does_not_exceed_pool(self) -> "PoolConfig":
        """Reject a readiness threshold larger than the requested pool."""
        if self.min_servers > self.servers:
            raise ValueError(
                "'pool.min_servers' cannot exceed 'pool.servers'."
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def _moved_gpus_per_server(cls, data: Any) -> Any:
        """Point the old ``gpus_per_vllm_server`` key at its new home."""
        if isinstance(data, dict) and "gpus_per_vllm_server" in data:
            raise ValueError(
                "'pool.gpus_per_vllm_server' moved to"
                " 'engine.tensor_parallel_size', which both vLLM providers"
                " read, so a step states its GPU count once."
            )
        return data


class ClientConfig(_StrictBase):
    """Provider, model and execution settings for one step.

    ``init`` is forwarded verbatim to the client constructor and ``options`` to
    the provider's ``*RuntimeOptions`` dataclass, so every provider-specific
    knob is reachable without this class having to enumerate them.

    Supplying ``base_urls``, ``hosts_file`` or ``url_glob`` (``vllm_online``
    only) turns the step into a multi-server run backed by
    [`VLLMQueueAnnotator`][llm_annotator.annotator.VLLMQueueAnnotator].

    Attributes:
        provider: Provider name, spelled exactly ``openai``, ``claude``,
            ``vllm_online`` or ``vllm_offline``. No other spelling is
            accepted, so a typo cannot silently pick a different backend.
        model: Model identifier. Optional only for ``vllm_online``, which can
            ask the server which model it serves.
        init: Extra keyword arguments for the client constructor.
        options: Fields of the provider's runtime-options dataclass.
        batch_size: Samples per inference batch.
        num_proc: Processes used for dataset preprocessing. Use ``null`` to
            disable multiprocessing.
        base_urls: vLLM server base URLs, for a multi-server pool.
        hosts_file: File with one vLLM base URL per line.
        url_glob: Glob matching files that each hold one vLLM base URL. It may
            be absolute, which is what a job scheduler writing into a scratch
            directory needs.
        queue_size: Batches kept in flight across the pool. ``null`` derives
            it from the pool, at four batches per concurrent request slot. A
            pool runs ``servers`` times ``max_concurrent_batches_per_client``
            requests at once, and a queue below that number would leave
            servers idle, so it is rejected. Sizes a pool, so it is accepted
            for ``vllm_online`` only.
        max_concurrent_batches_per_client: Maximum simultaneous batch requests
            sent to each vLLM server. Independent of ``batch_size``: each of
            these requests carries ``batch_size`` samples, so one server holds
            up to their product. Sizes a pool, so it is accepted for
            ``vllm_online`` only.
        wait_for_servers: Seconds to wait for the ``/health`` endpoints of
            ``pool.min_servers`` servers, or of every server the pool source
            names when it names fewer, before starting. ``0`` disables the
            check.
        engine: How this step's vLLM engine is built. Applies to both vLLM
            providers; rejected for the hosted ones.
        pool: How many servers this step wants. Only meaningful for
            ``vllm_online`` steps whose servers are started for them.
        gen_kwargs: Extra request parameters merged over ``options`` on every
            generation call, for anything the options dataclass does not name.
    """

    provider: ProviderName = "vllm_offline"
    model: str | None = None
    init: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    batch_size: int = Field(default=256, ge=1)
    num_proc: int | None = DEFAULT_CPU_COUNT
    base_urls: list[str] = Field(default_factory=list)
    hosts_file: Path | None = None
    url_glob: str | None = None
    queue_size: int | None = Field(default=None, ge=1)
    max_concurrent_batches_per_client: int = Field(
        default=DEFAULT_MAX_CONCURRENT_BATCHES, ge=1
    )
    wait_for_servers: float = 60.0
    engine: EngineConfig = Field(default_factory=EngineConfig)
    pool: PoolConfig = Field(default_factory=PoolConfig)
    gen_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider", mode="before")
    @classmethod
    def _validate_provider(cls, value: Any) -> Any:
        """Reject provider spellings other than the canonical names."""
        valid = get_args(ProviderName)
        if isinstance(value, str) and value not in valid:
            raise ValueError(
                f"Unknown provider '{value}'. Choose one of {sorted(valid)}."
            )
        return value

    @model_validator(mode="after")
    def _check_provider_combination(self) -> "ClientConfig":
        """Validate provider-specific requirements and options keys."""
        pool_keys = [
            name
            for name, value in (
                ("base_urls", self.base_urls),
                ("hosts_file", self.hosts_file),
                ("url_glob", self.url_glob),
            )
            if value
        ]
        if len(pool_keys) > 1:
            raise ValueError(
                "Provide at most one of 'base_urls', 'hosts_file' or"
                f" 'url_glob', got {pool_keys}."
            )
        if pool_keys and self.provider != "vllm_online":
            raise ValueError(
                f"'{pool_keys[0]}' describes a pool of vLLM servers and needs"
                f" provider 'vllm_online', not '{self.provider}'."
            )
        if self.model is None and self.provider != "vllm_online":
            raise ValueError(
                f"Provider '{self.provider}' needs an explicit 'model'."
            )

        self._check_pool_concurrency()

        engine_fields = set(EngineConfig.model_fields)
        if (
            not self.provider.startswith("vllm")
            and self.engine != EngineConfig()
        ):
            raise ValueError(
                f"'engine' configures a vLLM engine, so provider"
                f" '{self.provider}' has no use for it. Hosted providers are"
                " configured with 'init' and 'options'."
            )

        # `init` and `engine` would otherwise both be able to set
        # `max_model_len` and friends, which is the duplication `engine`
        # exists to remove.
        misplaced = sorted(engine_fields & set(self.init))
        if misplaced:
            raise ValueError(
                f"Move {misplaced} from 'init' to 'engine'. Engine settings"
                " live there so that both vLLM providers spell them the same"
                " way."
            )

        valid = {
            f.name for f in dataclasses.fields(_options_class(self.provider))
        }
        unknown = sorted(set(self.options) - valid)
        if unknown:
            raise ValueError(
                f"Unknown 'options' for provider '{self.provider}':"
                f" {unknown}. Valid options are {sorted(valid)}."
            )

        self._check_init_keys()
        return self

    def _check_init_keys(self) -> None:
        """Compare ``init`` against the constructor of the provider's client.

        The client classes import their provider SDK inside their methods, so
        the signature is available without ``openai``, ``anthropic`` or
        ``vllm`` being installed.

        Raises:
            ValueError: If ``init`` names something the constructor does not
                take, or a value that the client block states elsewhere.
        """
        if "model" in self.init:
            raise ValueError(
                "'init' sets 'model', which the client block names itself."
                " Write it as 'model' next to 'provider'."
            )
        if self.is_pool() and "base_url" in self.init:
            raise ValueError(
                "'init' sets 'base_url', but each server of a pool gets its"
                " own. The pool's servers are named by 'base_urls',"
                " 'hosts_file' or 'url_glob'."
            )

        try:
            client_cls = _client_class(self.provider)
        except ImportError:
            # Without the client class there is no signature to compare
            # against; the constructor reports the key when the step runs.
            return

        parameters = inspect.signature(client_cls.__init__).parameters
        if any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
            return
        accepted = {
            name
            for name, parameter in parameters.items()
            # 'model' has its own key in the client block, so it is not
            # offered here as an 'init' key.
            if name not in {"self", "model"}
            and parameter.kind is not parameter.VAR_POSITIONAL
        }
        unknown_init = sorted(set(self.init) - accepted)
        if unknown_init:
            raise ValueError(
                f"Unknown 'init' keys for provider '{self.provider}':"
                f" {unknown_init}. {client_cls.__name__} takes"
                f" {sorted(accepted)}."
            )

    def _check_pool_concurrency(self) -> None:
        """Reject pool concurrency keys that are misplaced or too small.

        Raises:
            ValueError: If ``queue_size`` or
                ``max_concurrent_batches_per_client`` is set on a provider
                without a server pool, or if ``queue_size`` is below the
                number of requests the pool runs at once.
        """
        # A merged step client carries every key, defaults included, so a
        # value counts as set only when it differs from the default.
        pool_only = [
            name
            for name, is_set in (
                ("queue_size", self.queue_size is not None),
                (
                    "max_concurrent_batches_per_client",
                    self.max_concurrent_batches_per_client
                    != DEFAULT_MAX_CONCURRENT_BATCHES,
                ),
            )
            if is_set
        ]
        if pool_only and self.provider != "vllm_online":
            names = " and ".join(f"'{name}'" for name in pool_only)
            raise ValueError(
                f"{names} size a pool of vLLM servers, so provider"
                f" '{self.provider}' has no use for them: it sends one"
                " request at a time. Use 'batch_size' to change how much"
                " work that request carries."
            )

        if self.queue_size is None:
            return

        servers = self.configured_servers()
        minimum = servers * self.max_concurrent_batches_per_client
        if self.queue_size < minimum:
            raise ValueError(
                f"'queue_size' is {self.queue_size}, below the minimum of"
                f" {minimum} for this step: {servers} server(s) times"
                f" {self.max_concurrent_batches_per_client}"
                " concurrent request(s) each"
                " ('max_concurrent_batches_per_client'). A queue smaller than"
                " that leaves servers idle. Set 'queue_size' to at least"
                f" {minimum}, or remove it to keep"
                f" {QUEUE_BATCHES_PER_SLOT} batches queued per request slot"
                f" ({QUEUE_BATCHES_PER_SLOT * minimum})."
            )

    def configured_servers(self) -> int:
        """Count the vLLM servers this config accounts for.

        ``pool.servers`` is what a job submitter is asked to start; explicit
        ``base_urls`` are servers that already exist. A ``hosts_file`` or
        ``url_glob`` is not read here, since it may not exist yet when the
        config is validated, so such a step counts ``pool.servers`` alone.

        Returns:
            The larger of ``pool.servers`` and the number of distinct
            ``base_urls``.

        Examples:
            >>> ClientConfig(
            ...     provider="vllm_online", model="m"
            ... ).configured_servers()
            1
            >>> ClientConfig(
            ...     provider="vllm_online",
            ...     model="m",
            ...     base_urls=[
            ...         "http://node01:8000/v1",
            ...         "http://node02:8000/v1",
            ...     ],
            ... ).configured_servers()
            2
        """
        return max(self.pool.servers, len(dict.fromkeys(self.base_urls)))

    def concurrency(self) -> dict[str, int | None]:
        """Report how many requests and samples this step keeps in flight.

        This is what a user needs in order to size a vLLM server's
        ``max_num_seqs``, and it is reported by
        ``llm-annotate --describe-steps`` before any server job is submitted.
        The pool keys are ``None`` for a provider without a pool, which runs
        one batch at a time.

        Returns:
            A mapping with ``batch_size``,
            ``max_concurrent_batches_per_client``, the effective
            ``queue_size``, ``max_requests_per_server``
            (``max_concurrent_batches_per_client`` times ``batch_size``,
            which is what a server's ``max_num_seqs`` has to cover) and
            ``max_requests_in_flight`` (that number times
            [`configured_servers`][llm_annotator.config.ClientConfig.configured_servers]).

        Examples:
            >>> ClientConfig(provider="claude", model="m").concurrency()
            {'batch_size': 256, 'max_concurrent_batches_per_client': None, 'queue_size': None, 'max_requests_per_server': None, 'max_requests_in_flight': None}
            >>> ClientConfig(
            ...     provider="vllm_online",
            ...     model="m",
            ...     batch_size=64,
            ...     pool={"servers": 4},
            ... ).concurrency()
            {'batch_size': 64, 'max_concurrent_batches_per_client': 4, 'queue_size': 64, 'max_requests_per_server': 256, 'max_requests_in_flight': 1024}
        """
        if self.provider != "vllm_online":
            return {
                "batch_size": self.batch_size,
                "max_concurrent_batches_per_client": None,
                "queue_size": None,
                "max_requests_per_server": None,
                "max_requests_in_flight": None,
            }

        servers = self.configured_servers()
        slots = servers * self.max_concurrent_batches_per_client
        return {
            "batch_size": self.batch_size,
            "max_concurrent_batches_per_client": (
                self.max_concurrent_batches_per_client
            ),
            "queue_size": VLLMQueueAnnotator.resolve_queue_size(
                self.queue_size, slots
            ),
            "max_requests_per_server": (
                self.max_concurrent_batches_per_client * self.batch_size
            ),
            "max_requests_in_flight": slots * self.batch_size,
        }

    def is_pool(self) -> bool:
        """Whether this client describes a pool of vLLM servers.

        Returns:
            ``True`` when any of the pool discovery keys is set.
        """
        return bool(self.base_urls or self.hosts_file or self.url_glob)

    def kind(self) -> StepKind:
        """Classify what a step on this client needs in order to run.

        This is the whole taxonomy a job submitter needs, and it is derived
        rather than configured, so a config cannot disagree with itself about
        which resources a step wants. A kind names the *resources* a step
        needs, not its provider, which is why provider ``vllm_online`` can
        yield either ``vllm_pool`` or ``vllm_online``:

        ``vllm_pool``
            Provider ``vllm_online``, but no servers were named, so they still
            have to be started for it.
        ``vllm_online``
            Provider ``vllm_online`` pointed at servers that someone else
            started.
        ``vllm_offline``
            It loads the model in-process, so it needs GPUs wherever the
            annotation itself runs.
        ``api``
            A hosted provider; no accelerator at all.

        Returns:
            The step kind.

        Examples:
            >>> ClientConfig(provider="vllm_online", model="m").kind()
            'vllm_pool'
            >>> ClientConfig(
            ...     provider="vllm_online",
            ...     model="m",
            ...     base_urls=["http://node01:8000/v1"],
            ... ).kind()
            'vllm_online'
            >>> ClientConfig(provider="claude", model="m").kind()
            'api'
        """
        if self.provider == "vllm_offline":
            return "vllm_offline"
        if self.provider == "vllm_online":
            return "vllm_online" if self.is_pool() else "vllm_pool"
        return "api"

    def cache_key(self) -> str:
        """Build a key identifying the underlying client resources.

        Two steps whose keys match can share one live client, which matters
        because loading a vLLM model takes minutes. Only constructor-level
        settings appear here: ``options`` and ``gen_kwargs`` are per-request
        and are passed to ``batch_generate``, so they never require a rebuild.
        ``engine`` does, because it decides how the engine itself is built.

        Returns:
            A stable string key.
        """
        return json.dumps(
            {
                "provider": self.provider,
                "model": self.model,
                "init": self.init,
                "engine": self.engine.model_dump(),
                "base_urls": self.base_urls,
                "hosts_file": str(self.hosts_file)
                if self.hosts_file
                else None,
                "url_glob": self.url_glob,
            },
            sort_keys=True,
            default=str,
        )

    def resolve_base_urls(self, root: Path) -> list[str]:
        """Collect the pool's base URLs from whichever source was configured.

        Args:
            root: Directory that relative paths and globs resolve against.

        Returns:
            The base URLs, in file/glob order.

        Raises:
            ValueError: If the configured source yields no URL.
        """
        if self.base_urls:
            return list(self.base_urls)

        if self.hosts_file is not None:
            pfin = _resolve_path(self.hosts_file, root)
            if not pfin.is_file():
                raise ValueError(f"hosts_file '{pfin}' does not exist.")
            lines = pfin.read_text(encoding="utf-8").splitlines()
        else:
            # `Path.glob` rejects an absolute pattern outright, and a scheduler
            # that writes its pool directory somewhere central has no relative
            # path to offer, so absolute patterns go through `glob.glob`.
            pattern = str(self.url_glob)
            if Path(pattern).is_absolute():
                matches = sorted(Path(hit) for hit in glob.glob(pattern))
                searched = "the filesystem"
            else:
                matches = sorted(root.glob(pattern))
                searched = f"'{root}'"
            if not matches:
                raise ValueError(
                    f"url_glob '{self.url_glob}' matched no files under"
                    f" {searched}."
                )
            lines = []
            for match in matches:
                lines.extend(match.read_text(encoding="utf-8").splitlines())

        urls = [line.strip() for line in lines if line.strip()]
        if not urls:
            raise ValueError("No vLLM server URLs found for the client pool.")
        return urls

    def build_options(
        self, output_schema: dict[str, Any] | None = None
    ) -> ProviderRuntimeOptions:
        """Instantiate the provider's runtime-options dataclass.

        Args:
            output_schema: Optional JSON schema for structured output. It is
                passed through to the annotator rather than set here, so this
                argument only guards against setting it twice.

        Returns:
            The populated options instance.

        Raises:
            ValueError: If ``json_schema`` is set in ``options`` while an
                ``output_schema`` is also configured for the step.
        """
        if output_schema is not None and "json_schema" in self.options:
            raise ValueError(
                "Set the schema either as the step's 'output_schema' or as"
                " client options 'json_schema', not both."
            )
        return _options_class(self.provider)(**self.options)

    def build_client(self, root: Path) -> Client[Any] | list[Client[Any]]:
        """Instantiate the client, or one client per server for a pool.

        Args:
            root: Directory that relative paths and globs resolve against.

        Returns:
            A single client, or a list of clients when a pool is configured.
        """
        kwargs = dict(self.init)
        if self.model is not None:
            kwargs["model"] = self.model
        if self.provider == "vllm_offline":
            engine_kwargs = self.engine.as_llm_kwargs()
            # `extra` names arguments this class does not, so it can only
            # travel as the client's own passthrough.
            kwargs.update(
                {
                    k: v
                    for k, v in engine_kwargs.items()
                    if k not in self.engine.extra
                }
            )
            if self.engine.extra:
                kwargs["extra_vllm_kwargs"] = dict(self.engine.extra)

        if not self.is_pool():
            return _client_class(self.provider)(**kwargs)

        # A pool is `vllm_online`-only (enforced in validation), so the server
        # client is named directly here rather than looked up: only it takes
        # the `base_url` that distinguishes one pool member from the next.
        base_urls = self.resolve_base_urls(root)
        if self.wait_for_servers:
            # The pool source is read once here, so it can name fewer servers
            # than `min_servers` while the rest are still starting. Waiting for
            # more servers than it names could never succeed; the ones that
            # arrive later are admitted by the watcher instead.
            min_ready = min(
                self.pool.min_servers, len(dict.fromkeys(base_urls))
            )
            LOGGER.info(
                f"Waiting up to {self.wait_for_servers:g}s for"
                f" {min_ready} vLLM server(s) to become ready..."
            )
            base_urls = wait_for_servers(
                base_urls, self.wait_for_servers, min_ready
            )

        return [VLLMOnlineClient(base_url=url, **kwargs) for url in base_urls]

    def build_annotator(self, root: Path, verbose: bool = False) -> Annotator:
        """Instantiate the annotator that drives this client.

        A pool of servers yields a
        [`VLLMQueueAnnotator`][llm_annotator.annotator.VLLMQueueAnnotator];
        everything else yields a plain
        [`Annotator`][llm_annotator.annotator.Annotator].

        Args:
            root: Directory that relative paths and globs resolve against.
            verbose: Whether the annotator should log progress information.

        Returns:
            The annotator, ready to run.
        """
        one_or_more_clients = self.build_client(root)
        if isinstance(one_or_more_clients, list):
            LOGGER.info(
                f"Annotating over {len(one_or_more_clients)} vLLM server(s)."
            )
            expected_servers = self._expected_pool_size(root)
            annotator = VLLMQueueAnnotator(
                clients=one_or_more_clients,
                batch_size=self.batch_size,
                queue_size=self.queue_size,
                max_concurrent_batches_per_client=(
                    self.max_concurrent_batches_per_client
                ),
                max_workers=(
                    expected_servers * self.max_concurrent_batches_per_client
                ),
                num_proc=self.num_proc,
                verbose=verbose,
            )
            self._watch_pool(root, annotator)
            return annotator
        return Annotator(
            client=one_or_more_clients,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
            verbose=verbose,
        )

    def _expected_pool_size(self, root: Path) -> int:
        """Return how many distinct vLLM servers this pool can grow to."""
        if self.base_urls:
            return self.configured_servers()
        try:
            return max(
                self.pool.servers,
                len(list(dict.fromkeys(self.resolve_base_urls(root)))),
            )
        except ValueError:
            return self.pool.servers

    def _watch_pool(self, root: Path, annotator: VLLMQueueAnnotator) -> None:
        """Add configured vLLM servers to an active pool once they are ready.

        A server is a candidate whenever the pool does not hold it, so this
        admits a server that starts late as well as one that the annotator
        evicted and that answers ``/health`` again.
        """
        kwargs = dict(self.init)
        if self.model is not None:
            kwargs["model"] = self.model

        def discover() -> list[str]:
            try:
                return self.resolve_base_urls(root)
            except ValueError:
                return []

        def watch() -> None:
            while not annotator.is_shutting_down:
                pooled_urls = annotator.client_base_urls()
                candidates = [
                    url for url in discover() if url not in pooled_urls
                ]
                with ThreadPoolExecutor(
                    max_workers=len(candidates) or 1
                ) as pool:
                    readiness = list(
                        pool.map(
                            lambda url: server_is_healthy(url, 5), candidates
                        )
                    )

                for url, is_ready in zip(candidates, readiness, strict=True):
                    if annotator.is_shutting_down:
                        return
                    if is_ready:
                        annotator.add_client_for_base_url(
                            url,
                            lambda base_url: VLLMOnlineClient(
                                base_url=base_url, **kwargs
                            ),
                        )
                if annotator.wait_for_shutdown(timeout=1):
                    return

        threading.Thread(
            target=watch, name="vllm-pool-watcher", daemon=True
        ).start()


class StepConfig(_StrictBase):
    """One annotation pass over the dataset produced by the previous step.

    Prompts and schemas may be given inline or as a file path, never both. File
    paths resolve against the directory holding the config file, so a config
    directory can be moved or shared as a unit.

    Attributes:
        name: Unique step name. Drives the step directory and, by default, the
            ``task_prefix`` that namespaces this step's output columns.
        type: ``"annotate"`` runs over the incoming dataset; ``"generate"``
            synthesises a dataset from ``prompts`` and must come first.
        prompt: Inline prompt template with ``{column}`` placeholders.
        prompt_file: File holding the prompt template.
        system_prompt: Inline system message.
        system_prompt_file: File holding the system message.
        output_schema: Inline JSON schema for structured output.
        output_schema_file: File holding the JSON schema.
        prompts: Prompts for a ``generate`` step, or a file with one per line.
        num_samples: How often to repeat a single ``generate`` prompt.
        client: Client overrides merged over the pipeline-level client block.
            This is a partial block -- it is validated only after merging, by
            [`PipelineConfig.step_client`][llm_annotator.config.PipelineConfig.step_client],
            so a step can change a single option without repeating
            ``provider`` and ``model``.
        task_prefix: Prefix for this step's internal columns and artifacts.
            Defaults to ``"<name>_"``.
        sort_by_length: Sort prompts by length for more efficient batching.
        num_retries_invalid: Retries for samples that fail schema validation.
        max_samples_per_output_file: Samples per JSONL progress file.
            ``"auto"`` (the default) is one percent of the rows with a
            floor of 1000, so at most 100 files are written and a resume
            stays cheap. A fixed number trades the samples lost at a
            crash against the cost of rescanning the files on every
            resume; 0 writes a single file of unlimited size.
        max_consecutive_failed_batches: Abort the step once this many
            batches in a row come back with every sample errored, instead
            of continuing to burn compute against an unresponsive backend.
            0 disables the check.
        upload_every_n_samples: Hub progress-backup cadence. Needs ``hub_id``.
        hub_id: Optional Hub dataset id for this step's prepared-data and
            progress backup, which makes a crashed step resumable from the Hub.
        rename: Mapping from produced column name to its final name.
        drop_columns: Columns to remove after the step finishes.
        filter_invalid: Drop rows whose schema validation still failed after
            all retries. Requires an output schema.
        keep_messages: Keep this step's rendered ``messages`` column instead of
            dropping it once the step is done.
        force_data_preparation: Rebuild prepared data even if it is cached.
    """

    name: str
    type: StepType = "annotate"
    prompt: str | None = None
    prompt_file: Path | None = None
    system_prompt: str | None = None
    system_prompt_file: Path | None = None
    output_schema: dict[str, Any] | None = None
    output_schema_file: Path | None = None
    prompts: list[str] | Path | None = None
    num_samples: int | None = None
    client: dict[str, Any] | None = None
    task_prefix: str | None = None
    sort_by_length: bool | Literal["shortest_first", "longest_first"] = False
    num_retries_invalid: int = Field(default=5, ge=0)
    max_samples_per_output_file: (
        Annotated[int, Field(ge=0)] | Literal["auto"]
    ) = "auto"
    max_consecutive_failed_batches: int = Field(default=10, ge=0)
    upload_every_n_samples: int | None = None
    hub_id: str | None = None
    rename: dict[str, str] = Field(default_factory=dict)
    drop_columns: list[str] = Field(default_factory=list)
    filter_invalid: bool = False
    keep_messages: bool = False
    force_data_preparation: bool = False

    @field_validator("name")
    @classmethod
    def _non_empty_name(cls, value: str) -> str:
        """Reject blank step names, which would make artifact paths clash."""
        if not value.strip():
            raise ValueError("Step 'name' must not be empty.")
        return value

    @model_validator(mode="after")
    def _check_exclusive_sources(self) -> "StepConfig":
        """Enforce inline-or-file for prompts and schemas, plus step-type rules."""
        for inline, from_file in (
            ("prompt", "prompt_file"),
            ("system_prompt", "system_prompt_file"),
            ("output_schema", "output_schema_file"),
        ):
            if getattr(self, inline) is not None and (
                getattr(self, from_file) is not None
            ):
                raise ValueError(
                    f"Step '{self.name}': provide either '{inline}' or"
                    f" '{from_file}', not both."
                )

        if self.type == "annotate":
            if self.prompt is None and self.prompt_file is None:
                raise ValueError(
                    f"Step '{self.name}': an 'annotate' step needs 'prompt' or"
                    " 'prompt_file'."
                )
            if self.prompts is not None or self.num_samples is not None:
                raise ValueError(
                    f"Step '{self.name}': 'prompts' and 'num_samples' only"
                    " apply to a 'generate' step."
                )
        elif self.prompts is None:
            raise ValueError(
                f"Step '{self.name}': a 'generate' step needs 'prompts'."
            )

        if self.filter_invalid and (
            self.output_schema is None and self.output_schema_file is None
        ):
            raise ValueError(
                f"Step '{self.name}': 'filter_invalid' relies on schema"
                " validation, so it needs 'output_schema' or"
                " 'output_schema_file'."
            )
        return self

    def resolved_task_prefix(self) -> str:
        """Get the prefix namespacing this step's columns and artifacts.

        Returns:
            The explicit ``task_prefix`` when set, else ``"<name>_"``.
        """
        if self.task_prefix is not None:
            return self.task_prefix
        return f"{self.name}_"

    def resolved_prompt(self, root: Path) -> str | None:
        """Get the prompt template, reading ``prompt_file`` when needed.

        Args:
            root: Directory that relative paths resolve against.

        Returns:
            The prompt template, or ``None`` for a ``generate`` step that has
            no extra template.
        """
        if self.prompt is not None:
            return self.prompt
        if self.prompt_file is not None:
            return _read_text(self.prompt_file, root)
        return None

    def resolved_system_prompt(self, root: Path) -> str | None:
        """Get the system message, reading ``system_prompt_file`` when needed.

        Args:
            root: Directory that relative paths resolve against.

        Returns:
            The system message, or ``None`` when the step has none.
        """
        if self.system_prompt is not None:
            return self.system_prompt
        if self.system_prompt_file is not None:
            return _read_text(self.system_prompt_file, root)
        return None

    def resolved_output_schema(self, root: Path) -> dict[str, Any] | None:
        """Get the JSON schema, reading ``output_schema_file`` when needed.

        Args:
            root: Directory that relative paths resolve against.

        Returns:
            The schema mapping, or ``None`` when the step has none.

        Raises:
            ValueError: If the schema file does not decode to a mapping.
        """
        if self.output_schema is not None:
            return self.output_schema
        if self.output_schema_file is None:
            return None

        pfin = _resolve_path(self.output_schema_file, root)
        schema = json.loads(pfin.read_text(encoding="utf-8"))
        if not isinstance(schema, dict):
            raise ValueError(
                f"Schema file '{pfin}' must contain a JSON object, got"
                f" {type(schema).__name__}."
            )
        return schema

    def resolved_prompts(self, root: Path) -> list[str]:
        """Get the prompt list for a ``generate`` step.

        A ``.json`` path is read as a list of strings; any other path is read
        as a file with one prompt per line, where blank lines are skipped. A
        single prompt is repeated ``num_samples`` times, mirroring
        [`generate_dataset`][llm_annotator.annotator.Annotator.generate_dataset].

        Args:
            root: Directory that relative paths resolve against.

        Returns:
            The prompts, one per sample to generate.

        Raises:
            ValueError: If no prompt could be resolved, or if a ``.json`` file
                does not hold a non-empty list of strings.
        """
        if isinstance(self.prompts, Path):
            if self.prompts.suffix.lower() == ".json":
                prompts = _read_json_prompts(self.prompts, root)
            else:
                text = _read_text(self.prompts, root)
                prompts = [line for line in text.splitlines() if line.strip()]
        else:
            prompts = list(self.prompts or [])

        if len(prompts) == 1 and self.num_samples:
            prompts = prompts * self.num_samples
        elif self.num_samples:
            prompts = prompts[: self.num_samples]

        if not prompts:
            raise ValueError(
                f"Step '{self.name}': 'prompts' resolved to an empty list."
            )
        return prompts


class PipelineConfig(_StrictBase):
    """A complete, sequentially executed annotation pipeline.

    Each step annotates the dataset produced by the step before it, so a later
    prompt can reference columns an earlier step created. Every step writes its
    own subdirectory under ``output_dir`` and is skipped on a re-run once it has
    finished, which makes a long pipeline restartable.

    Attributes:
        output_dir: Root directory for all step artifacts and the final
            result. A relative value resolves against ``config_dir``, same as
            every other path in the config.
        steps: The steps to run, in order. At least one is required.
        dataset: Input dataset for the first step. Not needed when the first
            step is a ``generate`` step.
        client: Default client settings, merged into every step's own
            ``client``. Optional: a pipeline whose steps each name their own
            provider and model needs no shared default at all.
        hub_id: Optional Hub dataset id for the *final* dataset. Per-step
            backups are configured with a step-level ``hub_id`` instead.
        idx_column: Column name used as the stable per-sample identifier that
            drives resumption. It must not exist in the source dataset. The
            first step adds it, every step output keeps it, and the final
            dataset does not have it.
        overwrite: Delete existing step directories before running, discarding
            any resumable progress.
        verbose: Whether the annotator logs progress information.
        log_level: Package log level for the CLI.
        config_dir: Directory that relative paths in this config resolve
            against. Set automatically by
            [`load_pipeline_config`][llm_annotator.config.load_pipeline_config].

    Examples:
        >>> config = PipelineConfig(
        ...     output_dir="outputs/demo",
        ...     dataset={"name": "stanfordnlp/imdb", "split": "test"},
        ...     client={"provider": "openai", "model": "gpt-4o-mini"},
        ...     steps=[{"name": "classify", "prompt": "Rate: {text}"}],
        ... )
        >>> [step.resolved_task_prefix() for step in config.steps]
        ['classify_']
    """

    output_dir: Path
    steps: list[StepConfig] = Field(min_length=1)
    dataset: DatasetConfig | None = None
    client: ClientConfig | None = None
    hub_id: str | None = None
    idx_column: str = "idx"
    overwrite: bool = False
    verbose: bool = True
    log_level: str = "INFO"
    config_dir: Path = Field(default_factory=Path.cwd)

    @model_validator(mode="after")
    def _resolve_output_dir(self) -> "PipelineConfig":
        """Resolve a relative ``output_dir`` against ``config_dir``."""
        self.output_dir = _resolve_path(self.output_dir, self.config_dir)
        return self

    @model_validator(mode="after")
    def _check_pipeline(self) -> "PipelineConfig":
        """Validate step names, columns, dataset and generate-step placement."""
        names = [step.name for step in self.steps]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"Step names must be unique, found duplicates: {duplicates}."
            )

        prefixes = [step.resolved_task_prefix() for step in self.steps]
        dup_prefixes = sorted({p for p in prefixes if prefixes.count(p) > 1})
        if dup_prefixes:
            raise ValueError(
                "Step task prefixes must be unique, found duplicates:"
                f" {dup_prefixes}."
            )

        for step in self.steps:
            touched = {*step.rename, *step.rename.values(), *step.drop_columns}
            if self.idx_column in touched:
                raise ValueError(
                    f"Step '{step.name}' renames or drops '{self.idx_column}',"
                    " the 'idx_column'. It identifies a row in every step and"
                    " is removed from the final dataset automatically."
                )

        for step in self.steps[1:]:
            if step.type == "generate":
                raise ValueError(
                    f"Step '{step.name}' is a 'generate' step, which replaces"
                    " the dataset instead of annotating it, so it can only be"
                    " the first step."
                )

        if self.steps[0].type == "annotate" and self.dataset is None:
            raise ValueError(
                "The pipeline needs a 'dataset' block because its first step"
                f" ('{self.steps[0].name}') annotates an existing dataset."
            )
        if self.steps[0].type == "generate" and self.dataset is not None:
            raise ValueError(
                "The first step generates its own data, so the 'dataset' block"
                " would be ignored. Remove one of the two."
            )

        # Step client blocks are fragments, so they can only be validated once
        # merged. Do it now rather than halfway through a long run.
        for step in self.steps:
            self.step_client(step)
        return self

    def step_client(self, step: StepConfig) -> ClientConfig:
        """Merge a step's client overrides over the pipeline-level defaults.

        Either level may be omitted: a pipeline whose steps all share one model
        needs only the top-level block, and a pipeline whose steps each use a
        different model needs only the per-step blocks. When both are present,
        merging is one level deep -- ``init`` and ``options`` are merged
        key-by-key so a step can change a single option without repeating the
        whole block, while other keys are replaced outright.

        The one exception is a step that names a *different* ``provider``. It
        inherits no key whose value belongs to the provider it was written
        for: ``init`` and ``options`` on every switch, plus ``engine`` outside
        the vLLM providers and the pool keys (``base_urls``, ``hosts_file``,
        ``url_glob``, ``pool``, ``queue_size``,
        ``max_concurrent_batches_per_client``, ``wait_for_servers``) outside
        ``vllm_online``. Dropping ``init`` is what keeps one provider's
        credentials and base URL from reaching another. A key the step writes
        itself is kept.

        The step's block is only validated here, after merging, because on its
        own it is a fragment that need not name a ``provider`` or ``model``.

        Args:
            step: The step whose effective client settings are wanted.

        Returns:
            The effective client configuration for that step.

        Raises:
            ValueError: If the step ends up with no client at all, or if the
                merged result is not a valid client configuration. The
                offending step is named either way.
        """
        if not step.client:
            if self.client is None:
                raise ValueError(
                    f"Step '{step.name}' has no client to run on. Add a"
                    " top-level 'client' block to share one across steps, or"
                    " a 'client' block to this step."
                )
            return self.client

        override = dict(step.client)
        if self.client is None:
            # Nothing to inherit, so the step's block is the whole config and
            # ClientConfig's own defaults and validation apply to it directly.
            merged = override
        else:
            base = self.client.model_dump()
            merged = {**base, **override}
            if merged["provider"] != base["provider"]:
                for key in _provider_bound_keys(merged["provider"]):
                    if key not in override:
                        merged.pop(key, None)
            else:
                for key in ("init", "options"):
                    if key in override:
                        merged[key] = {
                            **base[key],
                            **(override[key] or {}),
                        }

        try:
            return ClientConfig.model_validate(merged)
        except ValueError as exc:
            raise ValueError(
                f"Step '{step.name}': invalid 'client' block. {exc}"
            ) from exc

    def step_dir(self, index: int) -> Path:
        """Get the directory holding one step's artifacts.

        Args:
            index: Zero-based index of the step.

        Returns:
            ``<output_dir>/<NN>-<name>``, numbered from 1.
        """
        step = self.steps[index]
        return self.output_dir / f"{index + 1:02d}-{step.name}"

    def describe_steps(self) -> list[dict[str, Any]]:
        """Summarise what each step needs in order to run.

        This is the machine-readable half of the config, meant for a job
        submitter that has to start the right resources for each step without
        parsing YAML itself. Everything here is derived from the config, so the
        submitter cannot disagree with the run about which model a step uses.

        Returns:
            One mapping per step, in pipeline order, each with the step's
            ``index`` (from 1), ``name``, ``kind``, ``provider``, ``model``,
            the ``servers`` / ``gpus_per_vllm_server`` it wants, the
            ``min_servers`` it needs before it can start, and the effective
            concurrency from
            [`ClientConfig.concurrency`][llm_annotator.config.ClientConfig.concurrency].
        """
        described = []
        for index, step in enumerate(self.steps):
            client = self.step_client(step)
            described.append(
                {
                    "index": index + 1,
                    "name": step.name,
                    "kind": client.kind(),
                    "provider": client.provider,
                    "model": client.model,
                    "servers": client.pool.servers,
                    "min_servers": client.pool.min_servers,
                    "gpus_per_vllm_server": client.engine.tensor_parallel_size,
                    "step_dir": str(self.step_dir(index)),
                    **client.concurrency(),
                }
            )
        return described


def load_pipeline_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
    step_client_overrides: dict[str, dict[str, Any]] | None = None,
) -> PipelineConfig:
    """Load and validate a pipeline config from a JSON or YAML file.

    ``config_dir`` is set to the file's parent directory, so every relative
    path inside the config resolves against the config file rather than the
    current working directory.

    Args:
        path: Path to the config file.
        overrides: Optional config keys that take precedence over the file. A
            key may be dotted to reach a nested value, and an integer segment
            indexes a list, so ``{"dataset.max_num_samples": 2000}`` and
            ``{"steps.0.client.batch_size": 8}`` both set a single value. A
            missing intermediate block is created.
        step_client_overrides: Optional per-step client fragments, keyed by
            step name, merged into that step's own ``client`` block before
            validation. This is how a job runner tells one step -- and only
            that step -- where the servers it should use are, without
            disturbing steps that run on a different provider.

    Returns:
        The validated pipeline configuration.

    Raises:
        ValueError: If an override key does not fit the shape of the config,
            or if ``step_client_overrides`` names a step the config does not
            define.
    """
    pfin = Path(path).expanduser().resolve()
    data = load_config_file(pfin)
    data.setdefault("config_dir", pfin.parent)
    for key, value in (overrides or {}).items():
        _apply_override(data, key, value)

    if step_client_overrides:
        steps = data.get("steps") or []
        known = {step.get("name") for step in steps if isinstance(step, dict)}
        unknown = sorted(set(step_client_overrides) - known)
        if unknown:
            raise ValueError(
                f"Cannot override the client of unknown step(s) {unknown}."
                f" This config defines {sorted(n for n in known if n)}."
            )
        for step in steps:
            override = step_client_overrides.get(step.get("name"))
            if override:
                step["client"] = {**(step.get("client") or {}), **override}

    return PipelineConfig.model_validate(data)


def _resolve_path(path: str | Path, root: Path) -> Path:
    """Resolve a possibly relative config path against a root directory.

    Args:
        path: Path as written in the config file.
        root: Directory that relative paths resolve against.

    Returns:
        An absolute path.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def _read_text(path: str | Path, root: Path) -> str:
    """Read a UTF-8 text file referenced from a config file.

    The file is used exactly as it is on disk, whatever its suffix, so a prompt
    is what the file says.

    Args:
        path: Path as written in the config file.
        root: Directory that relative paths resolve against.

    Returns:
        The file contents.

    Raises:
        FileNotFoundError: If the file does not exist, with the resolved path
            in the message so a mis-set relative path is obvious.
    """
    pfin = _resolve_path(path, root)
    if not pfin.is_file():
        raise FileNotFoundError(
            f"File '{path}' referenced from the config does not exist"
            f" (resolved to '{pfin}')."
        )

    return pfin.read_text(encoding="utf-8")


def _read_json_prompts(path: str | Path, root: Path) -> list[str]:
    """Read a ``.json`` prompts file of a ``generate`` step.

    Args:
        path: Path as written in the config file.
        root: Directory that relative paths resolve against.

    Returns:
        The prompts, in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON, or does not hold a non-empty
            list of strings.
    """
    pfin = _resolve_path(path, root)
    if not pfin.is_file():
        raise FileNotFoundError(
            f"File '{path}' referenced from the config does not exist"
            f" (resolved to '{pfin}')."
        )

    expected = (
        "A '.json' prompts file holds a list of strings, one per prompt."
    )
    try:
        payload = json.loads(pfin.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Prompts file '{pfin}' is not valid JSON: {exc}. {expected}"
        ) from exc

    if not isinstance(payload, list):
        raise ValueError(
            f"Prompts file '{pfin}' holds a"
            f" {type(payload).__name__}. {expected}"
        )
    if not payload:
        raise ValueError(f"Prompts file '{pfin}' holds an empty list.")

    wrong = [
        index
        for index, entry in enumerate(payload)
        if not isinstance(entry, str)
    ]
    if wrong:
        raise ValueError(
            f"Prompts file '{pfin}' holds entries that are not strings, at"
            f" position(s) {wrong}. {expected}"
        )
    return payload


__all__ = [
    "ClientConfig",
    "DatasetConfig",
    "EngineConfig",
    "PipelineConfig",
    "PoolConfig",
    "StepConfig",
    "StepKind",
    "load_config_file",
    "load_pipeline_config",
    "wait_for_servers",
]
