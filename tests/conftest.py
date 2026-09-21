"""Test configuration and fixtures for llm_annotator tests."""

import os
import shutil
import socket
import tempfile
import types
from pathlib import Path
from typing import Any, cast

import pytest
from datasets import Dataset
from huggingface_hub import delete_repo

from llm_annotator.annotator import Annotator
from llm_annotator.clients.vllm_offline_client import VLLMOfflineClient
from llm_annotator.utils import get_hf_username


@pytest.fixture(scope="session")
def hf_username():
    """Get the Hugging Face username from the token (session scoped).

    Returning the username as a session-scoped fixture lets other session
    fixtures depend on it (for example cleanup tasks) and ensures the value
    is computed only once.
    """
    return get_hf_username()


@pytest.fixture(scope="session")
def test_model_id():
    """Model ID for testing."""
    return "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="session")
def test_dataset_name():
    """Dataset name for testing."""
    return "stanfordnlp/imdb"


@pytest.fixture(scope="session")
def test_remote_dataset_name(hf_username):
    """Remote dataset name for upload testing.

    This fixture uses the `hf_username` fixture so it resolves to the current
    user's account when available. If no username is available we skip so
    unit tests remain deterministic.
    """

    if hf_username:
        return f"{hf_username}/llm_annotator_test_ds"
    pytest.skip("No Hugging Face username available for remote dataset tests")


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture(scope="session")
def prompt_template_file(temp_dir):
    """Create a session-scoped test prompt template file in temp dir."""
    template_path = temp_dir / "test_prompt.txt"
    template_content = """Analyze the sentiment of the following movie review and classify it as positive or negative.

Review: {text}

Classification:"""
    template_path.write_text(template_content, encoding="utf-8")
    return template_path


@pytest.fixture(scope="session")
def json_schema_file(temp_dir):
    """JSON schema for guided decoding tests."""
    json_path = temp_dir / "test_schema.json"
    json_schema_content = """{
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative"]
        }
    },
    "required": ["sentiment"]
}"""
    json_path.write_text(json_schema_content, encoding="utf-8")
    return json_path


@pytest.fixture(scope="session")
def test_annotator(test_model_id, prompt_template_file):
    """Create a test annotator instance."""
    client = VLLMOfflineClient(model=test_model_id)
    return Annotator(
        client=client,
        num_proc=None,
    )


@pytest.fixture(scope="session")
def small_test_dataset():
    """Create a small test dataset for quick testing."""
    return Dataset.from_dict(
        {
            "text": [
                "This movie is absolutely fantastic! I loved every minute of it.",
                "Terrible film, boring and poorly acted.",
                "An okay movie, nothing special but watchable.",
            ],
            "label": [1, 0, 1],  # positive, negative, positive
        }
    )


@pytest.fixture(scope="session", autouse=True)
def cleanup_remote_datasets():
    """Clean up any test datasets from HuggingFace Hub after all tests.

    This cleanup is opt-in and disabled by default to keep local test runs
    fully offline. Enable it by setting ``LLM_ANNOTATOR_ALLOW_NETWORK_TESTS=1``.
    """
    allow_network = os.environ.get(
        "LLM_ANNOTATOR_ALLOW_NETWORK_TESTS", ""
    ).lower() in {"1", "true", "yes"}

    yield

    if not allow_network:
        return

    # Cleanup after all tests
    try:
        hf_username = get_hf_username()
        if not hf_username:
            pytest.skip("No Hugging Face username available for upload tests")

        test_repo = f"{hf_username}/llm_annotator_test_ds"
        delete_repo(test_repo, repo_type="dataset", missing_ok=True)
        print(f"Cleaned up test dataset: {test_repo}")
    except Exception as e:
        print(f"Warning: Could not clean up test dataset: {e}")


@pytest.fixture(autouse=True, scope="session")
def quiet_vllm_logging():
    import logging
    import os

    logger = logging.getLogger("vllm")
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(logging.NullHandler())
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"


@pytest.fixture
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail fast if any test accidentally attempts a network connection."""

    def _deny_connect(*args: object, **kwargs: object) -> None:
        _ = args
        _ = kwargs
        raise AssertionError("Network access is blocked in unit tests")

    monkeypatch.setattr(socket.socket, "connect", _deny_connect, raising=True)


@pytest.fixture
def fake_openai_module(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Provide a minimal fake OpenAI SDK and capture request payloads."""
    _default_batch_output = (
        '{"id": "resp-1", "custom_id": "request-0", "response": '
        '{"status_code": 200, "body": {"model": "fake-model", "choices": '
        '[{"finish_reason": "stop", "message": {"role": "assistant", '
        '"content": "hello"}}], "usage": {"completion_tokens": 7, '
        '"prompt_tokens": 10, "total_tokens": 17}}}, "error": null}'
    )

    state: dict[str, Any] = {
        "last_create_kwargs": None,
        "create_calls": [],
        "create_raises": None,
        # One response spec per last-user-message content, so a concurrent
        # batch stays deterministic. A spec may set "content", "reasoning",
        # "finish_reason", "completion_tokens" or "raises".
        "create_responses": {},
        "openai_init_kwargs": [],
        "model_list": ["served-model"],
        # Batch API state
        "batch_output_content": _default_batch_output,
        "batch_initial_status": "validating",
        "batch_retrieve_responses": [],
        "created_batches": [],
        "cancelled_batches": [],
        "uploaded_files": [],
        "deleted_files": [],
        "file_contents": {},
        "delete_raises": None,
    }

    def response_spec(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Look up the spec for one create() call by its last message."""
        messages = cast(list[Any], kwargs.get("messages") or [])
        content = messages[-1].get("content") if messages else None
        specs = cast(dict[Any, Any], state["create_responses"])
        return cast(dict[str, Any], specs.get(content, {}))

    class FakeCompletions:
        def create(self, **kwargs: Any) -> object:
            state["last_create_kwargs"] = kwargs
            cast(list[Any], state["create_calls"]).append(kwargs)
            if state["create_raises"] is not None:
                raise cast(Exception, state["create_raises"])
            spec = response_spec(kwargs)
            if spec.get("raises") is not None:
                raise cast(Exception, spec["raises"])
            usage = types.SimpleNamespace(
                completion_tokens=spec.get("completion_tokens", 7)
            )
            message_fields: dict[str, Any] = {
                "content": spec.get("content", " hello ")
            }
            if spec.get("reasoning") is not None:
                message_fields["reasoning"] = spec["reasoning"]
            choice = types.SimpleNamespace(
                finish_reason=spec.get("finish_reason", "stop"),
                message=types.SimpleNamespace(**message_fields),
            )
            return types.SimpleNamespace(
                choices=[choice],
                usage=usage,
                model="fake-model",
            )

    class FakeFiles:
        def create(self, file: Any, purpose: str) -> object:
            state["uploaded_files"].append({"file": file, "purpose": purpose})
            return types.SimpleNamespace(id="file-fake")

        def content(self, file_id: str) -> object:
            """Return the JSONL of one file id.

            ``file_contents`` holds a body per file id, for a batch that wrote
            both an output and an error file. A file id it does not name falls
            back to ``batch_output_content``.
            """
            contents: dict[str, str] = state["file_contents"]
            if file_id in contents:
                return types.SimpleNamespace(text=contents[file_id])
            return types.SimpleNamespace(text=state["batch_output_content"])

        def delete(self, file_id: str) -> object:
            state["deleted_files"].append(file_id)
            if state["delete_raises"] is not None:
                raise cast(Exception, state["delete_raises"])
            return types.SimpleNamespace(id=file_id, deleted=True)

    class FakeBatches:
        def create(
            self,
            input_file_id: str,
            endpoint: str,
            completion_window: str,
        ) -> object:
            state["created_batches"].append(
                {
                    "input_file_id": input_file_id,
                    "endpoint": endpoint,
                    "completion_window": completion_window,
                }
            )
            return types.SimpleNamespace(
                id="batch-fake",
                status=state["batch_initial_status"],
                output_file_id="file-output-fake",
                error_file_id=None,
            )

        def retrieve(self, batch_id: str) -> object:
            poll_responses: list[Any] = state["batch_retrieve_responses"]
            if poll_responses:
                return poll_responses.pop(0)
            return types.SimpleNamespace(
                id=batch_id,
                status="completed",
                output_file_id="file-output-fake",
                error_file_id=None,
            )

        def cancel(self, batch_id: str) -> None:
            state["cancelled_batches"].append(batch_id)

    class FakeDefaultHttpxClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeOpenAI:
        def __init__(self, **kwargs: Any):
            cast(list[Any], state["openai_init_kwargs"]).append(kwargs)
            self.api_key = kwargs.get("api_key")
            self.base_url = kwargs.get("base_url")
            self.timeout = kwargs.get("timeout")
            self.max_retries = kwargs.get("max_retries")
            self.http_client = kwargs.get("http_client")
            self.chat = types.SimpleNamespace(completions=FakeCompletions())
            self.models = types.SimpleNamespace(
                list=lambda: types.SimpleNamespace(
                    data=[
                        types.SimpleNamespace(id=m)
                        for m in state["model_list"]
                    ]
                )
            )
            self.files = FakeFiles()
            self.batches = FakeBatches()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI  # type: ignore[attr-defined]
    fake_openai.DefaultHttpxClient = FakeDefaultHttpxClient  # type: ignore[attr-defined]

    types_mod = types.ModuleType("openai.types")
    chat_mod = types.ModuleType("openai.types.chat")
    cc_mod = types.ModuleType("openai.types.chat.chat_completion")

    class FakeChatCompletion:
        def __init__(self, **kwargs: object):
            choices = kwargs.get("choices")
            if isinstance(choices, list):
                normalized_choices: list[object] = []
                for choice in choices:
                    if isinstance(choice, dict):
                        message = choice.get("message")
                        if isinstance(message, dict):
                            message = types.SimpleNamespace(**message)
                        normalized_choices.append(
                            types.SimpleNamespace(
                                finish_reason=choice.get("finish_reason"),
                                message=message,
                            )
                        )
                    else:
                        normalized_choices.append(choice)
                kwargs["choices"] = normalized_choices
            if "usage" not in kwargs:
                kwargs["usage"] = types.SimpleNamespace(completion_tokens=None)
            elif isinstance(kwargs["usage"], dict):
                kwargs["usage"] = types.SimpleNamespace(
                    **cast(dict[str, object], kwargs["usage"])
                )
            for key, value in kwargs.items():
                setattr(self, key, value)

        @classmethod
        def model_validate(cls, data: dict[str, Any]) -> "FakeChatCompletion":
            return cls(**data)

    cc_mod.ChatCompletion = FakeChatCompletion  # type: ignore[attr-defined]

    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)
    monkeypatch.setitem(__import__("sys").modules, "openai.types", types_mod)
    monkeypatch.setitem(
        __import__("sys").modules, "openai.types.chat", chat_mod
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "openai.types.chat.chat_completion",
        cc_mod,
    )

    return state


@pytest.fixture
def fake_anthropic_module(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Provide a minimal fake Anthropic SDK and capture request payloads."""
    state: dict[str, Any] = {
        "last_create_kwargs": None,
        "create_raises": None,
        "anthropic_init_kwargs": [],
    }

    class FakeMessagesAPI:
        def create(self, **kwargs: object) -> object:
            state["last_create_kwargs"] = kwargs
            if state["create_raises"] is not None:
                raise cast(Exception, state["create_raises"])
            return types.SimpleNamespace(
                usage=types.SimpleNamespace(output_tokens=4),
                stop_reason="end_turn",
                model="claude-fake",
                content=[
                    types.SimpleNamespace(type="text", text="first line"),
                    types.SimpleNamespace(type="text", text="second line"),
                ],
            )

    class FakeAnthropic:
        def __init__(self, **kwargs: Any):
            cast(list[Any], state["anthropic_init_kwargs"]).append(kwargs)
            self.api_key = kwargs.get("api_key")
            self.timeout = kwargs.get("timeout")
            self.max_retries = kwargs.get("max_retries")
            self.messages = FakeMessagesAPI()

    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = FakeAnthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_anthropic)

    return state


@pytest.fixture
def fake_google_genai_module(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Provide a minimal fake Google GenAI SDK and capture request payloads."""
    state: dict[str, Any] = {
        "last_generate_kwargs": None,
        "generate_raises": None,
    }

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs: object):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FakeModels:
        def generate_content(self, **kwargs: object) -> object:
            state["last_generate_kwargs"] = kwargs
            if state["generate_raises"] is not None:
                raise cast(Exception, state["generate_raises"])
            candidate = types.SimpleNamespace(
                finish_reason=types.SimpleNamespace(value="STOP"),
                token_count=5,
            )
            return types.SimpleNamespace(
                candidates=[candidate],
                text=' {"label": "ok"} ',
                model_version="gemini-fake",
            )

    class FakeClient:
        def __init__(self, api_key: str | None = None):
            self.api_key = api_key
            self.models = FakeModels()

    google_mod = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")
    types_mod = types.ModuleType("google.genai.types")

    genai_mod.Client = FakeClient  # type: ignore[attr-defined]
    genai_mod.types = types_mod  # type: ignore[attr-defined]
    types_mod.GenerateContentConfig = FakeGenerateContentConfig  # type: ignore[attr-defined]
    google_mod.genai = genai_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(__import__("sys").modules, "google", google_mod)
    monkeypatch.setitem(__import__("sys").modules, "google.genai", genai_mod)
    monkeypatch.setitem(
        __import__("sys").modules,
        "google.genai.types",
        types_mod,
    )

    return state
