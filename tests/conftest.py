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
    state: dict[str, Any] = {
        "last_create_kwargs": None,
        "last_post_url": None,
        "last_post_json": None,
        "create_raises": None,
        "post_json": {"choices": []},
        "model_list": ["served-model"],
    }

    class FakeHTTPResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return cast(dict[str, object], state["post_json"])

    class FakeHTTPClient:
        def post(self, url: str, json: dict[str, object]) -> FakeHTTPResponse:
            state["last_post_url"] = url
            state["last_post_json"] = json
            return FakeHTTPResponse()

    class FakeCompletions:
        def create(self, **kwargs: object) -> object:
            state["last_create_kwargs"] = kwargs
            if state["create_raises"] is not None:
                raise cast(Exception, state["create_raises"])
            usage = types.SimpleNamespace(completion_tokens=7)
            choice = types.SimpleNamespace(
                finish_reason="stop",
                message=types.SimpleNamespace(content=" hello "),
            )
            return types.SimpleNamespace(
                choices=[choice],
                usage=usage,
                model="fake-model",
            )

    class FakeOpenAI:
        def __init__(
            self, api_key: str | None = None, base_url: str | None = None
        ):
            self.api_key = api_key
            self.base_url = base_url
            self.chat = types.SimpleNamespace(completions=FakeCompletions())
            self.models = types.SimpleNamespace(
                list=lambda: types.SimpleNamespace(
                    data=[
                        types.SimpleNamespace(id=m)
                        for m in state["model_list"]
                    ]
                )
            )
            self._client = FakeHTTPClient()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI  # type: ignore[attr-defined]

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
            for key, value in kwargs.items():
                setattr(self, key, value)

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
    }

    class FakeMessagesAPI:
        def __init__(self):
            self.batches = types.SimpleNamespace(cancel=lambda _batch_id: None)

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
        def __init__(self, api_key: str | None = None):
            self.api_key = api_key
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
