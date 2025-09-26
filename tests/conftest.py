"""Test configuration and fixtures for llm_annotator tests."""

import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset
from huggingface_hub import HfApi, delete_repo
from vllm import RequestOutput

from llm_annotator.annotator import Annotator


class MockAnnotator(Annotator):
    def __post_init__(self):
        """Initialize with additional test-specific attributes."""
        super().__post_init__()
        # Initialize attributes that are normally set during _load_dataset
        self.dataset_config = None
        self.dataset_split = None
        self.streaming = False
        # Initialize tokenizer if needed for testing
        try:
            if not hasattr(self, "tokenizer") or self.tokenizer is None:
                self._load_tokenizer()
        except Exception:
            # If tokenizer loading fails in tests, create a mock
            from unittest.mock import Mock

            self.tokenizer = Mock()
            self.tokenizer.apply_chat_template = Mock(return_value="mocked prompt")

    def _process_output(self, output: RequestOutput) -> dict[str, Any]:
        """Process model output for testing.

        Args:
            output: The raw output from the vLLM model.

        Returns:
            Dictionary containing processed test annotations.
        """
        generated_text = output.outputs[0].text.strip() if output.outputs else ""
        return {
            "generated_text": generated_text,
            "finish_reason": output.outputs[0].finish_reason if output.outputs else "unknown",
            "num_tokens": len(output.outputs[0].token_ids) if output.outputs else 0,
        }


@pytest.fixture(scope="session")
def hf_username():
    """Get the Hugging Face username from the token (session scoped).

    Returning the username as a session-scoped fixture lets other session
    fixtures depend on it (for example cleanup tasks) and ensures the value
    is computed only once.
    """
    whoami = HfApi().whoami()
    if whoami and "name" in whoami and whoami["type"] == "user":
        return whoami["name"]
    return None


@pytest.fixture
def test_model_id():
    """Model ID for testing."""
    return "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture
def test_dataset_name():
    """Dataset name for testing."""
    return "stanfordnlp/imdb"


@pytest.fixture
def test_remote_dataset_name(hf_username):
    """Remote dataset name for upload testing.

    This fixture uses the `hf_username` fixture so it resolves to the current
    user's account when available. If no username is available we skip so
    unit tests remain deterministic.
    """

    if hf_username:
        return f"{hf_username}/llm_annotator_test_ds"
    pytest.skip("No Hugging Face username available for remote dataset tests")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def prompt_template_file(temp_dir):
    """Create a test prompt template file."""
    template_path = temp_dir / "test_prompt.txt"
    template_content = """Analyze the sentiment of the following movie review and classify it as positive or negative.

Review: {text}

Classification:"""
    template_path.write_text(template_content, encoding="utf-8")
    return template_path


@pytest.fixture
def test_annotator(test_model_id, prompt_template_file):
    """Create a test annotator instance with disabled multiprocessing."""
    return MockAnnotator(
        model_id=test_model_id,
        prompt_template_file=prompt_template_file,
        num_proc=None,  # Disable multiprocessing to avoid pickle issues
    )


@pytest.fixture
def test_annotator_with_upload(test_model_id, prompt_template_file, hf_username):
    """Create a test annotator instance for testing hub upload functionality."""
    return MockAnnotator(
        model_id=test_model_id,
        prompt_template_file=prompt_template_file,
        num_proc=None,  # Disable multiprocessing to avoid pickle issues
        upload_every_n_samples=2,
        new_hub_id=f"{hf_username}/llm_annotator_test_ds",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration/slow tests when no HF token is present.

    Integration and slow tests are selected by pytest markers (for example
    `pytest -m integration`). We don't require a custom CLI flag; instead
    tests that need external access will be skipped automatically unless a
    Hugging Face token is available in the environment.
    """
    from huggingface_hub import HfApi

    # Get current HF user
    whoami = HfApi().whoami()

    skip_reason = "Skipping integration/slow test because no Hugging Face user logged in."
    skip_marker = pytest.mark.skip(reason=skip_reason)

    for item in items:
        if any(m.name in ("integration", "slow") for m in item.iter_markers()):
            if not whoami or "name" not in whoami or whoami["type"] != "user":
                item.add_marker(skip_marker)


@pytest.fixture
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


def create_mock_dataset_from_dict(data_dict):
    """Create a dataset from dict without triggering multiprocessing issues."""
    dataset = Dataset.from_dict(data_dict)
    return dataset


@pytest.fixture(autouse=True)
def _patch_load_dataset(monkeypatch, small_test_dataset):
    """Autouse fixture to patch load_dataset so tests are deterministic and do not hit the Hub."""

    def _fake_load(*args, **kwargs):
        # Be permissive in the signature to match the real `load_dataset` usage
        # dataset_name is typically the first positional argument
        dataset_name = args[0] if len(args) > 0 else kwargs.get("path") or kwargs.get("dataset_name")

        # Return the small_test_dataset for many known test names
        if dataset_name in ("mock", "stanfordnlp/imdb", "mock/dataset", "empty/dataset"):
            return small_test_dataset

        # Fallback to real load_dataset for others, preserve original args/kwargs
        from datasets import load_dataset as _real_load

        return _real_load(*args, **kwargs)

    monkeypatch.setattr("llm_annotator.annotator.load_dataset", _fake_load)


@pytest.fixture(scope="session", autouse=True)
def cleanup_remote_datasets(hf_username):
    """Clean up any test datasets from HuggingFace Hub after all tests.

    Depend on `hf_username` so the fixture can be used in session-scoped
    cleanup while still respecting the user's account configuration.
    """
    yield
    # Cleanup after all tests
    try:
        if not hf_username:
            return
        test_repo = f"{hf_username}/llm_annotator_test_ds"
        delete_repo(test_repo, repo_type="dataset", missing_ok=True)
        print(f"Cleaned up test dataset: {test_repo}")
    except Exception as e:
        print(f"Warning: Could not clean up test dataset: {e}")


@pytest.fixture
def json_schema():
    """JSON schema for guided decoding tests."""
    return """{
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative"]
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1
            }
        },
        "required": ["sentiment", "confidence"]
    }"""


@pytest.fixture
def prompt_template_with_schema(temp_dir):
    """Create a prompt template that works with JSON schema."""
    template_path = temp_dir / "schema_prompt.txt"
    template_content = """Analyze the sentiment of the following movie review and respond with a JSON object.

Review: {text}

Respond with JSON containing "sentiment" (positive/negative) and "confidence" (0-1):"""
    template_path.write_text(template_content, encoding="utf-8")
    return template_path
