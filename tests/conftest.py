"""Test configuration and fixtures for llm_annotator tests."""

import shutil
import tempfile
from pathlib import Path

import pytest
from datasets import Dataset
from huggingface_hub import HfApi, delete_repo

from llm_annotator.annotator import Annotator


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
    return Annotator(
        model_id=test_model_id,
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
def cleanup_remote_datasets(hf_username):
    """Clean up any test datasets from HuggingFace Hub after all tests.

    Depend on `hf_username` so the fixture can be used in session-scoped
    cleanup while still respecting the user's account configuration.
    """
    yield
    # Cleanup after all tests
    try:
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
