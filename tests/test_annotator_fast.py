"""Fast unit tests for Annotator class (no LLM required).

These tests mock the Annotator class to avoid torch/vllm dependencies.

Note: Some tests that require Dataset operations are difficult to mock completely
due to the datasets library's internal use of pickle/dill with type checking.
Those tests will work with the full torch/vllm installation but may fail with
mocked dependencies. For complete test coverage, run slow tests which use
the actual dependencies.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from datasets import Dataset


# Create actual classes for mocking
class MockLLM:
    """Mock LLM class for testing."""

    pass


class MockSamplingParams:
    """Mock SamplingParams class for testing."""

    pass


class MockStructuredOutputsParams:
    """Mock StructuredOutputsParams class for testing."""

    pass


class MockRequestOutput:
    """Mock RequestOutput class for testing."""

    pass


# Mock torch and vllm before importing Annotator
mock_torch = MagicMock()
mock_cuda = MagicMock()
mock_torch.cuda = mock_cuda

sys.modules["torch"] = mock_torch
sys.modules["torch.cuda"] = mock_cuda

mock_vllm = MagicMock()
mock_vllm.LLM = MockLLM
mock_vllm.SamplingParams = MockSamplingParams
mock_vllm.RequestOutput = MockRequestOutput

sys.modules["vllm"] = mock_vllm
sys.modules["vllm.distributed"] = MagicMock()

mock_sampling_params_module = MagicMock()
mock_sampling_params_module.StructuredOutputsParams = MockStructuredOutputsParams
sys.modules["vllm.sampling_params"] = mock_sampling_params_module

from llm_annotator.annotator import Annotator


class TestAnnotatorInit:
    """Test Annotator initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        annotator = Annotator(model="test-model")
        assert annotator.model == "test-model"
        assert annotator.pipe is None
        assert annotator.num_proc is None
        assert annotator.tensor_parallel_size == 1
        assert annotator.max_num_seqs == 256

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        annotator = Annotator(
            model="test-model",
            num_proc=4,
            tensor_parallel_size=2,
            max_num_seqs=128,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            verbose=True,
        )
        assert annotator.model == "test-model"
        assert annotator.num_proc == 4
        assert annotator.tensor_parallel_size == 2
        assert annotator.max_num_seqs == 128
        assert annotator.gpu_memory_utilization == 0.8
        assert annotator.enforce_eager is True
        assert annotator.verbose is True

    def test_init_extra_vllm_kwargs(self):
        """Test initialization with extra vLLM kwargs."""
        extra_kwargs = {"custom_param": "value"}
        annotator = Annotator(model="test-model", extra_vllm_init_kwargs=extra_kwargs)
        assert annotator.extra_vllm_init_kwargs == extra_kwargs


class TestAnnotatorContextManager:
    """Test Annotator as context manager."""

    @patch("llm_annotator.annotator.Annotator.destroy_model")
    def test_context_manager(self, mock_destroy):
        """Test using Annotator as context manager."""
        with Annotator(model="test-model") as annotator:
            assert annotator.model == "test-model"
        mock_destroy.assert_called_once()

    @patch("llm_annotator.annotator.Annotator.destroy_model")
    def test_context_manager_with_exception(self, mock_destroy):
        """Test context manager cleanup on exception."""
        with pytest.raises(ValueError):
            with Annotator(model="test-model"):
                raise ValueError("Test error")
        mock_destroy.assert_called_once()


class TestGetSkipIdxs:
    """Test _get_skip_idxs method."""

    def test_get_skip_idxs_empty_dir(self, tmp_path):
        """Test with no existing output files."""
        annotator = Annotator(model="test-model")
        skip_idxs = annotator._get_skip_idxs(
            pdout=tmp_path, idx_column="idx", dataset_split=None, dataset_config=None
        )
        assert skip_idxs == set()

    def test_get_skip_idxs_with_existing_files(self, tmp_path):
        """Test with existing output files."""
        # Create a JSONL file with some processed samples
        output_file = tmp_path / "output.jsonl"
        with output_file.open("w") as f:
            f.write(json.dumps({"idx": 0, "response": "test1"}) + "\n")
            f.write(json.dumps({"idx": 1, "response": "test2"}) + "\n")
            f.write(json.dumps({"idx": 5, "response": "test3"}) + "\n")

        annotator = Annotator(model="test-model")
        skip_idxs = annotator._get_skip_idxs(
            pdout=tmp_path, idx_column="idx", dataset_split=None, dataset_config=None
        )
        assert skip_idxs == {0, 1, 5}

    def test_get_skip_idxs_with_split_filter(self, tmp_path):
        """Test filtering by dataset split."""
        output_file = tmp_path / "output.jsonl"
        with output_file.open("w") as f:
            f.write(json.dumps({"idx": 0, "dataset_split": "train", "response": "test1"}) + "\n")
            f.write(json.dumps({"idx": 1, "dataset_split": "test", "response": "test2"}) + "\n")
            f.write(json.dumps({"idx": 2, "dataset_split": "train", "response": "test3"}) + "\n")

        annotator = Annotator(model="test-model")
        skip_idxs = annotator._get_skip_idxs(
            pdout=tmp_path, idx_column="idx", dataset_split="train", dataset_config=None
        )
        assert skip_idxs == {0, 2}


class TestCreateMessages:
    """Test _create_messages method."""

    def test_create_messages_basic(self):
        """Test basic message creation."""
        annotator = Annotator(model="test-model")
        sample = {"text": "Hello world", "label": 1}
        prompt_template = "Analyze: {text}"

        result = annotator._create_messages(
            sample=sample,
            idx=0,
            prompt_fields=["text"],
            prompt_template=prompt_template,
            idx_column="idx",
            task_prefix="",
        )

        assert result["idx"] == 0
        assert "prompted" in result
        assert len(result["prompted"]) == 1
        assert result["prompted"][0]["role"] == "user"
        assert result["prompted"][0]["content"] == "Analyze: Hello world"

    def test_create_messages_multiple_fields(self):
        """Test message creation with multiple template fields."""
        annotator = Annotator(model="test-model")
        sample = {"text": "Hello", "label": "positive"}
        prompt_template = "Text: {text}, Label: {label}"

        result = annotator._create_messages(
            sample=sample,
            idx=5,
            prompt_fields=["text", "label"],
            prompt_template=prompt_template,
            idx_column="idx",
            task_prefix="test_",
        )

        assert result["idx"] == 5
        assert "test_prompted" in result
        assert result["test_prompted"][0]["content"] == "Text: Hello, Label: positive"


class TestPreprocessDataset:
    """Test _preprocess_dataset method."""

    def test_preprocess_dataset_default(self):
        """Test default preprocessing (no-op)."""
        annotator = Annotator(model="test-model")
        dataset = Dataset.from_dict({"text": ["hello", "world"]})

        result = annotator._preprocess_dataset(dataset=dataset)
        assert result == dataset
        assert len(result) == 2


class TestPostprocessDataset:
    """Test _postprocess_dataset method."""

    def test_postprocess_dataset_default(self):
        """Test default postprocessing (no-op)."""
        annotator = Annotator(model="test-model")
        dataset = Dataset.from_dict({"text": ["hello", "world"]})

        result = annotator._postprocess_dataset(dataset=dataset)
        assert result == dataset
        assert len(result) == 2


class TestProcessOutput:
    """Test _process_output method."""

    def test_process_output_basic(self):
        """Test basic output processing without schema."""
        annotator = Annotator(model="test-model")

        # Create a mock output
        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = "This is a response"
        mock_output.outputs[0].finish_reason = "stop"
        mock_output.outputs[0].token_ids = [1, 2, 3, 4, 5]

        result = annotator._process_output(output=mock_output, output_schema=None, task_prefix="")

        assert result["response"] == "This is a response"
        assert result["finish_reason"] == "stop"
        assert result["num_tokens"] == 5

    def test_process_output_with_schema(self):
        """Test output processing with JSON schema."""
        annotator = Annotator(model="test-model")

        # Create a mock output with valid JSON
        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = '{"sentiment": "positive"}'
        mock_output.outputs[0].finish_reason = "stop"
        mock_output.outputs[0].token_ids = [1, 2, 3]

        schema = {"type": "object", "properties": {"sentiment": {"type": "string"}}, "required": ["sentiment"]}

        result = annotator._process_output(output=mock_output, output_schema=schema, task_prefix="")

        assert result["response"] == '{"sentiment": "positive"}'
        assert result["sentiment"] == "positive"
        assert result["valid_fields"] is True

    def test_process_output_with_invalid_json(self):
        """Test output processing with invalid JSON."""
        annotator = Annotator(model="test-model")

        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = "not valid json"
        mock_output.outputs[0].finish_reason = "stop"
        mock_output.outputs[0].token_ids = [1, 2, 3]

        schema = {"type": "object", "properties": {"sentiment": {"type": "string"}}, "required": ["sentiment"]}

        result = annotator._process_output(output=mock_output, output_schema=schema, task_prefix="")

        assert result["response"] == "not valid json"
        assert result["valid_fields"] is False

    def test_process_output_with_task_prefix(self):
        """Test output processing with task prefix."""
        annotator = Annotator(model="test-model")

        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = "response"
        mock_output.outputs[0].finish_reason = "stop"
        mock_output.outputs[0].token_ids = [1, 2]

        result = annotator._process_output(output=mock_output, output_schema=None, task_prefix="test_")

        assert "test_response" in result
        assert "test_finish_reason" in result
        assert "test_num_tokens" in result


class TestGetPfoutName:
    """Test get_pfout_name method."""

    def test_get_pfout_name_unlimited(self, tmp_path):
        """Test file name generation with unlimited samples."""
        annotator = Annotator(model="test-model")
        output_dir = tmp_path / "outputs"

        result = annotator.get_pfout_name(pdout=output_dir, max_samples_per_output_file=0, processed_n_samples=100)

        assert result == output_dir / "outputs.jsonl"

    def test_get_pfout_name_with_limit(self, tmp_path):
        """Test file name generation with sample limit."""
        annotator = Annotator(model="test-model")
        output_dir = tmp_path / "outputs"

        # First batch (0-99 samples)
        result1 = annotator.get_pfout_name(pdout=output_dir, max_samples_per_output_file=100, processed_n_samples=0)
        assert result1 == output_dir / "outputs_0.jsonl"

        # Second batch (100-199 samples)
        result2 = annotator.get_pfout_name(pdout=output_dir, max_samples_per_output_file=100, processed_n_samples=100)
        assert result2 == output_dir / "outputs_1.jsonl"

        # Third batch (200-299 samples)
        result3 = annotator.get_pfout_name(pdout=output_dir, max_samples_per_output_file=100, processed_n_samples=250)
        assert result3 == output_dir / "outputs_2.jsonl"


class TestLoadDatasetValidation:
    """Test _load_dataset validation."""

    def test_load_dataset_both_dataset_and_name(self):
        """Test error when both dataset and dataset_name are provided."""
        annotator = Annotator(model="test-model")
        dataset = Dataset.from_dict({"text": ["hello"]})

        with pytest.raises(ValueError, match="Provide only one"):
            annotator._load_dataset(
                prompt_template="Test: {text}",
                pdout=Path("/tmp/test"),
                idx_column="idx",
                dataset_name="test/dataset",
                dataset=dataset,
            )

    def test_load_dataset_neither_dataset_nor_name(self):
        """Test error when neither dataset nor dataset_name are provided."""
        annotator = Annotator(model="test-model")

        with pytest.raises(ValueError, match="Either 'dataset' or 'dataset_name' must be provided"):
            annotator._load_dataset(
                prompt_template="Test: {text}",
                pdout=Path("/tmp/test"),
                idx_column="idx",
            )

    def test_load_dataset_invalid_max_num_samples(self):
        """Test error with invalid max_num_samples."""
        annotator = Annotator(model="test-model")
        dataset = Dataset.from_dict({"text": ["hello"]})

        with pytest.raises(ValueError, match="must be a positive integer"):
            annotator._load_dataset(
                prompt_template="Test: {text}",
                pdout=Path("/tmp/test"),
                idx_column="idx",
                dataset=dataset,
                max_num_samples=-1,
            )

    def test_load_dataset_streaming_without_max_samples(self):
        """Test error when streaming without max_num_samples."""
        annotator = Annotator(model="test-model")
        dataset = Dataset.from_dict({"text": ["hello"]})

        with pytest.raises(ValueError, match="must be set to a positive integer"):
            annotator._load_dataset(
                prompt_template="Test: {text}",
                pdout=Path("/tmp/test"),
                idx_column="idx",
                dataset=dataset,
                streaming=True,
            )

    def test_load_dataset_missing_template_field(self):
        """Test error when dataset missing required template field."""
        annotator = Annotator(model="test-model")
        dataset = Dataset.from_dict({"wrong_field": ["hello"]})

        with pytest.raises(ValueError, match="Template contains field"):
            annotator._load_dataset(
                prompt_template="Test: {text}",
                pdout=Path("/tmp/test"),
                idx_column="idx",
                dataset=dataset,
                prompt_fields=["text"],
            )


class TestAnnotateDatasetValidation:
    """Test annotate_dataset validation."""

    def test_annotate_dataset_prefix_not_in_template(self):
        """Test error when prefix not in template."""
        annotator = Annotator(model="test-model")
        dataset = Dataset.from_dict({"text": ["hello"]})

        with pytest.raises(ValueError, match="must be a substring"):
            annotator.annotate_dataset(
                output_dir="/tmp/test",
                full_prompt_template="Test: {text}",
                prompt_template_prefix="This is not in the template",
                dataset=dataset,
            )

    def test_annotate_dataset_invalid_max_samples_per_file(self):
        """Test error with invalid max_samples_per_output_file."""
        annotator = Annotator(model="test-model")
        dataset = Dataset.from_dict({"text": ["hello"]})

        with pytest.raises(ValueError, match="must be None or 0 or a positive integer"):
            annotator.annotate_dataset(
                output_dir="/tmp/test",
                full_prompt_template="Test: {text}",
                dataset=dataset,
                max_samples_per_output_file=-1,
            )

    def test_annotate_dataset_upload_without_hub_id(self):
        """Test error when upload_every_n_samples set without new_hub_id."""
        annotator = Annotator(model="test-model")
        dataset = Dataset.from_dict({"text": ["hello"]})

        with pytest.raises(ValueError, match="new_hub_id must be provided"):
            annotator.annotate_dataset(
                output_dir="/tmp/test",
                full_prompt_template="Test: {text}",
                dataset=dataset,
                upload_every_n_samples=100,
            )

    def test_annotate_dataset_invalid_keep_columns(self):
        """Test error with invalid keep_columns type."""
        annotator = Annotator(model="test-model")
        dataset = Dataset.from_dict({"text": ["hello"]})

        with pytest.raises(TypeError, match="must be None, True, a string, or a collection"):
            annotator.annotate_dataset(
                output_dir="/tmp/test",
                full_prompt_template="Test: {text}",
                dataset=dataset,
                keep_columns=123,  # Invalid type
            )


class TestDestroyModel:
    """Test destroy_model method."""

    @patch("llm_annotator.annotator.destroy_model_parallel")
    @patch("llm_annotator.annotator.destroy_distributed_environment")
    @patch("llm_annotator.annotator.cuda")
    def test_destroy_model_with_pipe(self, mock_cuda, mock_destroy_dist, mock_destroy_parallel):
        """Test destroying an initialized model."""
        annotator = Annotator(model="test-model")
        # Mock a pipe
        annotator.pipe = Mock()
        annotator.pipe.llm_engine = Mock()
        annotator.pipe.llm_engine.model_executor = Mock()
        annotator.pipe.llm_engine.engine_core = Mock()

        annotator.destroy_model()

        mock_destroy_parallel.assert_called_once()
        mock_destroy_dist.assert_called_once()
        mock_cuda.empty_cache.assert_called_once()
        assert annotator.pipe is None

    def test_destroy_model_without_pipe(self):
        """Test destroying when no model is loaded."""
        annotator = Annotator(model="test-model")
        # Should not raise an error
        annotator.destroy_model()
        assert annotator.pipe is None
