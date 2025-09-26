"""Test the Annotator class initialization and configuration."""

import pytest
from conftest import MockAnnotator

from llm_annotator.annotator import Annotator


class TestAnnotatorInitialization:
    """Test Annotator class initialization and validation."""

    def test_basic_initialization(self, test_model_id, prompt_template_file):
        """Test basic annotator initialization with minimal parameters."""
        annotator = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
        )

        assert annotator.model_id == test_model_id
        assert annotator.prompt_template_file == prompt_template_file
        assert annotator.tensor_parallel_size == 1
        assert annotator.max_num_seqs == 256
        assert annotator.idx_column == "idx"
        assert annotator.processed_n_samples == 0

    def test_initialization_with_all_parameters(self, test_model_id, prompt_template_file, json_schema):
        """Test annotator initialization with all parameters set."""
        annotator = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            prompt_field_swapper={"old_field": "new_field"},
            output_schema=json_schema,
            whitespace_pattern=r"\s+",
            idx_column="custom_idx",
            num_proc=4,
            tensor_parallel_size=2,
            max_num_seqs=128,
            gpu_memory_utilization=0.7,
            enforce_eager=False,
            quantization="awq",
            verbose=True,
            keep_columns=["text", "label"],
            upload_every_n_samples=5,
            max_samples_per_output_file=100,
            new_hub_id="test/dataset",
            max_model_len=4096,
            enable_thinking=True,
            prefix="test_prefix",
        )

        assert annotator.model_id == test_model_id
        assert annotator.output_schema == json_schema
        assert annotator.idx_column == "custom_idx"
        assert annotator.tensor_parallel_size == 2
        assert annotator.max_num_seqs == 128
        assert annotator.gpu_memory_utilization == 0.7
        assert annotator.quantization == "awq"
        assert annotator.verbose is True
        assert annotator.keep_columns == {"text", "label", "custom_idx"}
        assert annotator.upload_every_n_samples == 5
        assert annotator.new_hub_id == "test/dataset"
        assert annotator.prefix == "test_prefix"

    def test_prompt_template_loading(self, test_model_id, prompt_template_file):
        """Test that prompt template is loaded correctly."""
        annotator = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
        )

        expected_template = prompt_template_file.read_text(encoding="utf-8")
        assert annotator.prompt_template == expected_template
        assert "text" in annotator.prompt_fields

    def test_prompt_field_swapper(self, test_model_id, temp_dir):
        """Test that prompt field swapper works correctly."""
        # Create template with placeholder to replace
        template_path = temp_dir / "swap_test.txt"
        template_content = "Review: {text}\nCategory: {old_field}\nResult:"
        template_path.write_text(template_content, encoding="utf-8")

        annotator = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=template_path,
            prompt_field_swapper={"old_field": "sentiment"},
        )

        expected_template = "Review: {text}\nCategory: sentiment\nResult:"
        assert annotator.prompt_template == expected_template
        assert annotator.prompt_fields == ("text",)  # old_field should be gone

    def test_keep_columns_configuration(self, test_model_id, prompt_template_file):
        """Test different keep_columns configurations."""
        # Test with True (keep all)
        annotator1 = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            keep_columns=True,
        )
        assert annotator1.keep_columns is True

        # Test with string
        annotator2 = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            keep_columns="text",
        )
        assert annotator2.keep_columns == {"text", "idx"}

        # Test with list
        annotator3 = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            keep_columns=["text", "label"],
        )
        assert annotator3.keep_columns == {"text", "label", "idx"}

        # Test with None/empty (keep only idx)
        annotator4 = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            keep_columns=None,
        )
        assert annotator4.keep_columns == {"idx"}

    def test_validation_errors(self, test_model_id, prompt_template_file):
        """Test validation errors during initialization."""
        # Test negative upload_every_n_samples
        with pytest.raises(ValueError, match="upload_every_n_samples must be a positive integer"):
            MockAnnotator(
                model_id=test_model_id,
                prompt_template_file=prompt_template_file,
                upload_every_n_samples=-1,
            )

        # Test upload_every_n_samples without new_hub_id
        with pytest.raises(ValueError, match="If upload_every_n_samples is set, new_hub_id must be provided"):
            MockAnnotator(
                model_id=test_model_id,
                prompt_template_file=prompt_template_file,
                upload_every_n_samples=5,
                new_hub_id=None,
            )

    def test_abstract_method_requirement(self, test_model_id, prompt_template_file):
        """Test that Annotator cannot be instantiated directly (abstract class)."""
        with pytest.raises(TypeError):
            Annotator(
                model_id=test_model_id,
                prompt_template_file=prompt_template_file,
            )

    def test_max_samples_per_output_file_configuration(self, test_model_id, prompt_template_file):
        """Test max_samples_per_output_file configuration."""
        # Test with None (should become 0)
        annotator1 = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            max_samples_per_output_file=None,
        )
        assert annotator1.max_samples_per_output_file == 0

        # Test with negative value (should become 0)
        annotator2 = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            max_samples_per_output_file=-5,
        )
        assert annotator2.max_samples_per_output_file == 0

        # Test with positive value
        annotator3 = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            max_samples_per_output_file=100,
        )
        assert annotator3.max_samples_per_output_file == 100
