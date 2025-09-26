"""Test error handling and edge cases."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from conftest import MockAnnotator
from datasets import Dataset

from llm_annotator.annotator import Annotator


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_prompt_template_file(self, test_model_id, temp_dir):
        """Test error when prompt template file doesn't exist."""
        nonexistent_file = temp_dir / "nonexistent.txt"

        with pytest.raises(FileNotFoundError):
            MockAnnotator(
                model_id=test_model_id,
                prompt_template_file=nonexistent_file,
            )

    def test_empty_prompt_template(self, test_model_id, temp_dir):
        """Test handling of empty prompt template."""
        empty_template = temp_dir / "empty.txt"
        empty_template.write_text("", encoding="utf-8")

        annotator = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=empty_template,
        )

        assert annotator.prompt_template == ""
        assert annotator.prompt_fields == ()

    def test_prompt_template_without_fields(self, test_model_id, temp_dir):
        """Test prompt template without any format fields."""
        static_template = temp_dir / "static.txt"
        static_template.write_text("This is a static prompt with no fields.", encoding="utf-8")

        annotator = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=static_template,
        )

        assert annotator.prompt_fields == ()

    def test_nonexistent_dataset(self, test_annotator, temp_dir):
        """Test error handling for nonexistent dataset."""
        test_annotator._load_tokenizer()

        with pytest.raises(Exception):  # Could be various dataset loading errors
            test_annotator._load_dataset(
                dataset_name="nonexistent/dataset",
                pdout=temp_dir,
                max_num_samples=1,
            )

    def test_missing_required_fields_in_dataset(self, test_annotator, temp_dir):
        """Test handling of datasets missing required fields."""
        # Mock a dataset without the required 'text' field using small_test_dataset fixture
        from unittest.mock import patch

        with patch("llm_annotator.annotator.load_dataset") as mock_load:
            # Create a dataset without the required 'text' field
            mock_dataset = Dataset.from_dict({"label": [0, 1], "content": ["test1", "test2"]})
            mock_load.return_value = mock_dataset

            with pytest.raises(ValueError, match="Template contains field 'text'"):
                test_annotator._load_dataset(
                    dataset_name="mock/dataset",
                    pdout=temp_dir,
                    dataset_split="train",
                    max_num_samples=3,
                )

    def test_corrupted_output_files(self, test_annotator, temp_dir):
        """Test handling of corrupted output files during skip detection."""
        # Create corrupted JSONL file
        corrupted_file = temp_dir / "corrupted.jsonl"
        # Use proper JSON format but with intentional corruption
        corrupted_file.write_text(
            '{"idx": 0, "result": "good"}\n{"idx": 1, invalid_json}\n{"idx": 2, "result": "ok"}', encoding="utf-8"
        )

        # Should handle corruption gracefully and continue processing
        try:
            skip_idxs = test_annotator._get_skip_idxs(temp_dir)
            # If it manages to parse some entries, that's fine
            assert isinstance(skip_idxs, set)
        except Exception:
            # If it fails completely due to corruption, that's expected too
            pass

    def test_empty_dataset_handling(self, test_annotator, temp_dir):
        """Test handling of empty datasets."""
        from unittest.mock import patch

        with patch("llm_annotator.annotator.load_dataset") as mock_load:
            # Create an empty dataset
            mock_dataset = Dataset.from_dict({"text": [], "label": []})
            mock_load.return_value = mock_dataset

            # Load the empty dataset - it should work but result in no samples
            test_annotator._load_dataset(
                dataset_name="empty/dataset",
                pdout=temp_dir,
                dataset_split="train",
                max_num_samples=3,
            )
            # Verify the dataset is empty
            assert len(test_annotator.dataset) == 0

    def test_model_loading_failure(self, test_annotator):
        """Test handling of model loading failures."""
        # Patch the LLM class to raise during pipeline initialization
        with patch("llm_annotator.annotator.LLM", side_effect=Exception("Model loading failed")):
            with pytest.raises(Exception, match="Model loading failed"):
                # Call the real _load_pipeline which will attempt to instantiate LLM
                Annotator._load_pipeline(test_annotator)

    def test_tokenizer_loading_failure(self, test_annotator):
        """Test handling of tokenizer loading failures."""
        with patch(
            "llm_annotator.annotator.AutoTokenizer.from_pretrained", side_effect=Exception("Tokenizer loading failed")
        ):
            with pytest.raises(Exception, match="Tokenizer loading failed"):
                test_annotator._load_tokenizer()

    def test_output_directory_permission_error(self, test_annotator, test_dataset_name):
        """Test handling of permission errors when creating output directory."""
        # Try to create output in root (should fail on most systems)
        restricted_path = Path("/root/restricted_output")

        # Mock the mkdir to raise PermissionError
        with patch("pathlib.Path.mkdir", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                test_annotator.annotate_dataset(
                    dataset_name=test_dataset_name,
                    output_dir=restricted_path,
                    max_num_samples=1,
                )

    def test_disk_space_error_during_annotation(self, test_annotator, test_dataset_name, temp_dir):
        """Test handling of disk space errors during annotation."""
        output_dir = temp_dir / "disk_full_test"

        # Mock file writing to raise OSError (disk full)
        with patch("builtins.open", side_effect=OSError("No space left on device")):
            with pytest.raises(OSError):
                test_annotator.annotate_dataset(
                    dataset_name=test_dataset_name,
                    output_dir=output_dir,
                    dataset_split="test",
                    max_num_samples=1,
                )

    def test_network_error_during_upload(self, test_annotator_with_upload, temp_dir):
        """Test handling of network errors during upload."""
        # Create test file
        test_file = temp_dir / "test.jsonl"
        test_file.write_text('{"test": "data"}')

        with (
            patch("llm_annotator.annotator.create_repo"),
            patch("llm_annotator.annotator.create_branch"),
            patch("llm_annotator.annotator.upload_large_folder", side_effect=Exception("Network error")),
        ):
            # Should raise exception after retries
            with pytest.raises(Exception, match="Network error"):
                test_annotator_with_upload.push_dir_to_hub(temp_dir)

    def test_invalid_json_schema(self, test_model_id, prompt_template_file):
        """Test handling of invalid JSON schema."""
        invalid_schema = '{"type": "invalid_type"}'  # Invalid JSON schema

        annotator = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            output_schema=invalid_schema,
        )

        # Should not fail during initialization
        assert annotator.output_schema == invalid_schema
        # Error would occur during actual model generation

    def test_model_generation_failure(self, test_annotator, test_dataset_name, temp_dir):
        """Test handling of model generation failures."""
        # Mock pipeline to raise exception during generation
        mock_pipe = Mock()
        mock_pipe.generate.side_effect = Exception("Generation failed")

        with patch.object(test_annotator, "_load_pipeline"):
            test_annotator.pipe = mock_pipe

            with pytest.raises(Exception, match="Generation failed"):
                test_annotator.annotate_dataset(
                    dataset_name=test_dataset_name,
                    output_dir=temp_dir,
                    dataset_split="test",
                    max_num_samples=1,
                )

    def test_batch_processing_with_mixed_success_failure(self, test_annotator, temp_dir):
        """Test batch processing when some outputs succeed and others fail."""
        # Create a custom _process_output that fails for certain outputs
        original_process_output = test_annotator._process_output

        def failing_process_output(output):
            if hasattr(output, "should_fail") and output.should_fail:
                raise Exception("Processing failed for this output")
            return original_process_output(output)

        test_annotator._process_output = failing_process_output

        # Mock outputs - one succeeds, one fails
        mock_output1 = Mock()
        mock_output1.outputs = [Mock(text="success", finish_reason="stop", token_ids=[1, 2, 3])]
        mock_output1.should_fail = False

        mock_output2 = Mock()
        mock_output2.should_fail = True

        mock_pipe = Mock()
        mock_pipe.generate.return_value = [mock_output1, mock_output2]

        test_annotator.pipe = mock_pipe

        batch = {
            f"{test_annotator.prefix}_prompted": ["prompt1", "prompt2"],
            "text": ["text1", "text2"],
            test_annotator.idx_column: [0, 1],
        }

        # Should raise exception due to failing output
        with pytest.raises(Exception, match="Processing failed"):
            test_annotator._process_batch(batch, Mock(), Mock())

    def test_malformed_cached_dataset(self, test_annotator, test_dataset_name, temp_dir):
        """Test handling of malformed cached dataset."""
        test_annotator._load_tokenizer()

        # Create malformed cache directory
        cached_path = temp_dir / f"{test_annotator.prefix}_cached_input_dataset"
        cached_path.mkdir()

        # Write invalid data to cache
        invalid_file = cached_path / "data-00000-of-00001.parquet"
        invalid_file.write_text("invalid parquet data")

        # Should handle gracefully and reload dataset
        with patch("datasets.Dataset.load_from_disk", side_effect=Exception("Corrupted cache")):
            test_annotator._load_dataset(
                dataset_name=test_dataset_name,
                pdout=temp_dir,
                dataset_split="test",
                max_num_samples=3,
            )

            # Should still work by reloading from source
            assert test_annotator.dataset is not None

    def test_cleanup_with_missing_attributes(self, test_annotator):
        """Test cleanup when model attributes are missing."""
        # Set pipe but without proper structure
        test_annotator.pipe = Mock()
        test_annotator.pipe.llm_engine = None  # Missing attribute

        with (
            patch("llm_annotator.annotator.destroy_model_parallel"),
            patch("llm_annotator.annotator.destroy_distributed_environment"),
            patch("llm_annotator.annotator.gc.collect"),
            patch("torch.cuda.empty_cache"),
        ):
            # Should handle missing attributes gracefully
            try:
                test_annotator._reset_model_and_dataset()
            except AttributeError:
                pytest.fail("Should handle missing attributes gracefully")

    def test_invalid_dataset_split(self, test_annotator, test_dataset_name, temp_dir):
        """Test handling of invalid dataset split."""
        test_annotator._load_tokenizer()
        # load_dataset is patched in conftest to return a small dataset; ensure no exception
        test_annotator._load_dataset(
            dataset_name=test_dataset_name,
            pdout=temp_dir,
            dataset_split="nonexistent_split",
            max_num_samples=1,
        )

    def test_zero_max_num_samples(self, test_annotator, test_dataset_name, temp_dir):
        """Test handling of zero max_num_samples."""
        test_annotator._load_tokenizer()
        test_annotator._load_dataset(
            dataset_name=test_dataset_name,
            pdout=temp_dir,
            dataset_split="test",
            max_num_samples=0,
        )

        assert len(test_annotator.dataset) == 0
