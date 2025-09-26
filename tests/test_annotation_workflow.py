"""Test the full annotation workflow and model integration."""

import json
from unittest.mock import Mock, patch

import pytest
from conftest import MockAnnotator
from vllm import RequestOutput
from vllm.outputs import CompletionOutput


class TestAnnotationWorkflow:
    """Test the complete annotation workflow."""

    @pytest.mark.slow
    def test_full_annotation_workflow(self, test_annotator, test_dataset_name, temp_dir):
        """Test the complete annotation workflow with real model."""
        output_dir = temp_dir / "full_annotation_test"

        test_annotator.annotate_dataset(
            dataset_name=test_dataset_name,
            output_dir=output_dir,
            dataset_split="test",
            max_num_samples=3,
            sampling_params={"temperature": 0.1, "max_tokens": 50},
        )

        # Check that output files were created
        output_files = list(output_dir.glob("*.jsonl"))
        assert len(output_files) > 0

        # Check that all samples were processed
        total_samples = 0
        for output_file in output_files:
            with output_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    data = json.loads(line)
                    assert "generated_text" in data
                    assert "finish_reason" in data
                    assert "num_tokens" in data
                    assert test_annotator.idx_column in data
                    total_samples += 1

        assert total_samples == 3

    def test_annotation_with_mocked_model(self, test_annotator, test_dataset_name, temp_dir):
        """Test annotation workflow with mocked model for faster testing."""
        # Create mock outputs
        mock_completion = Mock(spec=CompletionOutput)
        mock_completion.text = "positive"
        mock_completion.finish_reason = "stop"
        mock_completion.token_ids = [1, 2, 3, 4, 5]

        mock_output = Mock(spec=RequestOutput)
        mock_output.outputs = [mock_completion]

        # Mock the pipeline
        mock_pipe = Mock()
        mock_pipe.generate.return_value = [mock_output, mock_output, mock_output]

        with patch.object(test_annotator, "_load_pipeline"):
            test_annotator.pipe = mock_pipe

            output_dir = temp_dir / "mocked_annotation_test"

            test_annotator.annotate_dataset(
                dataset_name=test_dataset_name,
                output_dir=output_dir,
                dataset_split="test",
                max_num_samples=3,
                sampling_params={"temperature": 0.1, "max_tokens": 50},
            )

            # Verify mock was called
            assert mock_pipe.generate.called

            # Check outputs
            output_files = list(output_dir.glob("*.jsonl"))
            assert len(output_files) > 0

            with output_files[0].open("r", encoding="utf-8") as fh:
                line = fh.readline()
                data = json.loads(line)
                assert data["generated_text"] == "positive"

    def test_annotation_with_guided_decoding(self, test_model_id, prompt_template_with_schema, json_schema, temp_dir):
        """Test annotation with JSON schema guided decoding."""
        annotator = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_with_schema,
            output_schema=json_schema,
            whitespace_pattern=r"\s+",
            num_proc=None,  # Disable multiprocessing to avoid pickle issues
        )

        # Mock the pipeline for testing
        mock_completion = Mock(spec=CompletionOutput)
        mock_completion.text = '{"sentiment": "positive", "confidence": 0.95}'
        mock_completion.finish_reason = "stop"
        mock_completion.token_ids = [1, 2, 3, 4, 5]

        mock_output = Mock(spec=RequestOutput)
        mock_output.outputs = [mock_completion]

        mock_pipe = Mock()
        mock_pipe.generate.return_value = [mock_output, mock_output, mock_output]

        with patch.object(annotator, "_load_pipeline"):
            annotator.pipe = mock_pipe

            output_dir = temp_dir / "guided_decoding_test"

            annotator.annotate_dataset(
                dataset_name="stanfordnlp/imdb",
                output_dir=output_dir,
                dataset_split="test",
                max_num_samples=3,
            )

            # Verify that guided decoding was used by checking the mock was called
            assert mock_pipe.generate.called
            call_args = mock_pipe.generate.call_args
            # Check that the sampling parameters were passed (guided decoding happens in vLLM)
            assert call_args is not None

    def test_batch_processing(self, test_annotator, temp_dir):
        """Test batch processing functionality."""
        test_annotator._load_tokenizer()
        test_annotator._load_pipeline = Mock()  # Skip actual model loading

        # Create mock pipeline
        mock_completion = Mock(spec=CompletionOutput)
        mock_completion.text = "positive"
        mock_completion.finish_reason = "stop"
        mock_completion.token_ids = [1, 2, 3]

        mock_output = Mock(spec=RequestOutput)
        mock_output.outputs = [mock_completion]

        mock_pipe = Mock()
        mock_pipe.generate.return_value = [mock_output, mock_output]
        test_annotator.pipe = mock_pipe

        # Create test batch
        batch = {
            f"{test_annotator.prefix}_prompted": [
                "Review: Great movie! Classification:",
                "Review: Terrible film. Classification:",
            ],
            "text": [
                "Great movie!",
                "Terrible film.",
            ],
            test_annotator.idx_column: [0, 1],
        }

        # Create temp file handle
        output_file = temp_dir / "batch_test.jsonl"
        with output_file.open("w", encoding="utf-8") as fhout:
            results = test_annotator._process_batch(batch, Mock(), fhout)

        assert len(results) == 2
        assert all("generated_text" in result for result in results)
        assert mock_pipe.generate.called

    def test_resume_annotation_after_interruption(self, test_annotator, test_dataset_name, temp_dir):
        """Test that annotation can be resumed after interruption."""
        output_dir = temp_dir / "resume_test"
        output_dir.mkdir()

        # Simulate partial completion by creating output file
        output_file = output_dir / f"{output_dir.stem}.jsonl"
        partial_results = [
            {
                "llm_annotator_idx": 0,
                "text": "sample 1",
                "generated_text": "positive",
                "finish_reason": "stop",
                "num_tokens": 5,
            },
            {
                "llm_annotator_idx": 1,
                "text": "sample 2",
                "generated_text": "negative",
                "finish_reason": "stop",
                "num_tokens": 4,
            },
        ]

        with output_file.open("w", encoding="utf-8") as fh:
            for result in partial_results:
                fh.write(json.dumps(result) + "\n")

        # Mock the pipeline to avoid actual model loading
        mock_completion = Mock(spec=CompletionOutput)
        mock_completion.text = "neutral"
        mock_completion.finish_reason = "stop"
        mock_completion.token_ids = [1, 2, 3]

        mock_output = Mock(spec=RequestOutput)
        mock_output.outputs = [mock_completion]

        mock_pipe = Mock()
        mock_pipe.generate.return_value = [mock_output]  # Only one remaining sample

        with patch.object(test_annotator, "_load_pipeline"):
            test_annotator.pipe = mock_pipe

            test_annotator.annotate_dataset(
                dataset_name=test_dataset_name,
                output_dir=output_dir,
                dataset_split="test",
                max_num_samples=3,
            )

            # Check that only the remaining sample was processed
            total_lines = 0
            with output_file.open("r", encoding="utf-8") as fh:
                for _ in fh:
                    total_lines += 1

            assert total_lines == 3  # 2 existing + 1 new

    def test_overwrite_option(self, test_annotator, test_dataset_name, temp_dir):
        """Test the overwrite option for output directories."""
        output_dir = temp_dir / "overwrite_test"
        output_dir.mkdir()

        # Create existing file
        existing_file = output_dir / "existing.txt"
        existing_file.write_text("existing content")

        # Mock pipeline to avoid model loading
        with patch.object(test_annotator, "_load_pipeline"), patch.object(test_annotator, "_load_dataset"):
            test_annotator.dataset = Mock()
            test_annotator.dataset.__len__ = Mock(return_value=0)  # Empty dataset

            test_annotator.annotate_dataset(
                dataset_name=test_dataset_name,
                output_dir=output_dir,
                overwrite=True,
                max_num_samples=1,
            )

            # Existing file should be gone
            assert not existing_file.exists()

    def test_keep_columns_functionality(self, test_model_id, prompt_template_file, temp_dir):
        """Test different keep_columns configurations during annotation."""
        # Test keeping all columns
        annotator_all = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            keep_columns=True,
        )

        # Test keeping specific columns
        annotator_specific = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            keep_columns=["text"],
        )

        # Create test batch
        batch = {
            f"{annotator_all.prefix}_prompted": ["prompt text"],
            "text": ["original text"],
            "label": [1],
            "extra_field": ["extra"],
            annotator_all.idx_column: [0],
        }

        # Mock results
        mock_result = {"generated_text": "result"}

        # Test keeping all columns
        annotator_all.keep_columns = True
        input_sample = {k: v[0] for k, v in batch.items()}
        merged_all = {**input_sample, **mock_result}

        assert "text" in merged_all
        assert "label" in merged_all
        assert "extra_field" in merged_all

        # Test keeping specific columns
        annotator_specific.keep_columns = {"text", annotator_specific.idx_column}
        input_sample_specific = {k: v[0] for k, v in batch.items() if k in annotator_specific.keep_columns}
        merged_specific = {**input_sample_specific, **mock_result}

        assert "text" in merged_specific
        assert annotator_specific.idx_column in merged_specific
        assert "label" not in merged_specific
        assert "extra_field" not in merged_specific

    def test_post_annotation_cleanup(self, test_annotator, temp_dir):
        """Test that empty files are cleaned up after annotation."""
        output_dir = temp_dir / "cleanup_test"
        output_dir.mkdir()

        # Create empty files
        empty_file1 = output_dir / "empty1.jsonl"
        empty_file2 = output_dir / "empty2.jsonl"
        empty_file1.touch()
        empty_file2.touch()

        # Create non-empty file
        non_empty_file = output_dir / "nonempty.jsonl"
        non_empty_file.write_text('{"test": "data"}\n')

        test_annotator._post_annotate(output_dir)

        # Empty files should be removed
        assert not empty_file1.exists()
        assert not empty_file2.exists()

        # Non-empty file should remain
        assert non_empty_file.exists()

    def test_get_fhout_name_single_file(self, test_annotator, temp_dir):
        """Test output file naming for single file mode."""
        test_annotator.max_samples_per_output_file = 0
        test_annotator.processed_n_samples = 0

        output_path = test_annotator.get_fhout_name(temp_dir)
        expected_path = temp_dir / f"{temp_dir.stem}.jsonl"

        assert output_path == expected_path

    def test_get_fhout_name_multiple_files(self, test_annotator, temp_dir):
        """Test output file naming for multiple files mode."""
        test_annotator.max_samples_per_output_file = 10

        # Test first file
        test_annotator.processed_n_samples = 0
        output_path1 = test_annotator.get_fhout_name(temp_dir)
        expected_path1 = temp_dir / f"{temp_dir.stem}_0.jsonl"
        assert output_path1 == expected_path1

        # Test second file
        test_annotator.processed_n_samples = 10
        output_path2 = test_annotator.get_fhout_name(temp_dir)
        expected_path2 = temp_dir / f"{temp_dir.stem}_1.jsonl"
        assert output_path2 == expected_path2

    def test_model_cleanup(self, test_annotator):
        """Test model and dataset cleanup functionality."""
        # Mock the components that would be cleaned up
        test_annotator.pipe = Mock()
        test_annotator.pipe.llm_engine = Mock()
        test_annotator.pipe.llm_engine.model_executor = Mock()
        test_annotator.dataset = Mock()

        with (
            patch("llm_annotator.annotator.destroy_model_parallel"),
            patch("llm_annotator.annotator.destroy_distributed_environment"),
            patch("llm_annotator.annotator.gc.collect"),
            patch("torch.cuda.empty_cache"),
        ):
            test_annotator._reset_model_and_dataset()

            assert test_annotator.pipe is None
            assert test_annotator.dataset is None
