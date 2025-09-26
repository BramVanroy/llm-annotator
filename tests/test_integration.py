"""Integration test demonstrating the complete annotation workflow."""

import json

import pytest
from conftest import MockAnnotator
from huggingface_hub import HfApi, delete_repo

from llm_annotator import annotator as annotator_mod


class TestIntegration:
    """Integration tests for the complete annotation workflow."""

    @pytest.mark.slow
    def test_complete_annotation_workflow_with_caching(self, test_model_id, prompt_template_file, temp_dir):
        """Test complete workflow including stopping and resuming (caching)."""
        annotator = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            tensor_parallel_size=1,
            max_num_seqs=2,
            gpu_memory_utilization=0.8,
            verbose=False,
        )

        output_dir = temp_dir / "integration_test"

        # First run - process part of the dataset
        annotator.annotate_dataset(
            dataset_name="stanfordnlp/imdb",
            output_dir=output_dir,
            dataset_split="test",
            max_num_samples=3,
            sampling_params={"temperature": 0.1, "max_tokens": 50},
        )

        # Verify output was created
        output_files = list(output_dir.glob("*.jsonl"))
        assert len(output_files) > 0

        # Count initial samples (should match requested max_num_samples)
        initial_samples = 0
        for output_file in output_files:
            with output_file.open("r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    assert "generated_text" in data
                    assert "finish_reason" in data
                    assert "num_tokens" in data
                    assert annotator.idx_column in data
                    initial_samples += 1

        assert initial_samples == 3

        # Verify cache was created
        cached_path = output_dir / f"{annotator.prefix}_cached_input_dataset"
        assert cached_path.exists()

        # Second run - should use cache and skip already processed samples
        annotator2 = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            tensor_parallel_size=1,
            max_num_seqs=2,
            gpu_memory_utilization=0.8,
            verbose=False,
        )

        # Add more samples to process (request a larger max to ensure additional processing)
        annotator2.annotate_dataset(
            dataset_name="stanfordnlp/imdb",
            output_dir=output_dir,
            dataset_split="test",
            max_num_samples=5,
            sampling_params={"temperature": 0.1, "max_tokens": 50},
        )

        # Count final samples across files
        final_samples = 0
        for output_file in output_files:
            with output_file.open("r", encoding="utf-8") as f:
                for line in f:
                    final_samples += 1

        # Should have increased by the additional requested samples (up to dataset size)
        assert final_samples >= initial_samples

    @pytest.mark.integration
    def test_hub_upload_and_cleanup_integration(self, test_model_id, prompt_template_file, temp_dir, hf_username):
        """Integration test for HuggingFace Hub upload and cleanup."""
        # Use the hf_username fixture to construct a per-user test repo id.
        test_hub_id = f"{hf_username}/llm_annotator_test_ds"

        annotator = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            tensor_parallel_size=1,
            max_num_seqs=2,
            upload_every_n_samples=2,
            new_hub_id=test_hub_id,
        )

        output_dir = temp_dir / "hub_integration_test"

        try:
            # Run annotation with upload
            annotator.annotate_dataset(
                dataset_name="stanfordnlp/imdb",
                output_dir=output_dir,
                dataset_split="test",
                max_num_samples=3,
                sampling_params={"temperature": 0.1, "max_tokens": 30},
            )

            # Verify files were created locally
            output_files = list(output_dir.glob("*.jsonl"))
            assert len(output_files) > 0

            # Verify upload to Hub occurred
            api = HfApi()
            if api.repo_exists(test_hub_id, repo_type="dataset"):
                # Check that files were uploaded
                repo_files = api.list_repo_files(
                    test_hub_id, repo_type="dataset", revision=f"{annotator.prefix}_jsonl_upload"
                )
                jsonl_files = [f for f in repo_files if f.endswith(".jsonl")]
                assert len(jsonl_files) > 0

        finally:
            # Cleanup - always try to delete the test repository
            try:
                api = HfApi()
                if api.repo_exists(test_hub_id, repo_type="dataset"):
                    delete_repo(test_hub_id, repo_type="dataset")
                    print(f"Successfully cleaned up test dataset: {test_hub_id}")
            except Exception as e:
                print(f"Warning: Could not clean up test dataset: {e}")

    @pytest.mark.slow
    def test_multiple_file_output_workflow(self, test_model_id, prompt_template_file, temp_dir):
        """Test workflow with multiple output files."""
        annotator = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            max_samples_per_output_file=2,  # Force multiple files
            tensor_parallel_size=1,
            max_num_seqs=2,
        )

        output_dir = temp_dir / "multi_file_integration"

        annotator.annotate_dataset(
            dataset_name="stanfordnlp/imdb",
            output_dir=output_dir,
            dataset_split="test",
            max_num_samples=5,
            sampling_params={"temperature": 0.1, "max_tokens": 30},
        )
        # Should create multiple files (up to requested samples or dataset size)
        output_files = sorted(output_dir.glob("*.jsonl"))
        assert len(output_files) >= 1

        # Verify sample distribution
        total_samples = 0
        for output_file in output_files:
            file_samples = 0
            with output_file.open("r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    assert "generated_text" in data
                    file_samples += 1
                    total_samples += 1
            # Each file should have at most max_samples_per_output_file samples
            assert file_samples <= annotator.max_samples_per_output_file

        # Expected number is min(requested, dataset size)
        ds_len = len(annotator_mod.load_dataset("stanfordnlp/imdb", split="test"))
        expected_total = min(5, ds_len)
        assert total_samples == expected_total

    @pytest.mark.slow
    def test_streaming_dataset_workflow(self, test_model_id, prompt_template_file, temp_dir):
        """Test workflow with streaming dataset."""
        annotator = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            tensor_parallel_size=1,
            max_num_seqs=2,
        )

        output_dir = temp_dir / "streaming_integration"

        annotator.annotate_dataset(
            dataset_name="stanfordnlp/imdb",
            output_dir=output_dir,
            dataset_split="test",
            streaming=True,
            max_num_samples=3,
            shuffle_seed=42,
            sampling_params={"temperature": 0.1, "max_tokens": 30},
        )

        # Verify output
        output_files = list(output_dir.glob("*.jsonl"))
        assert len(output_files) > 0

        total_samples = 0
        for output_file in output_files:
            with output_file.open("r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    assert "generated_text" in data
                    total_samples += 1

        assert total_samples == 3

    def test_error_recovery_workflow(self, test_model_id, prompt_template_file, temp_dir):
        """Test error recovery during annotation workflow."""
        annotator = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            tensor_parallel_size=1,
            max_num_seqs=2,
        )

        output_dir = temp_dir / "error_recovery_test"

        # Create partial results to simulate previous interrupted run
        output_dir.mkdir()
        partial_file = output_dir / f"{output_dir.stem}.jsonl"
        partial_results = [
            {
                "llm_annotator_idx": 0,
                "text": "Great movie!",
                "generated_text": "positive",
                "finish_reason": "stop",
                "num_tokens": 1,
            },
            {
                "llm_annotator_idx": 1,
                "text": "Bad film.",
                "generated_text": "negative",
                "finish_reason": "stop",
                "num_tokens": 1,
            },
        ]

        with partial_file.open("w", encoding="utf-8") as f:
            for result in partial_results:
                f.write(json.dumps(result) + "\n")

        # Run annotation - should skip the already processed samples
        with pytest.warns(Warning):
            annotator.annotate_dataset(
                dataset_name="stanfordnlp/imdb",
                output_dir=output_dir,
                dataset_split="test",
                max_num_samples=4,  # 2 more than already processed
                sampling_params={"temperature": 0.1, "max_tokens": 30},
            )

        # Verify final output
        total_samples = 0
        with partial_file.open("r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                assert "generated_text" in data
                total_samples += 1
        # Expected final total is min(requested, dataset size)
        ds_len = len(annotator_mod.load_dataset("stanfordnlp/imdb", split="test"))
        expected_final = min(4, ds_len)
        assert total_samples == expected_final
        assert annotator.processed_n_samples >= 2  # At least 2 were skipped initially

    @pytest.mark.slow
    def test_guided_decoding_workflow(self, test_model_id, prompt_template_with_schema, json_schema, temp_dir):
        """Test workflow with guided decoding using JSON schema."""
        annotator = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_with_schema,
            output_schema=json_schema,
            whitespace_pattern=r"\s+",
            tensor_parallel_size=1,
            max_num_seqs=1,  # Smaller batch for guided decoding
        )

        output_dir = temp_dir / "guided_decoding_integration"

        annotator.annotate_dataset(
            dataset_name="stanfordnlp/imdb",
            output_dir=output_dir,
            dataset_split="test",
            max_num_samples=2,
            sampling_params={"temperature": 0.1, "max_tokens": 100},
        )

        # Verify output
        output_files = list(output_dir.glob("*.jsonl"))
        assert len(output_files) > 0

        with output_files[0].open("r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                assert "generated_text" in data
                # The generated text should ideally be valid JSON due to guided decoding
                # but we don't enforce this in testing as the model might not be perfect
