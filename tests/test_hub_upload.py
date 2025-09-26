"""Test HuggingFace Hub upload functionality."""

import json
from unittest.mock import patch

import pytest
from conftest import MockAnnotator
from huggingface_hub import HfApi


class TestHubUpload:
    """Test HuggingFace Hub upload functionality."""

    def test_push_dir_to_hub_mocked(self, test_annotator_with_upload, temp_dir):
        """Test push_dir_to_hub with mocked HuggingFace Hub calls."""
        # Create test files in directory
        test_file = temp_dir / "test_output.jsonl"
        test_file.write_text('{"idx": 0, "result": "test"}\n')

        config_file = temp_dir / "config.json"
        config_file.write_text('{"test": "config"}')

        # Mock HuggingFace Hub functions
        with (
            patch("llm_annotator.annotator.create_repo") as mock_create_repo,
            patch("llm_annotator.annotator.create_branch") as mock_create_branch,
            patch("llm_annotator.annotator.upload_large_folder") as mock_upload,
        ):
            test_annotator_with_upload.push_dir_to_hub(temp_dir)

            # Verify calls
            mock_create_repo.assert_called_once_with(
                test_annotator_with_upload.new_hub_id, repo_type="dataset", exist_ok=True, private=True
            )

            mock_create_branch.assert_called_once_with(
                test_annotator_with_upload.new_hub_id,
                repo_type="dataset",
                branch=f"{test_annotator_with_upload.prefix}_jsonl_upload",
                exist_ok=True,
            )

            mock_upload.assert_called_once()
            upload_call_kwargs = mock_upload.call_args.kwargs
            assert upload_call_kwargs["repo_id"] == test_annotator_with_upload.new_hub_id
            assert upload_call_kwargs["repo_type"] == "dataset"
            assert upload_call_kwargs["private"] is True
            assert "*.jsonl" in upload_call_kwargs["allow_patterns"]
            assert "*.json" in upload_call_kwargs["allow_patterns"]

    def test_upload_during_annotation(self, test_annotator_with_upload, test_dataset_name, temp_dir):
        """Test that upload is triggered during annotation at specified intervals."""
        # Mock HuggingFace Hub calls
        with (
            patch("llm_annotator.annotator.create_repo"),
            patch("llm_annotator.annotator.create_branch"),
            patch("llm_annotator.annotator.upload_large_folder") as mock_upload,
            patch.object(test_annotator_with_upload, "_load_pipeline"),
            patch.object(test_annotator_with_upload, "_process_batch") as mock_process,
        ):
            # Mock batch processing to return results
            mock_process.return_value = [
                {"generated_text": "positive", "finish_reason": "stop", "num_tokens": 5},
                {"generated_text": "negative", "finish_reason": "stop", "num_tokens": 4},
            ]

            test_annotator_with_upload.annotate_dataset(
                dataset_name=test_dataset_name,
                output_dir=temp_dir / "upload_test",
                dataset_split="test",
                max_num_samples=3,  # Will process in batches due to max_num_seqs=2
            )

            # Upload should be called because upload_every_n_samples=2
            assert mock_upload.called

    def test_upload_retry_mechanism(self, test_annotator_with_upload, temp_dir):
        """Test that upload retry mechanism works correctly."""
        with (
            patch("llm_annotator.annotator.create_repo"),
            patch("llm_annotator.annotator.create_branch"),
            patch("llm_annotator.annotator.upload_large_folder") as mock_upload,
        ):
            # Make upload fail twice, then succeed
            mock_upload.side_effect = [Exception("Network error"), Exception("Another error"), None]

            # The @retry decorator should handle this
            test_annotator_with_upload.push_dir_to_hub(temp_dir)

            # Should have been called 3 times (2 failures + 1 success)
            assert mock_upload.call_count == 3

    def test_upload_with_custom_branch_name(self, test_model_id, prompt_template_file, temp_dir):
        """Test upload with custom prefix affecting branch name."""
        custom_prefix = "custom_test_prefix"
        annotator = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            upload_every_n_samples=1,
            new_hub_id="test/repo",
            prefix=custom_prefix,
        )

        with (
            patch("llm_annotator.annotator.create_repo"),
            patch("llm_annotator.annotator.create_branch") as mock_create_branch,
            patch("llm_annotator.annotator.upload_large_folder"),
        ):
            annotator.push_dir_to_hub(temp_dir)

            # Check that branch name uses custom prefix
            mock_create_branch.assert_called_once_with(
                "test/repo", repo_type="dataset", branch=f"{custom_prefix}_jsonl_upload", exist_ok=True
            )

    def test_upload_patterns(self, test_annotator_with_upload, temp_dir):
        """Test that correct file patterns are used for upload."""
        # Create various test files
        (temp_dir / "data.jsonl").write_text('{"test": "data"}')
        (temp_dir / "config.json").write_text('{"config": "value"}')
        (temp_dir / "README.md").write_text("# Test")
        (temp_dir / "cached_data.txt").write_text("cache")

        # Create cached input dataset directory
        cached_dir = temp_dir / f"{test_annotator_with_upload.prefix}_cached_input_dataset"
        cached_dir.mkdir()
        (cached_dir / "cached_file.txt").write_text("cached")

        with (
            patch("llm_annotator.annotator.create_repo"),
            patch("llm_annotator.annotator.create_branch"),
            patch("llm_annotator.annotator.upload_large_folder") as mock_upload,
        ):
            test_annotator_with_upload.push_dir_to_hub(temp_dir)

            call_kwargs = mock_upload.call_args.kwargs

            # Should include JSONL and JSON files
            assert "*.jsonl" in call_kwargs["allow_patterns"]
            assert "*.json" in call_kwargs["allow_patterns"]

            # Should ignore cached dataset
            expected_ignore = f"{test_annotator_with_upload.prefix}_cached_input_dataset/*"
            assert expected_ignore in call_kwargs["ignore_patterns"]

    @pytest.mark.integration
    def test_real_hub_upload_and_cleanup(
        self, test_annotator_with_upload, test_remote_dataset_name, temp_dir, hf_username
    ):
        """Integration test with real HuggingFace Hub upload and cleanup."""
        # Create test output files
        output_file = temp_dir / "test_data.jsonl"
        test_data = [
            {"idx": 0, "text": "Test sample 1", "result": "positive"},
            {"idx": 1, "text": "Test sample 2", "result": "negative"},
        ]

        with output_file.open("w", encoding="utf-8") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        # Skip if user not logged in
        if not hf_username:
            pytest.skip("No HF token available for real hub upload test")

        try:
            # Upload to Hub
            test_annotator_with_upload.push_dir_to_hub(temp_dir)

            # Verify the dataset exists
            api = HfApi()
            assert api.repo_exists(test_remote_dataset_name, repo_type="dataset")

            # Check that files were uploaded -- they are stored in a separate branch
            repo_files = api.list_repo_files(
                test_remote_dataset_name,
                repo_type="dataset",
                revision=f"{test_annotator_with_upload.prefix}_jsonl_upload",
            )
            assert "test_data.jsonl" in repo_files

        finally:
            # Cleanup - delete the test repository
            try:
                from huggingface_hub import delete_repo

                delete_repo(test_remote_dataset_name, repo_type="dataset")
            except Exception as e:
                print(f"Warning: Could not clean up test dataset: {e}")

    def test_upload_without_hub_id_error(self, test_model_id, prompt_template_file):
        """Test that missing new_hub_id raises appropriate error during initialization."""
        with pytest.raises(ValueError, match="If upload_every_n_samples is set, new_hub_id must be provided"):
            MockAnnotator(
                model_id=test_model_id,
                prompt_template_file=prompt_template_file,
                upload_every_n_samples=5,
                new_hub_id=None,
            )

    def test_upload_disabled_by_default(self, test_annotator, test_dataset_name, temp_dir):
        """Test that upload is disabled by default."""
        assert test_annotator.upload_every_n_samples == 0
        assert test_annotator.new_hub_id is None

        with (
            patch("llm_annotator.annotator.upload_large_folder") as mock_upload,
            patch.object(test_annotator, "_load_pipeline"),
            patch.object(test_annotator, "_process_batch") as mock_process,
        ):
            mock_process.return_value = [{"generated_text": "test", "finish_reason": "stop", "num_tokens": 1}]

            test_annotator.annotate_dataset(
                dataset_name=test_dataset_name,
                output_dir=temp_dir / "no_upload_test",
                dataset_split="test",
                max_num_samples=1,
            )

            # Upload should never be called
            assert not mock_upload.called

    def test_multiple_output_files_upload(self, test_model_id, prompt_template_file, temp_dir):
        """Test upload behavior with multiple output files."""
        annotator = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            upload_every_n_samples=2,
            max_samples_per_output_file=1,  # Force multiple files
            new_hub_id="test/multi_file_repo",
        )

        with (
            patch("llm_annotator.annotator.create_repo"),
            patch("llm_annotator.annotator.create_branch"),
            patch("llm_annotator.annotator.upload_large_folder") as mock_upload,
            patch.object(annotator, "_load_pipeline"),
            patch.object(annotator, "_process_batch") as mock_process,
        ):
            # Mock processing results
            mock_process.return_value = [{"generated_text": "test", "finish_reason": "stop", "num_tokens": 1}]

            annotator.annotate_dataset(
                dataset_name="stanfordnlp/imdb",
                output_dir=temp_dir / "multi_file_test",
                dataset_split="test",
                max_num_samples=3,
            )

            # Should trigger upload when reaching upload_every_n_samples
            assert mock_upload.called
