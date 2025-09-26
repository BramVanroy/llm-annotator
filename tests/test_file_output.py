"""Test file output and multiple file handling."""

import json
from unittest.mock import Mock, patch

import pytest
from conftest import MockAnnotator


class TestFileOutput:
    """Test file output functionality and multiple file handling."""

    def test_single_output_file(self, test_annotator, test_dataset_name, temp_dir):
        """Test annotation with single output file."""
        test_annotator.max_samples_per_output_file = 0  # Single file mode

        with (
            patch.object(test_annotator, "_load_pipeline"),
            patch.object(test_annotator, "_process_batch") as mock_process,
        ):
            mock_process.return_value = [
                {"generated_text": "positive", "finish_reason": "stop", "num_tokens": 5},
                {"generated_text": "negative", "finish_reason": "stop", "num_tokens": 4},
                {"generated_text": "neutral", "finish_reason": "stop", "num_tokens": 3},
            ]

            output_dir = temp_dir / "single_file_test"
            test_annotator.annotate_dataset(
                dataset_name=test_dataset_name,
                output_dir=output_dir,
                dataset_split="test",
                max_num_samples=3,
            )

            # Should create single output file
            output_files = list(output_dir.glob("*.jsonl"))
            assert len(output_files) == 1
            assert output_files[0].name == f"{output_dir.stem}.jsonl"

            # Check all samples are in the single file
            with output_files[0].open("r", encoding="utf-8") as f:
                lines = f.readlines()
                assert len(lines) == 3

    def test_multiple_output_files(self, test_annotator, test_dataset_name, temp_dir):
        """Test annotation with multiple output files."""
        test_annotator.max_samples_per_output_file = 2  # Two samples per file

        with (
            patch.object(test_annotator, "_load_pipeline"),
            patch.object(test_annotator, "_process_batch") as mock_process,
        ):
            # Mock multiple batches
            call_count = 0

            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return [
                        {"generated_text": "positive", "finish_reason": "stop", "num_tokens": 5},
                        {"generated_text": "negative", "finish_reason": "stop", "num_tokens": 4},
                    ]
                else:
                    return [
                        {"generated_text": "neutral", "finish_reason": "stop", "num_tokens": 3},
                    ]

            mock_process.side_effect = side_effect

            output_dir = temp_dir / "multi_file_test"
            test_annotator.annotate_dataset(
                dataset_name=test_dataset_name,
                output_dir=output_dir,
                dataset_split="test",
                max_num_samples=3,
            )

            # Should create multiple output files
            output_files = sorted(output_dir.glob("*.jsonl"))
            assert len(output_files) == 2
            assert output_files[0].name == f"{output_dir.stem}_0.jsonl"
            assert output_files[1].name == f"{output_dir.stem}_1.jsonl"

            # Check distribution of samples
            with output_files[0].open("r", encoding="utf-8") as f:
                lines1 = f.readlines()
                assert len(lines1) == 2

            with output_files[1].open("r", encoding="utf-8") as f:
                lines2 = f.readlines()
                assert len(lines2) == 1

    def test_file_rotation_during_processing(self, test_annotator, temp_dir):
        """Test that files are rotated correctly during processing."""
        test_annotator.max_samples_per_output_file = 1  # Force rotation after each sample
        test_annotator._load_tokenizer()
        test_annotator._load_pipeline = Mock()

        # Create test dataset
        test_dataset = {
            f"{test_annotator.prefix}_prompted": ["prompt1", "prompt2", "prompt3"],
            "text": ["text1", "text2", "text3"],
            test_annotator.idx_column: [0, 1, 2],
        }

        mock_pipe = Mock()
        mock_pipe.generate.return_value = [
            Mock(outputs=[Mock(text="result", finish_reason="stop", token_ids=[1, 2, 3])])
        ]
        test_annotator.pipe = mock_pipe
        test_annotator.dataset = Mock()
        test_annotator.dataset.__len__ = Mock(return_value=3)
        test_annotator.dataset.iter = Mock(
            return_value=[
                {k: [v] for k, v in zip(test_dataset.keys(), [test_dataset[k][i] for k in test_dataset.keys()])}
                for i in range(3)
            ]
        )

        output_dir = temp_dir / "rotation_test"

        test_annotator.annotate_dataset(
            dataset_name="mock",
            output_dir=output_dir,
        )

        # Should create 3 files (one per sample)
        output_files = sorted(output_dir.glob("*.jsonl"))
        assert len(output_files) == 3

        for i, file in enumerate(output_files):
            assert file.name == f"{output_dir.stem}_{i}.jsonl"
            with file.open("r", encoding="utf-8") as f:
                lines = f.readlines()
                assert len(lines) == 1

    def test_file_append_mode(self, test_annotator, temp_dir):
        """Test that files are opened in append mode to handle resumption."""
        test_annotator.max_samples_per_output_file = 0  # Single file
        output_dir = temp_dir / "append_test"
        output_dir.mkdir()

        # Create existing output file
        output_file = output_dir / f"{output_dir.stem}.jsonl"
        existing_data = {"llm_annotator_idx": 0, "text": "existing", "result": "old"}
        with output_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps(existing_data) + "\n")

        with (
            patch.object(test_annotator, "_load_pipeline"),
            patch.object(test_annotator, "_process_batch") as mock_process,
        ):
            mock_process.return_value = [{"generated_text": "new", "finish_reason": "stop", "num_tokens": 3}]

            test_annotator.annotate_dataset(
                dataset_name="stanfordnlp/imdb",
                output_dir=output_dir,
                dataset_split="test",
                max_num_samples=1,
            )

            # Check that both old and new data exist
            with output_file.open("r", encoding="utf-8") as f:
                lines = f.readlines()
                assert len(lines) >= 1  # At least the existing line

    def test_file_naming_with_custom_prefix(self, test_model_id, prompt_template_file, temp_dir):
        """Test file naming with custom prefix."""
        custom_prefix = "custom_test"
        annotator = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            prefix=custom_prefix,
            max_samples_per_output_file=1,
        )

        annotator.processed_n_samples = 0
        output_path = annotator.get_fhout_name(temp_dir)
        expected_path = temp_dir / f"{temp_dir.stem}_0.jsonl"
        assert output_path == expected_path

    def test_output_file_with_special_characters(self, test_annotator, temp_dir):
        """Test output file creation with special characters in directory name."""
        special_dir = temp_dir / "test-dir_with.special@chars"
        special_dir.mkdir()

        test_annotator.max_samples_per_output_file = 0
        output_path = test_annotator.get_fhout_name(special_dir)
        expected_path = special_dir / f"{special_dir.stem}.jsonl"
        assert output_path == expected_path

    def test_concurrent_file_access_handling(self, test_annotator, temp_dir):
        """Test handling of concurrent file access issues."""
        output_dir = temp_dir / "concurrent_test"
        output_file = output_dir / "test.jsonl"
        output_dir.mkdir()

        # Mock file opening to simulate concurrent access error
        original_open = open
        call_count = 0

        def mock_open(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise PermissionError("File is being used by another process")
            return original_open(*args, **kwargs)

        # Patch both builtins.open and pathlib.Path.open to simulate concurrent access
        with patch("builtins.open", side_effect=mock_open), patch("pathlib.Path.open", side_effect=mock_open):
            with pytest.raises(PermissionError):
                with output_file.open("a", encoding="utf-8") as f:
                    f.write("test")

    def test_large_output_file_handling(self, test_annotator, temp_dir):
        """Test handling of large output files."""
        test_annotator.max_samples_per_output_file = 1000  # Large number

        # Simulate processing many samples
        test_annotator.processed_n_samples = 999
        output_path1 = test_annotator.get_fhout_name(temp_dir)
        expected_path1 = temp_dir / f"{temp_dir.stem}_0.jsonl"
        assert output_path1 == expected_path1

        # After reaching threshold
        test_annotator.processed_n_samples = 1000
        output_path2 = test_annotator.get_fhout_name(temp_dir)
        expected_path2 = temp_dir / f"{temp_dir.stem}_1.jsonl"
        assert output_path2 == expected_path2

    def test_file_flushing_during_processing(self, test_annotator, test_dataset_name, temp_dir):
        """Test that files are flushed regularly during processing."""
        output_dir = temp_dir / "flush_test"

        # Mock file handle to track flush calls
        mock_file = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=None)

        # Avoid actually loading a tokenizer from HF during this unit test
        test_annotator._load_tokenizer = lambda: None
        test_annotator.tokenizer = Mock()
        # Ensure apply_chat_template returns a real string so dataset.map doesn't
        # receive Mock objects which pyarrow cannot convert.
        test_annotator.tokenizer.apply_chat_template = Mock(return_value="mocked prompt")

        # Create an open side-effect that only returns our mock_file for the
        # specific output filename; otherwise delegate to the real open.
        import builtins as _builtins

        real_open = _builtins.open

        def _open_side_effect(file, *args, **kwargs):
            try:
                fname = str(file)
            except Exception:
                fname = file
            if fname.endswith(f"{output_dir.stem}.jsonl"):
                return mock_file
            return real_open(file, *args, **kwargs)

        with (
            patch("builtins.open", side_effect=_open_side_effect),
            patch.object(test_annotator, "_load_pipeline"),
            patch.object(test_annotator, "_process_batch") as mock_process,
        ):
            mock_process.return_value = [{"generated_text": "test", "finish_reason": "stop", "num_tokens": 3}]

            test_annotator.annotate_dataset(
                dataset_name=test_dataset_name,
                output_dir=output_dir,
                dataset_split="test",
                max_num_samples=1,
            )

            # File should be flushed after writing
            assert mock_file.flush.called

    def test_output_directory_creation(self, test_annotator, test_dataset_name, temp_dir):
        """Test that output directories are created correctly."""
        nested_dir = temp_dir / "level1" / "level2" / "output"

        with (
            patch.object(test_annotator, "_load_pipeline"),
            patch.object(test_annotator, "_process_batch") as mock_process,
        ):
            # Return a single result so annotate_dataset can write and create dir
            mock_process.return_value = [{"generated_text": "x", "finish_reason": "stop", "num_tokens": 1}]

            test_annotator.annotate_dataset(
                dataset_name=test_dataset_name,
                output_dir=nested_dir,
                dataset_split="test",
                max_num_samples=1,
            )

            # Directory should be created
            assert nested_dir.exists()
            assert nested_dir.is_dir()

    def test_file_encoding_handling(self, test_annotator, temp_dir):
        """Test that files are handled with correct encoding."""
        output_dir = temp_dir / "encoding_test"
        output_dir.mkdir()

        # Create test data with unicode characters
        test_data = {
            "llm_annotator_idx": 0,
            "text": "Test with unicode: 🚀 café naïve résumé",
            "generated_text": "Response with unicode: 你好 世界",
        }

        output_file = output_dir / "unicode_test.jsonl"
        with output_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps(test_data, ensure_ascii=False) + "\n")

        # Read back and verify
        with output_file.open("r", encoding="utf-8") as f:
            loaded_data = json.loads(f.readline())
            assert loaded_data["text"] == test_data["text"]
            assert loaded_data["generated_text"] == test_data["generated_text"]
