"""Test dataset loading and preprocessing functionality."""

import json

import pytest
from conftest import MockAnnotator
from datasets import Dataset


class TestDatasetOperations:
    """Test dataset loading, preprocessing, and caching."""

    def test_dataset_loading_basic(self, test_annotator, test_dataset_name, temp_dir):
        """Test basic dataset loading functionality."""
        test_annotator._load_tokenizer()
        test_annotator._load_dataset(
            dataset_name=test_dataset_name,
            pdout=temp_dir,
            dataset_split="test",
            max_num_samples=3,
        )

        assert test_annotator.dataset is not None
        assert len(test_annotator.dataset) == 3
        assert test_annotator.dataset_split == "test"
        assert f"{test_annotator.prefix}_prompted" in test_annotator.dataset.column_names
        assert test_annotator.idx_column in test_annotator.dataset.column_names

    def test_dataset_caching(self, test_annotator, test_dataset_name, temp_dir):
        """Test that dataset caching works correctly."""
        test_annotator._load_tokenizer()

        # First load - should create cache
        test_annotator._load_dataset(
            dataset_name=test_dataset_name,
            pdout=temp_dir,
            dataset_split="test",
            max_num_samples=3,
        )

        cached_path = temp_dir / f"{test_annotator.prefix}_cached_input_dataset"
        assert cached_path.exists()

        first_dataset = test_annotator.dataset

        # Second load - should use cache
        test_annotator.dataset = None  # Reset
        test_annotator._load_dataset(
            dataset_name=test_dataset_name,
            pdout=temp_dir,
            dataset_split="test",
            max_num_samples=3,
        )

        assert test_annotator.dataset is not None
        assert len(test_annotator.dataset) == len(first_dataset)

    def test_no_use_cached_input_dataset_option(self, test_model_id, prompt_template_file, test_dataset_name, temp_dir):
        """Test that use_cached_input_dataset=False option prevents caching."""
        annotator = MockAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            use_cached_input_dataset=False,
        )

        annotator._load_tokenizer()

        # First load
        annotator._load_dataset(
            dataset_name=test_dataset_name,
            pdout=temp_dir,
            dataset_split="test",
            max_num_samples=3,
        )

        # Create a fake cache to ensure it's ignored
        cached_path = temp_dir / f"{annotator.prefix}_cached_input_dataset"
        cached_path.mkdir(exist_ok=True)
        fake_dataset = Dataset.from_dict({"fake": ["data"]})
        fake_dataset.save_to_disk(cached_path)

        # Second load - should ignore cache
        annotator.dataset = None
        annotator._load_dataset(
            dataset_name=test_dataset_name,
            pdout=temp_dir,
            dataset_split="test",
            max_num_samples=3,
        )

        assert "fake" not in annotator.dataset.column_names

    def test_streaming_dataset_loading(self, test_annotator, test_dataset_name, temp_dir):
        """Test streaming dataset loading functionality."""
        test_annotator._load_tokenizer()
        test_annotator._load_dataset(
            dataset_name=test_dataset_name,
            pdout=temp_dir,
            dataset_split="test",
            streaming=True,
            max_num_samples=3,
        )

        assert test_annotator.dataset is not None
        assert len(test_annotator.dataset) == 3
        assert test_annotator.streaming is True

    def test_streaming_without_max_samples_error(self, test_annotator, test_dataset_name, temp_dir):
        """Test that streaming without max_num_samples raises an error."""
        test_annotator._load_tokenizer()

        with pytest.raises(ValueError, match="Streaming mode requires max_num_samples"):
            test_annotator._load_dataset(
                dataset_name=test_dataset_name,
                pdout=temp_dir,
                dataset_split="test",
                streaming=True,
                max_num_samples=None,
            )

    def test_dataset_shuffling(self, test_annotator, test_dataset_name, temp_dir):
        """Test dataset shuffling with seed."""
        test_annotator._load_tokenizer()

        # Load with seed
        test_annotator._load_dataset(
            dataset_name=test_dataset_name,
            pdout=temp_dir,
            dataset_split="test",
            max_num_samples=3,
            shuffle_seed=42,
        )

        first_order = test_annotator.dataset["text"][:3]

        # Load again with same seed
        test_annotator.dataset = None
        test_annotator.dataset_cache = False  # Force reload
        test_annotator._load_dataset(
            dataset_name=test_dataset_name,
            pdout=temp_dir,
            dataset_split="test",
            max_num_samples=3,
            shuffle_seed=42,
        )

        second_order = test_annotator.dataset["text"][:3]
        assert first_order == second_order

    def test_skip_existing_samples(self, test_annotator, test_dataset_name, temp_dir):
        """Test that existing processed samples are skipped."""
        test_annotator._load_tokenizer()

        # Create fake output file with some processed samples
        output_file = temp_dir / "test.jsonl"
        fake_outputs = [
            {"idx": 0, "text": "sample 1", "result": "positive"},
            {"idx": 1, "text": "sample 2", "result": "negative"},
        ]

        with output_file.open("w", encoding="utf-8") as f:
            for output in fake_outputs:
                f.write(json.dumps(output) + "\n")

        test_annotator._load_dataset(
            dataset_name=test_dataset_name,
            pdout=temp_dir,
            dataset_split="test",
            max_num_samples=3,
        )

        # Should only have 1 sample left (3 total - 2 already processed)
        assert len(test_annotator.dataset) == 1
        assert test_annotator.processed_n_samples == 2

    def test_get_skip_idxs(self, test_annotator, temp_dir):
        """Test _get_skip_idxs method directly."""
        # Create multiple output files
        file1 = temp_dir / "output1.jsonl"
        file2 = temp_dir / "output2.jsonl"

        outputs1 = [
            {"idx": 0, "result": "positive"},
            {"idx": 1, "result": "negative"},
        ]

        outputs2 = [
            {"idx": 2, "result": "neutral"},
            {"idx": 3, "result": "positive"},
        ]

        with file1.open("w", encoding="utf-8") as f:
            for output in outputs1:
                f.write(json.dumps(output) + "\n")

        with file2.open("w", encoding="utf-8") as f:
            for output in outputs2:
                f.write(json.dumps(output) + "\n")

        skip_idxs = test_annotator._get_skip_idxs(temp_dir)
        assert skip_idxs == {0, 1, 2, 3}

    def test_get_skip_idxs_with_filters(self, test_annotator, temp_dir):
        """Test _get_skip_idxs with dataset split and config filters."""
        test_annotator.dataset_split = "test"
        test_annotator.dataset_config = "binary"

        output_file = temp_dir / "mixed.jsonl"
        outputs = [
            {"idx": 0, "dataset_split": "test", "dataset_config": "binary", "result": "positive"},
            {"idx": 1, "dataset_split": "train", "dataset_config": "binary", "result": "negative"},
            {"idx": 2, "dataset_split": "test", "dataset_config": "multiclass", "result": "neutral"},
            {"idx": 3, "dataset_split": "test", "dataset_config": "binary", "result": "positive"},
        ]

        with output_file.open("w", encoding="utf-8") as f:
            for output in outputs:
                f.write(json.dumps(output) + "\n")

        skip_idxs = test_annotator._get_skip_idxs(temp_dir)
        # Should only skip idx 0 and 3 (matching both split and config)
        assert skip_idxs == {0, 3}

    def test_preprocess_and_postprocess_hooks(self, test_model_id, prompt_template_file, temp_dir):
        """Test that preprocessing and postprocessing hooks are called."""

        class CustomAnnotator(MockAnnotator):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.preprocess_called = False
                self.postprocess_called = False

            def _preprocess_dataset(self, dataset):
                self.preprocess_called = True
                # Add a custom column
                return dataset.map(lambda x: {**x, "custom_field": "preprocessed"})

            def _postprocess_dataset(self, dataset):
                self.postprocess_called = True
                # Filter to only keep certain samples
                return dataset.filter(lambda x: len(x["text"]) > 10)

        annotator = CustomAnnotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
        )

        annotator._load_tokenizer()
        annotator._load_dataset(
            dataset_name="stanfordnlp/imdb",
            pdout=temp_dir,
            dataset_split="test",
            max_num_samples=3,
        )

        assert annotator.preprocess_called
        assert annotator.postprocess_called
        assert "custom_field" in annotator.dataset.column_names

    def test_empty_output_files_handling(self, test_annotator, temp_dir):
        """Test that empty output files are handled correctly."""
        # Create empty files
        empty_file1 = temp_dir / "empty1.jsonl"
        empty_file2 = temp_dir / "empty2.jsonl"
        empty_file1.touch()
        empty_file2.touch()

        # Create non-empty file
        normal_file = temp_dir / "normal.jsonl"
        with normal_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"idx": 0, "result": "test"}) + "\n")

        skip_idxs = test_annotator._get_skip_idxs(temp_dir)
        assert skip_idxs == {0}  # Only from the non-empty file
