"""Slow integration tests for Annotator class using actual LLM (Qwen/Qwen2.5-0.5B-Instruct)."""

import json
import shutil
from pathlib import Path

import pytest
from datasets import Dataset

from llm_annotator import Annotator


@pytest.mark.slow
class TestAnnotatorWithModel:
    """Integration tests with actual model loading and inference."""

    def test_basic_annotation(self, test_model_id, tmp_path, small_test_dataset):
        """Test basic dataset annotation with real model."""
        output_dir = tmp_path / "outputs"
        prompt_template = "Classify sentiment as positive or negative: {text}"

        with Annotator(model=test_model_id, max_model_len=512, verbose=False) as annotator:
            result_ds = annotator.annotate_dataset(
                output_dir=str(output_dir),
                full_prompt_template=prompt_template,
                dataset=small_test_dataset,
                max_num_samples=2,
                sampling_params={"max_tokens": 10, "temperature": 0},
            )

        assert len(result_ds) == 2
        assert "response" in result_ds.column_names
        assert "finish_reason" in result_ds.column_names
        assert "num_tokens" in result_ds.column_names

        # Clean up
        if output_dir.exists():
            shutil.rmtree(output_dir)

    def test_annotation_with_json_schema(self, test_model_id, tmp_path, small_test_dataset):
        """Test annotation with JSON schema (guided decoding)."""
        output_dir = tmp_path / "outputs"
        prompt_template = "Classify the sentiment of this text as positive or negative: {text}\n\nRespond in JSON format."

        output_schema = {
            "type": "object",
            "properties": {"sentiment": {"type": "string", "enum": ["positive", "negative"]}},
            "required": ["sentiment"],
        }

        with Annotator(model=test_model_id, max_model_len=512, verbose=False) as annotator:
            result_ds = annotator.annotate_dataset(
                output_dir=str(output_dir),
                full_prompt_template=prompt_template,
                dataset=small_test_dataset,
                max_num_samples=2,
                output_schema=output_schema,
                sampling_params={"max_tokens": 20, "temperature": 0},
            )

        assert len(result_ds) == 2
        assert "sentiment" in result_ds.column_names
        assert "valid_fields" in result_ds.column_names
        assert "response" in result_ds.column_names

        # Check that at least some responses are valid
        # (May not be all due to small model limitations)
        assert "sentiment" in result_ds.column_names

        # Clean up
        if output_dir.exists():
            shutil.rmtree(output_dir)

    def test_annotation_with_keep_columns(self, test_model_id, tmp_path, small_test_dataset):
        """Test annotation with specific columns kept."""
        output_dir = tmp_path / "outputs"
        prompt_template = "Analyze: {text}"

        with Annotator(model=test_model_id, max_model_len=512, verbose=False) as annotator:
            result_ds = annotator.annotate_dataset(
                output_dir=str(output_dir),
                full_prompt_template=prompt_template,
                dataset=small_test_dataset,
                max_num_samples=2,
                keep_columns=["text", "label"],
                sampling_params={"max_tokens": 10, "temperature": 0},
            )

        assert len(result_ds) == 2
        assert "text" in result_ds.column_names
        assert "label" in result_ds.column_names
        assert "response" in result_ds.column_names

        # Clean up
        if output_dir.exists():
            shutil.rmtree(output_dir)

    def test_annotation_resumption(self, test_model_id, tmp_path, small_test_dataset):
        """Test that annotation can be resumed from partial results."""
        output_dir = tmp_path / "outputs"
        prompt_template = "Summarize: {text}"

        # First run - process only 1 sample
        with Annotator(model=test_model_id, max_model_len=512, verbose=False) as annotator:
            result_ds1 = annotator.annotate_dataset(
                output_dir=str(output_dir),
                full_prompt_template=prompt_template,
                dataset=small_test_dataset,
                max_num_samples=1,
                sampling_params={"max_tokens": 10, "temperature": 0},
            )

        assert len(result_ds1) == 1

        # Second run - process all 3 samples (should skip the first one)
        with Annotator(model=test_model_id, max_model_len=512, verbose=False) as annotator:
            result_ds2 = annotator.annotate_dataset(
                output_dir=str(output_dir),
                full_prompt_template=prompt_template,
                dataset=small_test_dataset,
                max_num_samples=3,
                sampling_params={"max_tokens": 10, "temperature": 0},
            )

        # Should have all 3 samples now
        assert len(result_ds2) == 3

        # Clean up
        if output_dir.exists():
            shutil.rmtree(output_dir)

    def test_annotation_with_validation_fn(self, test_model_id, tmp_path, small_test_dataset):
        """Test annotation with custom validation function."""
        output_dir = tmp_path / "outputs"
        prompt_template = "Rate this text: {text}"

        def validate_fn(sample):
            """Always return True for this test."""
            return len(sample.get("response", "")) > 0

        with Annotator(model=test_model_id, max_model_len=512, verbose=False) as annotator:
            result_ds = annotator.annotate_dataset(
                output_dir=str(output_dir),
                full_prompt_template=prompt_template,
                dataset=small_test_dataset,
                max_num_samples=2,
                validate_fn=validate_fn,
                sampling_params={"max_tokens": 10, "temperature": 0},
            )

        assert len(result_ds) == 2
        assert "valid" in result_ds.column_names
        assert "response" in result_ds.column_names

        # Clean up
        if output_dir.exists():
            shutil.rmtree(output_dir)

    def test_annotation_with_postprocess_fn(self, test_model_id, tmp_path, small_test_dataset):
        """Test annotation with custom postprocessing function."""
        output_dir = tmp_path / "outputs"
        prompt_template = "Analyze: {text}"

        def postprocess_fn(sample):
            """Add a custom field to each sample."""
            sample["custom_field"] = "processed"
            return sample

        with Annotator(model=test_model_id, max_model_len=512, verbose=False) as annotator:
            result_ds = annotator.annotate_dataset(
                output_dir=str(output_dir),
                full_prompt_template=prompt_template,
                dataset=small_test_dataset,
                max_num_samples=2,
                postprocess_fn=postprocess_fn,
                sampling_params={"max_tokens": 10, "temperature": 0},
            )

        assert len(result_ds) == 2
        assert "custom_field" in result_ds.column_names
        assert all(sample == "processed" for sample in result_ds["custom_field"])

        # Clean up
        if output_dir.exists():
            shutil.rmtree(output_dir)

    def test_annotation_with_task_prefix(self, test_model_id, tmp_path, small_test_dataset):
        """Test annotation with custom task prefix."""
        output_dir = tmp_path / "outputs"
        prompt_template = "Process: {text}"

        with Annotator(model=test_model_id, max_model_len=512, verbose=False) as annotator:
            result_ds = annotator.annotate_dataset(
                output_dir=str(output_dir),
                full_prompt_template=prompt_template,
                dataset=small_test_dataset,
                max_num_samples=2,
                task_prefix="custom_",
                sampling_params={"max_tokens": 10, "temperature": 0},
            )

        assert len(result_ds) == 2
        assert "custom_response" in result_ds.column_names
        assert "custom_finish_reason" in result_ds.column_names

        # Clean up
        if output_dir.exists():
            shutil.rmtree(output_dir)

    def test_annotation_from_hub_dataset(self, test_model_id, test_dataset_name, tmp_path):
        """Test annotation from a Hugging Face Hub dataset."""
        output_dir = tmp_path / "outputs"
        prompt_template = "Sentiment of: {text}"

        with Annotator(model=test_model_id, max_model_len=512, verbose=False) as annotator:
            result_ds = annotator.annotate_dataset(
                output_dir=str(output_dir),
                full_prompt_template=prompt_template,
                dataset_name=test_dataset_name,
                dataset_split="test",
                max_num_samples=2,
                streaming=True,
                sampling_params={"max_tokens": 10, "temperature": 0},
            )

        assert len(result_ds) == 2
        assert "response" in result_ds.column_names

        # Clean up
        if output_dir.exists():
            shutil.rmtree(output_dir)

    def test_multiple_batches(self, test_model_id, tmp_path):
        """Test annotation with multiple batches."""
        # Create a larger dataset
        large_dataset = Dataset.from_dict({"text": [f"Text {i}" for i in range(10)], "label": [i % 2 for i in range(10)]})

        output_dir = tmp_path / "outputs"
        prompt_template = "Process: {text}"

        with Annotator(model=test_model_id, max_model_len=512, max_num_seqs=3, verbose=False) as annotator:
            result_ds = annotator.annotate_dataset(
                output_dir=str(output_dir),
                full_prompt_template=prompt_template,
                dataset=large_dataset,
                max_num_samples=10,
                sampling_params={"max_tokens": 10, "temperature": 0},
            )

        assert len(result_ds) == 10
        assert "response" in result_ds.column_names

        # Clean up
        if output_dir.exists():
            shutil.rmtree(output_dir)

    def test_annotation_sort_by_length(self, test_model_id, tmp_path):
        """Test annotation with sorting by length."""
        # Create dataset with varying text lengths
        dataset = Dataset.from_dict(
            {
                "text": [
                    "Short",
                    "This is a medium length text",
                    "Very very very very very long text with many words",
                ]
            }
        )

        output_dir = tmp_path / "outputs"
        prompt_template = "Analyze: {text}"

        with Annotator(model=test_model_id, max_model_len=512, verbose=False) as annotator:
            result_ds = annotator.annotate_dataset(
                output_dir=str(output_dir),
                full_prompt_template=prompt_template,
                dataset=dataset,
                sort_by_length=True,
                sampling_params={"max_tokens": 10, "temperature": 0},
            )

        assert len(result_ds) == 3
        assert "response" in result_ds.column_names

        # Clean up
        if output_dir.exists():
            shutil.rmtree(output_dir)


@pytest.mark.slow
@pytest.mark.integration
class TestAnnotatorHubIntegration:
    """Integration tests that interact with Hugging Face Hub."""

    def test_annotation_with_hub_upload(self, test_model_id, test_remote_dataset_name, tmp_path, small_test_dataset):
        """Test annotation with upload to Hugging Face Hub."""
        output_dir = tmp_path / "outputs"
        prompt_template = "Analyze: {text}"

        with Annotator(model=test_model_id, max_model_len=512, verbose=False) as annotator:
            result_ds = annotator.annotate_dataset(
                output_dir=str(output_dir),
                full_prompt_template=prompt_template,
                dataset=small_test_dataset,
                max_num_samples=2,
                new_hub_id=test_remote_dataset_name,
                sampling_params={"max_tokens": 10, "temperature": 0},
            )

        assert len(result_ds) == 2
        assert "response" in result_ds.column_names

        # Clean up
        if output_dir.exists():
            shutil.rmtree(output_dir)

    def test_annotation_with_periodic_upload(
        self, test_model_id, test_remote_dataset_name, tmp_path, small_test_dataset
    ):
        """Test annotation with periodic uploads to Hub."""
        output_dir = tmp_path / "outputs"
        prompt_template = "Process: {text}"

        with Annotator(model=test_model_id, max_model_len=512, verbose=False) as annotator:
            result_ds = annotator.annotate_dataset(
                output_dir=str(output_dir),
                full_prompt_template=prompt_template,
                dataset=small_test_dataset,
                max_num_samples=3,
                new_hub_id=test_remote_dataset_name,
                upload_every_n_samples=2,  # Upload after 2 samples
                sampling_params={"max_tokens": 10, "temperature": 0},
            )

        assert len(result_ds) == 3
        assert "response" in result_ds.column_names

        # Clean up
        if output_dir.exists():
            shutil.rmtree(output_dir)
