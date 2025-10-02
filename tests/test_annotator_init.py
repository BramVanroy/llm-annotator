"""Test the Annotator class initialization and configuration."""

import json

import pytest

from llm_annotator.annotator import Annotator


class TestAnnotatorInitializationUnitTests:
    """Test Annotator class initialization and validation."""

    def test_annotator_prompt_template_file(self, test_model_id, prompt_template_file):
        """Test that the Annotator initializes correctly with a prompt template file."""
        anno = Annotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
        )
        assert anno.model_id == test_model_id
        assert anno.prompt_template_file == prompt_template_file
        assert anno.prompt_template == prompt_template_file.read_text(encoding="utf-8")

    def test_annotator_prompt_template(self, test_model_id, prompt_template_file):
        """Test that the Annotator initializes correctly with a prompt template file."""
        anno = Annotator(
            model_id=test_model_id,
            prompt_template=prompt_template_file.read_text(encoding="utf-8"),
        )
        assert anno.prompt_template_file is None
        assert anno.prompt_template == prompt_template_file.read_text(encoding="utf-8")

    def test_annotator_both_prompt_template_and_file(self, test_model_id, prompt_template_file):
        """Test that the Annotator raises an error if both prompt_template and prompt_template_file are provided."""
        with pytest.raises(ValueError, match="Only one of prompt_template_file or prompt_template should be provided"):
            Annotator(
                model_id=test_model_id,
                prompt_template_file=prompt_template_file,
                prompt_template="Some template",
            )

    def test_annotator_no_prompt_template(self, test_model_id):
        """Test that the Annotator raises an error if neither prompt_template nor prompt_template_file are provided."""
        with pytest.raises(ValueError, match="Either prompt_template_file or prompt_template must be provided"):
            Annotator(
                model_id=test_model_id,
            )

    def test_annotator_output_schema_file(self, test_model_id, prompt_template_file, json_schema_file):
        """Test that the Annotator initializes correctly with an output schema file."""
        anno = Annotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            output_schema_file=json_schema_file,
        )
        assert anno.output_schema_file == json_schema_file
        assert anno.output_schema == json.loads(json_schema_file.read_text(encoding="utf-8"))

    def test_annotator_output_schema(self, test_model_id, prompt_template_file, json_schema_file):
        """Test that the Annotator initializes correctly with an output schema file."""
        anno = Annotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            output_schema=json.loads(json_schema_file.read_text(encoding="utf-8")),
        )
        assert anno.output_schema_file is None
        assert anno.output_schema == json.loads(json_schema_file.read_text(encoding="utf-8"))

    def test_annotator_both_output_schema_and_file(self, test_model_id, prompt_template_file, json_schema_file):
        """Test that the Annotator raises an error if both output_schema and output_schema_file are provided."""
        with pytest.raises(ValueError, match="Only one of output_schema_file or output_schema should be provided"):
            Annotator(
                model_id=test_model_id,
                prompt_template_file=prompt_template_file,
                output_schema_file=json_schema_file,
                output_schema={"type": "object"},
            )

    def test_keep_columns_set(self, test_model_id, prompt_template_file):
        """Test that keep_columns is set correctly."""

        # List input
        anno = Annotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            keep_columns=["text", "sentiment"],
        )
        assert anno.keep_columns == {"text", "sentiment", anno.idx_column}

        # Tuple input
        anno = Annotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            keep_columns=("text", "sentiment"),
        )
        assert anno.keep_columns == {"text", "sentiment", anno.idx_column}

        # String input
        anno = Annotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            keep_columns="text",
        )
        assert anno.keep_columns == {"text", anno.idx_column}

        # None (default)
        anno = Annotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
        )
        assert anno.keep_columns == {anno.idx_column}

        # False (same as None)
        anno = Annotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
        )
        assert anno.keep_columns == {anno.idx_column}

        # True (will keep all)
        anno = Annotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            keep_columns=True,
        )
        assert anno.keep_columns is True

        str_generator = (str(i) for i in range(5))
        anno = Annotator(
            model_id=test_model_id,
            prompt_template_file=prompt_template_file,
            keep_columns=str_generator,
        )
        assert anno.keep_columns == {anno.idx_column}.union({str(i) for i in range(5)})

    def test_invalid_keep_columns(self, test_model_id, prompt_template_file):
        """Test that invalid keep_columns raises an error."""
        with pytest.raises(TypeError, match="keep_columns must be None, True, a string, or a collection of strings"):
            Annotator(
                model_id=test_model_id,
                prompt_template_file=prompt_template_file,
                keep_columns=123,
            )
