"""Fast unit tests for utils.py functions."""

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from llm_annotator.utils import (
    convert_int_to_str,
    count_lines,
    ensure_returns_bool,
    ensure_returns_dict,
    get_hash,
    remove_empty_jsonl_files,
    retry,
    yield_jsonl_robust,
)


class TestGetHash:
    """Test get_hash function."""

    def test_get_hash_basic(self):
        """Test basic hash generation."""
        text = "hello world"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert get_hash(text) == expected

    def test_get_hash_empty_string(self):
        """Test hash of empty string."""
        text = ""
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert get_hash(text) == expected

    def test_get_hash_consistency(self):
        """Test that same input produces same hash."""
        text = "test data"
        hash1 = get_hash(text)
        hash2 = get_hash(text)
        assert hash1 == hash2

    def test_get_hash_different_inputs(self):
        """Test that different inputs produce different hashes."""
        hash1 = get_hash("text1")
        hash2 = get_hash("text2")
        assert hash1 != hash2


class TestConvertIntToStr:
    """Test convert_int_to_str function."""

    def test_convert_small_numbers(self):
        """Test conversion of small numbers."""
        assert convert_int_to_str(0) == "0"
        assert convert_int_to_str(1) == "1"
        assert convert_int_to_str(999) == "999"

    def test_convert_thousands(self):
        """Test conversion of thousands."""
        assert convert_int_to_str(1000) == "1K"
        assert convert_int_to_str(1234) == "1.2K"
        assert convert_int_to_str(5000) == "5K"
        assert convert_int_to_str(999_999) == "1000K"

    def test_convert_millions(self):
        """Test conversion of millions."""
        assert convert_int_to_str(1_000_000) == "1M"
        assert convert_int_to_str(1_234_567) == "1.2M"
        assert convert_int_to_str(5_000_000) == "5M"

    def test_convert_billions(self):
        """Test conversion of billions."""
        assert convert_int_to_str(1_000_000_000) == "1B"
        assert convert_int_to_str(1_234_567_890) == "1.2B"
        assert convert_int_to_str(5_000_000_000) == "5B"

    def test_convert_edge_cases(self):
        """Test edge cases."""
        assert convert_int_to_str(1_001) == "1K"
        assert convert_int_to_str(1_001_000) == "1M"


class TestRetry:
    """Test retry decorator."""

    def test_retry_success_first_try(self):
        """Test function that succeeds on first try."""
        call_count = {"count": 0}

        @retry(num_retries=3, sleep_time_s=0)
        def successful_func():
            call_count["count"] += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count["count"] == 1

    def test_retry_success_after_failures(self):
        """Test function that succeeds after some failures."""
        call_count = {"count": 0}

        @retry(num_retries=3, sleep_time_s=0)
        def eventually_successful_func():
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise ValueError("Temporary error")
            return "success"

        result = eventually_successful_func()
        assert result == "success"
        assert call_count["count"] == 3

    def test_retry_failure_after_all_retries(self):
        """Test function that fails after all retries."""
        call_count = {"count": 0}

        @retry(num_retries=2, sleep_time_s=0)
        def always_failing_func():
            call_count["count"] += 1
            raise ValueError("Permanent error")

        with pytest.raises(ValueError, match="Permanent error"):
            always_failing_func()
        assert call_count["count"] == 3  # 1 initial + 2 retries


class TestYieldJsonlRobust:
    """Test yield_jsonl_robust function."""

    def test_yield_jsonl_basic(self, tmp_path):
        """Test basic JSONL reading."""
        jsonl_file = tmp_path / "test.jsonl"
        data = [
            {"id": 1, "text": "hello"},
            {"id": 2, "text": "world"},
        ]
        with jsonl_file.open("w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        results = list(yield_jsonl_robust([jsonl_file], disable_tqdm=True))
        assert len(results) == 2
        assert results[0] == {"id": 1, "text": "hello"}
        assert results[1] == {"id": 2, "text": "world"}

    def test_yield_jsonl_keep_columns(self, tmp_path):
        """Test reading with column filtering."""
        jsonl_file = tmp_path / "test.jsonl"
        data = [
            {"id": 1, "text": "hello", "extra": "data"},
            {"id": 2, "text": "world", "extra": "more"},
        ]
        with jsonl_file.open("w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        results = list(yield_jsonl_robust([jsonl_file], keep_columns=["id", "text"], disable_tqdm=True))
        assert len(results) == 2
        assert results[0] == {"id": 1, "text": "hello"}
        assert "extra" not in results[0]

    def test_yield_jsonl_skip_corrupt_lines(self, tmp_path):
        """Test that corrupt lines are skipped."""
        jsonl_file = tmp_path / "test.jsonl"
        with jsonl_file.open("w") as f:
            f.write('{"id": 1, "text": "hello"}\n')
            f.write('{"invalid json\n')  # Corrupt line
            f.write('{"id": 2, "text": "world"}\n')

        results = list(yield_jsonl_robust([jsonl_file], disable_tqdm=True))
        assert len(results) == 2
        assert results[0]["id"] == 1
        assert results[1]["id"] == 2

    def test_yield_jsonl_deduplicate(self, tmp_path):
        """Test deduplication functionality."""
        jsonl_file = tmp_path / "test.jsonl"
        data = [
            {"id": 1, "text": "hello"},
            {"id": 2, "text": "hello"},  # Duplicate text
            {"id": 3, "text": "world"},
        ]
        with jsonl_file.open("w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        results = list(yield_jsonl_robust([jsonl_file], deduplicate_on="text", disable_tqdm=True))
        assert len(results) == 2
        assert results[0]["text"] == "hello"
        assert results[1]["text"] == "world"

    def test_yield_jsonl_empty_file(self, tmp_path):
        """Test handling of empty files."""
        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.touch()

        results = list(yield_jsonl_robust([jsonl_file], disable_tqdm=True))
        assert len(results) == 0

    def test_yield_jsonl_multiple_files(self, tmp_path):
        """Test reading from multiple files."""
        file1 = tmp_path / "test1.jsonl"
        file2 = tmp_path / "test2.jsonl"

        with file1.open("w") as f:
            f.write('{"id": 1}\n')
        with file2.open("w") as f:
            f.write('{"id": 2}\n')

        results = list(yield_jsonl_robust([file1, file2], disable_tqdm=True))
        assert len(results) == 2


class TestCountLines:
    """Test count_lines function."""

    def test_count_lines_basic(self, tmp_path):
        """Test basic line counting."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\nline3\n")

        count = count_lines(test_file)
        assert count == 3

    def test_count_lines_empty_file(self, tmp_path):
        """Test counting lines in empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.touch()

        count = count_lines(test_file)
        assert count == 0

    def test_count_lines_single_line(self, tmp_path):
        """Test counting single line without newline."""
        test_file = tmp_path / "single.txt"
        test_file.write_text("single line")

        count = count_lines(test_file)
        assert count == 1


class TestRemoveEmptyJsonlFiles:
    """Test remove_empty_jsonl_files function."""

    def test_remove_empty_files(self, tmp_path):
        """Test removal of empty JSONL files."""
        # Create mix of empty and non-empty files
        empty_file = tmp_path / "empty.jsonl"
        empty_file.touch()

        non_empty_file = tmp_path / "data.jsonl"
        non_empty_file.write_text('{"id": 1}\n')

        # Also create a non-jsonl file to ensure it's not removed
        other_file = tmp_path / "other.txt"
        other_file.touch()

        removed = remove_empty_jsonl_files(tmp_path)

        assert len(removed) == 1
        assert empty_file in removed
        assert not empty_file.exists()
        assert non_empty_file.exists()
        assert other_file.exists()

    def test_remove_no_empty_files(self, tmp_path):
        """Test when there are no empty files."""
        non_empty = tmp_path / "data.jsonl"
        non_empty.write_text('{"id": 1}\n')

        removed = remove_empty_jsonl_files(tmp_path)
        assert len(removed) == 0
        assert non_empty.exists()


class TestEnsureReturnsBool:
    """Test ensure_returns_bool function."""

    def test_ensure_returns_bool_success(self):
        """Test function that returns bool."""

        def returns_bool():
            return True

        result = ensure_returns_bool(returns_bool)
        assert result is True

    def test_ensure_returns_bool_false(self):
        """Test function that returns False."""

        def returns_false():
            return False

        result = ensure_returns_bool(returns_false)
        assert result is False

    def test_ensure_returns_bool_failure(self):
        """Test function that doesn't return bool."""

        def returns_string():
            return "not a bool"

        with pytest.raises(TypeError, match="should return a bool"):
            ensure_returns_bool(returns_string)

    def test_ensure_returns_bool_with_args(self):
        """Test function with arguments."""

        def returns_bool_with_args(x, y):
            return x > y

        result = ensure_returns_bool(returns_bool_with_args, 5, 3)
        assert result is True


class TestEnsureReturnsDict:
    """Test ensure_returns_dict function."""

    def test_ensure_returns_dict_success(self):
        """Test function that returns dict."""

        def returns_dict():
            return {"key": "value"}

        result = ensure_returns_dict(returns_dict)
        assert result == {"key": "value"}

    def test_ensure_returns_dict_empty(self):
        """Test function that returns empty dict."""

        def returns_empty_dict():
            return {}

        result = ensure_returns_dict(returns_empty_dict)
        assert result == {}

    def test_ensure_returns_dict_failure(self):
        """Test function that doesn't return dict."""

        def returns_list():
            return ["not", "a", "dict"]

        with pytest.raises(TypeError, match="should return a dict"):
            ensure_returns_dict(returns_list)

    def test_ensure_returns_dict_with_args(self):
        """Test function with arguments."""

        def returns_dict_with_args(key, value):
            return {key: value}

        result = ensure_returns_dict(returns_dict_with_args, "name", "test")
        assert result == {"name": "test"}
