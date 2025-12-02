# LLM Annotator Test Suite

This directory contains comprehensive tests for the `llm-annotator` package. Tests are organized into **fast tests** and **slow tests** to optimize development workflow and maximize coverage.

## Test Organization

### Fast Tests (No LLM Required) - 44 Tests
These tests run quickly (< 1 second) and don't require loading a language model or heavy dependencies:

- **`test_utils.py`** (31 tests): Unit tests for utility functions
  - Hash generation (SHA256)
  - Number formatting (converting ints to K/M/B notation)
  - Retry decorator with exponential backoff
  - Robust JSONL file reading with error handling
  - JSONL deduplication
  - Column filtering
  - Line counting
  - Empty file removal
  - Type validation functions (ensure_returns_bool, ensure_returns_dict)

- **`test_annotator_fast.py`** (17 tests): Unit tests for Annotator class (mocked)
  - Initialization with various parameters
  - Context manager behavior (`with` statement)
  - Message/prompt creation and templating
  - Output processing with/without JSON schemas
  - File naming strategies (with sample limits)
  - Basic validation logic

### Slow Tests (Requires LLM) - 14 Tests
These tests require loading and running the `Qwen/Qwen2.5-0.5B-Instruct` model (requires ~2GB RAM and GPU access):

- **`test_annotator_slow.py`** (14 integration tests): End-to-end testing with actual model inference
  - Basic dataset annotation pipeline
  - JSON schema / guided decoding
  - Column selection and filtering
  - Annotation resumption from partial results
  - Custom validation functions
  - Custom postprocessing functions
  - Task prefixes for multi-task scenarios
  - Multiple batch processing
  - Dataset sorting by prompt length
  - Loading from Hugging Face Hub
  - Uploading to Hugging Face Hub (requires auth - marked `@pytest.mark.integration`)
  - Periodic Hub uploads during processing

## Running Tests

### Prerequisites

**For Fast Tests Only:**
```bash
pip install pytest pytest-cov datasets
```

**For All Tests (including slow/integration):**
```bash
pip install pytest pytest-cov datasets torch vllm transformers
# For GPU support, ensure CUDA is available
```

### Quick Start

```bash
# Run only fast tests (recommended for development)
pytest -m "not slow"

# Run all tests including slow ones
pytest

# Run only slow tests
pytest -m slow

# Run with coverage report
pytest --cov=llm_annotator --cov-report=term-missing
```

### Specific Test Execution

```bash
# Run specific test file
pytest tests/test_utils.py
pytest tests/test_annotator_fast.py
pytest tests/test_annotator_slow.py

# Run specific test class
pytest tests/test_utils.py::TestGetHash
pytest tests/test_annotator_slow.py::TestAnnotatorWithModel

# Run specific test function
pytest tests/test_utils.py::TestGetHash::test_get_hash_basic
```

### Skip Integration Tests

Integration tests require Hugging Face authentication and will create/delete test datasets:

```bash
# Skip integration tests (no Hub uploads/downloads)
pytest -m "not integration"

# Run only integration tests (requires: huggingface-cli login)
pytest -m integration
```

## Test Markers

The test suite uses pytest markers to categorize tests:

- `@pytest.mark.slow`: Tests requiring model loading (2-10 seconds each)
- `@pytest.mark.integration`: Tests interacting with Hugging Face Hub (requires authentication)

Configure these in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (may require model loading or longer runtime)",
    "integration: marks tests that interact with external systems",
]
```

## Fixtures (conftest.py)

The test suite provides reusable fixtures:

### Session-Scoped Fixtures
- `test_model_id`: Returns "Qwen/Qwen2.5-0.5B-Instruct"
- `test_dataset_name`: Returns "stanfordnlp/imdb"  
- `test_remote_dataset_name`: Returns "{username}/llm_annotator_test_ds" (requires HF auth)
- `temp_dir`: Temporary directory for test outputs (auto-cleaned)
- `small_test_dataset`: Small in-memory dataset (3 samples)
- `prompt_template_file`: Example prompt template file
- `json_schema_file`: Example JSON schema for guided decoding
- `test_annotator`: Pre-configured Annotator instance
- `hf_username`: Current HF username (None if not logged in)
- `cleanup_remote_datasets`: Auto-cleanup of test datasets on Hub (runs at end)
- `quiet_vllm_logging`: Suppresses vLLM logging during tests

## Coverage Summary

### Current Coverage (Fast Tests Only)
- **utils.py**: 97% coverage (95/98 statements)
- **annotator.py**: 24% coverage (partial - focuses on initialization and core logic)
- **Overall**: 40% coverage with fast tests alone

### Expected Coverage (All Tests)
With slow tests enabled (requiring torch/vllm):
- **utils.py**: 97% (fast tests cover utilities comprehensively)
- **annotator.py**: 70-80% estimated (slow tests exercise inference pipeline)
- **Overall**: 75-85% estimated total coverage

## Test Strategy

### Fast Tests
Focus on:
- ✅ Pure Python logic (no external dependencies)
- ✅ Input validation and error handling
- ✅ Data transformations (prompts, messages, schemas)
- ✅ Configuration and initialization
- ✅ File I/O operations
- ✅ Utility functions

Use mocking for:
- Model inference (torch/vllm)
- GPU operations
- Heavy compute operations

### Slow Tests
Focus on:
- ✅ Actual model loading and inference
- ✅ End-to-end annotation pipeline
- ✅ JSON schema guided decoding validation
- ✅ Batch processing and memory management
- ✅ Resume/checkpoint functionality
- ✅ Hub integration (upload/download)

Use real resources:
- Small, fast model (Qwen2.5-0.5B - ~500MB)
- Minimal test data (2-10 samples)
- Short generation (max_tokens=10-20)

## CI/CD Integration

### Recommended CI Pipeline

```yaml
# .github/workflows/test.yml
jobs:
  fast-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: pip install pytest pytest-cov datasets
      - name: Run fast tests
        run: pytest -m "not slow" --cov=llm_annotator

  slow-tests:
    runs-on: ubuntu-latest-gpu  # GPU runner
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run slow tests
        run: pytest -m slow --cov=llm_annotator
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

### Best Practices

1. **Pull Requests**: Run fast tests on every commit
2. **Merges to Main**: Run all tests including slow ones
3. **Nightly Builds**: Run all tests including integration tests
4. **Local Development**: Use `pytest -m "not slow"` for rapid iteration

## Test Data Philosophy

Tests use minimal data to ensure:
- ✅ Fast execution
- ✅ Predictable results
- ✅ Easy debugging
- ✅ Low resource usage

Example test data:
- 2-10 samples per test
- Short texts (< 100 characters)
- Deterministic outputs (temperature=0)
- Small model (0.5B parameters)
- Limited tokens (max_tokens=10-20)

## Contributing

When adding new features to llm-annotator:

1. **Always add fast tests first**
   - Test initialization, configuration, validation
   - Mock external dependencies
   - Aim for 90%+ coverage of pure Python logic

2. **Add slow tests for model-dependent features**
   - Test actual inference behavior
   - Verify end-to-end pipelines
   - Use small model and minimal data

3. **Update this README**
   - Document new test classes
   - Update coverage expectations
   - Add examples if needed

4. **Maintain the fast/slow distinction**
   - Fast tests should run in < 1 second total
   - Slow tests should use `@pytest.mark.slow`
   - Integration tests should use `@pytest.mark.integration`

## Troubleshooting

### ImportError: No module named 'torch'
- This is expected for fast tests
- Fast tests mock torch/vllm to avoid the dependency
- Install torch only if running slow tests

### HF Token Required
- Integration tests need: `huggingface-cli login` or `HF_TOKEN` env var
- Fast and slow tests work without authentication
- Use `-m "not integration"` to skip these tests

### Out of Memory
- Slow tests use small model (0.5B params)
- Reduce batch size in tests if needed
- Use CPU if GPU memory is limited: `CUDA_VISIBLE_DEVICES="" pytest -m slow`

### Tests Fail on Different Machines
- Model outputs may vary slightly between systems
- Tests use `temperature=0` for determinism
- Some tests may need adjustment for CPU vs GPU

## Performance Benchmarks

Typical execution times on standard hardware:

- **Fast tests (31 utils + 17 annotator)**: < 1 second total
- **Slow tests setup (model loading)**: 30-60 seconds (one-time per session)
- **Individual slow test**: 2-10 seconds each
- **All tests (fast + slow)**: 2-3 minutes total
- **Integration tests**: Additional 1-2 minutes (Hub I/O)

---

## Quick Reference

```bash
# Development workflow
pytest -m "not slow"                  # Fast iteration

# Pre-commit
pytest -m "not slow" --cov           # Check coverage

# Pre-PR
pytest -m "not integration"           # All local tests

# Full validation
pytest --cov=llm_annotator            # Everything
```
