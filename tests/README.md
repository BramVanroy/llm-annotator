# LLM Annotator Test Suite

This directory contains comprehensive tests for the `llm-annotator` package. Tests are organized into **fast tests** and **slow tests** to optimize development workflow.

## Test Organization

### Fast Tests (No LLM Required)
These tests run quickly and don't require loading a language model:

- **`test_utils.py`**: Unit tests for utility functions
  - Hash generation
  - Number formatting
  - Retry decorator
  - JSONL file handling
  - Line counting
  - Type validation functions

- **`test_annotator_fast.py`**: Unit tests for Annotator class
  - Initialization and configuration
  - Context manager behavior
  - Message creation
  - Output processing (mocked)
  - Dataset validation
  - File naming and path handling

### Slow Tests (Requires LLM)
These tests require loading and running the `Qwen/Qwen2.5-0.5B-Instruct` model:

- **`test_annotator_slow.py`**: Integration tests with actual model inference
  - Basic dataset annotation
  - JSON schema / guided decoding
  - Column selection
  - Annotation resumption
  - Custom validation functions
  - Custom postprocessing functions
  - Task prefixes
  - Multiple batches
  - Sorting by length
  - Hugging Face Hub integration (marked with `@pytest.mark.integration`)

## Running Tests

### Install Test Dependencies
```bash
pip install pytest pytest-cov datasets
```

### Run All Tests
```bash
pytest
```

### Run Only Fast Tests
```bash
pytest -m "not slow"
```

### Run Only Slow Tests
```bash
pytest -m slow
```

### Run with Coverage Report
```bash
pytest --cov=llm_annotator --cov-report=term-missing --cov-report=html
```

### Run Specific Test File
```bash
pytest tests/test_utils.py
pytest tests/test_annotator_fast.py
pytest tests/test_annotator_slow.py
```

### Run Specific Test Class or Function
```bash
pytest tests/test_utils.py::TestGetHash
pytest tests/test_annotator_fast.py::TestAnnotatorInit::test_init_basic
```

## Test Markers

- `@pytest.mark.slow`: Tests that require model loading (slower execution)
- `@pytest.mark.integration`: Tests that interact with external systems (Hugging Face Hub)

## Fixtures

The test suite uses several fixtures defined in `conftest.py`:

- `test_model_id`: Returns "Qwen/Qwen2.5-0.5B-Instruct" for slow tests
- `test_dataset_name`: Returns "stanfordnlp/imdb" for testing
- `small_test_dataset`: A small in-memory dataset for quick testing
- `temp_dir`: Temporary directory for test outputs (cleaned up after session)
- `test_remote_dataset_name`: Hub dataset name for integration tests (requires auth)

## Integration Tests

Integration tests (marked with `@pytest.mark.integration`) require:
1. Hugging Face authentication: `huggingface-cli login` or set `HF_TOKEN` environment variable
2. Network access to Hugging Face Hub
3. Write permissions to create/delete test datasets

These tests automatically clean up created datasets after completion.

## Coverage Goals

The test suite aims for maximal coverage of:

### Utils Module
- ✅ Hash generation and consistency
- ✅ Number formatting (K, M, B)
- ✅ Retry decorator with success/failure scenarios
- ✅ JSONL reading with error handling
- ✅ JSONL reading with deduplication
- ✅ Column filtering
- ✅ Line counting
- ✅ Empty file removal
- ✅ Type validation (bool/dict return checks)

### Annotator Class
#### Fast Tests
- ✅ Initialization with various parameters
- ✅ Context manager usage
- ✅ Resume logic (skip already processed samples)
- ✅ Message/prompt creation
- ✅ Dataset preprocessing/postprocessing hooks
- ✅ Output processing with/without schemas
- ✅ File naming strategies
- ✅ Input validation for all major parameters
- ✅ Model cleanup

#### Slow Tests
- ✅ End-to-end annotation pipeline
- ✅ JSON schema / guided decoding
- ✅ Column selection and filtering
- ✅ Annotation resumption from partial results
- ✅ Custom validation functions
- ✅ Custom postprocessing functions
- ✅ Task prefixes
- ✅ Batch processing
- ✅ Dataset sorting by length
- ✅ Hub dataset loading
- ✅ Hub dataset uploading
- ✅ Periodic Hub uploads

## CI/CD Considerations

For continuous integration:

```bash
# Run fast tests only (recommended for every commit)
pytest -m "not slow" --cov=llm_annotator

# Run all tests including slow ones (recommended for PRs/releases)
pytest --cov=llm_annotator

# Skip integration tests that require Hub auth
pytest -m "not integration"
```

## Test Data

Tests use minimal data to ensure fast execution:
- Small in-memory datasets (2-10 samples)
- Temporary directories (auto-cleaned)
- Small model: Qwen/Qwen2.5-0.5B-Instruct (~500MB)
- Short prompts and responses (max_tokens=10-20)

## Contributing

When adding new features:
1. Add fast unit tests for new utility functions
2. Add fast unit tests for new Annotator methods (mock LLM if needed)
3. Add slow integration tests only if testing actual LLM behavior
4. Update this README with new test coverage
