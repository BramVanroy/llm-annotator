# Tests - how to run

This document explains the test suite, the pytest markers we use, the expected developer workflow for running tests locally, and how integration tests are gated by Hugging Face authentication.

## Quick commands

- Run the fast, default test suite (unit / small tests only):

  ```bash
  python -m pytest -q
  ```

- Run only slow tests:

  ```bash
  python -m pytest -q -m slow
  ```

- Run only integration tests (network, Hub uploads, real-model runs):

  ```bash
  # You must be authenticated with the Hugging Face CLI first (see below)
  python -m pytest -q -m integration
  ```

- Run integration + slow tests together:

  ```bash
  python -m pytest -q -m "integration and slow"
  ```

## Markers we use

- `integration`
  - Meaning: Tests that interact with external systems: the Hugging Face Hub (uploads, downloads, repo creation/deletion), or that start real model engines (vLLM or other LLM runtimes).
  - Behavior: These tests are skipped by default unless the test runner can detect a logged-in Hugging Face account (see Authentication). They may perform remote uploads and deletions - tests attempt to clean up after themselves but use a per-user test repo to avoid collisions.

- `slow`
  - Meaning: Tests that take longer because of heavier I/O, retries, or computational work. Some `slow` tests are also `integration` tests; the markers are orthogonal so you can run them independently.

Notes on combining markers: use pytest's `-m` expression language, for example `-m "integration and slow"` or `-m "integration or slow"`.

## Authentication (how integration tests are gated)

Integration tests are not gated by a custom CLI flag. Instead the test collection code checks whether a Hugging Face user is available via the Hub API (the repository's `tests/conftest.py` uses `HfApi().whoami()` under the hood).

How to make integration tests run locally:

1. Install the Hugging Face CLI (`hf`) and authenticate:

   ```bash
   hf auth login
   ```

   The `hf auth login` flow stores your credentials locally and makes them available to `HfApi().whoami()`. If you are not logged in, tests marked `integration` (and `slow` tests that require remote access) will be skipped automatically during collection.

2. Run integration tests:

   ```bash
   python -m pytest -q -m integration
   ```

CI note: In CI you can either log in using the `hf` CLI (for manual runs) or provide a `HUGGINGFACE_HUB_TOKEN` / `HF_TOKEN` secret. The test gating checks `HfApi().whoami()` so either mechanism that makes the account available will work.

## Fixtures relevant to integration tests

- `hf_username` (session scoped): resolves the current Hugging Face account name and is used by tests that construct per-user test repo ids.
- `test_remote_dataset_name`: returns a per-user dataset id of the form `{hf_username}/llm_annotator_test_ds` or skips when no username is available.

These fixtures ensure uploads and cleanup target the test user's account instead of a hard-coded owner.

## Uploads and cleanup expectations

- Integration tests that upload to the Hub attempt to delete their test repos during a session-scoped cleanup. Ensure your account has sufficient permissions to delete the test repo if you want automatic cleanup to succeed.
- Tests use a per-user repo id (via `hf_username`) to avoid colliding with other developers' repos.

## Recommended developer workflow

1. Run fast unit tests frequently on every edit:

   ```bash
   python -m pytest -q
   ```

2. When you need to exercise Hub uploads or real-model behavior, log in and run integration tests locally:

   ```bash
   hf auth login
   python -m pytest -q -m integration
   ```

3. For long-running checks (profiling, full dataset runs, heavy GPU tests), run `-m slow` or combine markers as needed.

## Troubleshooting

- Integration tests skipped? Make sure `hf auth login` succeeded. You can verify by running a quick Python snippet:

  ```python
  from huggingface_hub import HfApi
  print(HfApi().whoami())
  ```

  This should return a dict with a `name` key when logged in as a user.

- Uploads failing due to permissions: ensure the logged-in account can create/delete dataset repos in your namespace.
