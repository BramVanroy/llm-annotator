# Tests - how to run

This document explains the test suite, the pytest markers we use, the expected developer workflow for running tests locally, and how integration tests are gated by Hugging Face authentication.

## Quick commands

- Run the fast, default test suite (unit / small tests only):

  ```bash
  python -m pytest -q
  ```
