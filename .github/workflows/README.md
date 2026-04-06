# GitHub Workflows

This directory contains the repository's GitHub Actions workflows.

## Files

- `ci.yml`: runs the Python test suite on pushes to `main` and on pull requests, then verifies that both Docker image targets build successfully.

## Notes

- The workflow installs dependencies through `make install-hf`.
- The test job targets Python 3.11.
- The Docker job builds the `backend` and `frontend` targets from the root `Dockerfile`.
