# Contributing Guide

Thank you for considering contributing to this project.

- This repo is currently getting prepared to enable contributions.

## Development Setup
- Go to [setup.md](setup.md) for instructions.

## Code Style
- Use `pre-commit` for linting, formatting, and quality checks
    > run:
    ```bash
    pre-commit run --all-files
    ```
- Follow existing project conventions
- All new code must include tests
- CI requires at least **90% test coverage**

## Branching
Create feature branches preferably using the following format:

- `feature/<name>`
- `fix/<issue>`
- `docs/<topic>`

## Commit Messages
Use clear, descriptive commit messages.

Examples:
- feat: add monitoring pipeline
- fix: correct the promotion step
- docs: update the glossary

## Pull Requests
- Create a pull request from a feature branch
- Write clear commit messages
- Link related issues in the PR description
- Ensure all CI checks pass

## Review Process
- All code must be reviewed before merging
- Tests must pass before approval
- Address reviewer feedback before merge

## Running tests
- Please check [testing.md](testing.md) for detailed guidance