# Setup Guide

This document contains the instructions for setting up the development environment.

## Prerequisites

- Python (version specified in environment.yml)
   - Specific version is `3.11.14`
   - `environment.yml` says `3.11`, because `.14` caused CI issues
   - This does not affect the code
- Conda (recommended)

## Installation

1. Clone the repository

```bash
git clone https://github.com/Sebastijan-Dominis/hotel_management
cd hotel_management
```

2. Create and activate the Conda environment:

```bash
conda env create -f environment.yml -n hotel_management # default name
conda activate hotel_management # if you used the default name
```

3. Install the rest of the packages:

```bash
pip install -r requirements.txt # for packages that can't be installed with conda-forge
```

4. Set up environment variables

In `{repo_root}/.env`.

### Descriptions

`ML_SERVICE_BACKEND_URL` = URL in which you want to run the `FastAPI` backend from `ml_service`
`ML_SERVICE_FRONTEND_URLS` = URLS you want to allow access in the backend

Notes:
- keep default frontend urls unless you change them in the frontend as well, or want to add more
- frontend ports currently explicitly defined in the frontend code -> change if security becomes an issue

### Defaults

`ML_SERVICE_BACKEND_URL`= http://127.0.0.1:8000
`ML_SERVICE_FRONTEND_URLS`= [http://localhost:8050, http://localhost:8051, http://localhost:8052, http://localhost:8053, http://localhost:8054, http://localhost:8055]


## Additional Notes

- Some of the packages had to be moved from environment.yml to requirements.txt, because the CI was failing
   - Only frontend/backend packages; should not impact anything
- The inclusion of specific package versions ensures consistency
- Author tested the code on Windows, while CI ensures UNIX compatibility (uses Ubuntu, includes tests)
- CI currently does not include testing on Windows and MacOS, due to higher costs of those services

## Expectations

- `environment.yml` already includes `pytest` and `pytest-cov` packages for testing and coverage generation
- `environment.yml` already includes `ruff`, `isort`, `mypy` and `pre-commit` packages for quality checks
- `environment.yml` already includes `pdoc` package for API doc generation
- GPU hyperparameter searching and training will only work on your machine as expected if you use Nvidia GPU(s); otherwise please use CPU
