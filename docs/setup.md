# Setup Guide

This document contains the instructions for setting up the development environment.

## Prerequisites

- Python (version specified in environment.yml)
   - Specific version is `3.11.14`
   - `environment.yml` says `3.11`, because `.14` caused CI issues
   - This does not affect the code
- Conda (recommended)

**OR**

- Docker

## Installation

### No Docker

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
[Click here](#environment-variables)

### Docker

1. Pull the required image

```bash
docker pull pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime
```

2. Build the image

```bash
docker compose build --no-cache
```

3. Set up environment variables

In `{repo_root}/.env`.
[Click here](#environment-variables)

### Environment Variables

`ML_SERVICE_BACKEND_URL` = URL in which you want to run the `FastAPI` backend from `ml_service`
`ML_SERVICE_FRONTEND_URL` = URL in which you want to run the `Dash` frontend from `ml_service`

Notes:
- keep default frontend urls unless you change them in the frontend as well, or want to add more
- frontend port currently explicitly defined in the frontend code -> change if security becomes a concern

### Defaults

`ML_SERVICE_BACKEND_URL`=http://localhost:8000
`ML_SERVICE_FRONTEND_URL`=http://localhost:8050

## Fake data generation

Fake data can be generated using a script found in `scripts/generators/generate_fake_data.py`.
If you want to do so, make sure to either:
1. uncomment and install the two commented-out packages in `requirements.txt`
2. uncomment `sdv` in `requirements.txt` and the relevant parts of the `Dockerfile` (if using `Docker`)

> Note: the code in that script is not modularized, as this repo does not focus on fake data generation,
> and instead uses fake data generation only as a utility for quick and easy simulation and testing.

- The repo should include some fake data at any given point, along with a trained fake data generation model.
- Only use this feature if necessary - the packages (sdv + torch) can act oddly depending on hardware, and
tend to take a while to install.

## Post-installation

You can now operate the ml workflow in following ways
- using `ml_service` in browser
   - backend on localhost:8000 (default)
   - frontend on localhost:8050
- manually
   - cli for pipelines
   - manual writing of configs

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
