# Setup Guide

This document contains the instructions for setting up the development environment.

## Prerequisites

- Python (version specified in environment.yml)
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
   pip install -r requirements-pip.txt # for packages that can't be installed with conda-forge
   ```

## Expectations

- environment.yml already includes pytest and pytest-cov for testing and coverage generation
- environment.yml already includes ruff, isort, mypy and pre-commit for quality checks
- GPU hyperparameter searching and training will only work on your machine as expected if you use Nvidia GPU(s); otherwise please use CPU
