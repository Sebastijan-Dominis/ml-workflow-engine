# API Reference

This directory contains the API documentation for the `hotel_management` project.  
It includes the main packages, classes, and functions that developers and data scientists can use.  

> Note: all detailed documentation is auto-generated from docstrings, using the `pdoc` package.

There are four folders, covering three of the main folders found in the repo - `ml/`, `pipelines/` and `scripts/`, along with `ml_service/`.

## Viewing

To view the documentation:

1. Ensure you have python installed and set up, e.g. a quick check:
```bash
python --version
```

2. Navigate to a folder of choice, e.g.:
```bash
cd docs/api/ml
```

3. Start the server on a port of your choice, e.g. `8000`:
```bash
python -m http.server 8000
```

4. Open your browser and visit http://localhost:8000

The documentation is static HTML, so you do not need `pdoc` installed to view it — only Python to run the local server.

## Generating

In order to generate documentation:

1. Install `pdoc`:
```bash
conda install -c conda-forge pdoc
```

*or*

```bash
pip install pdoc
```

> Note: `pdoc` package is also included in `environment.yml`.

2. Ensure `__init__.py` files exist
- Create them for each package, even if empty
- Adding a docstring for clarity is encouraged
- `pdoc` ignores packages without `__init__.py`

3. Generate the documentation:
- Run the following command from repo root.
```bash
pdoc -html {folder_of_choice} -o docs/api/{folder_of_choice}
```

- `folder_of_choice` = folder for which you are generating the documentation
- keep the naming and file structure consistent

**Best practice example**:
```bash
pdoc -html ml -o docs/api/ml
```

**Bad practice examples**:

```bash
pdoc -html ml -o docs/api/stuff
```

```bash
pdoc -html ml -o docs/ml
```

```bash
pdoc -html ml -o ml
```