---
myst:
   substitutions:
      precommit: "[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)"
      codestyle: |
         :::{image} https://img.shields.io/badge/Code%20Style-black-000000.svg
         :alt: Codestyle
         :target: https://github.com/psf/black
         :::
---
# Installation

{{python_versions_badge}} {{pypi_version_badge}} {{conda_version_badge}} {{pypi_downloads_badge}} {{conda_downloads_badge}} {{pypi_build_badge}}

`UNIQUE` is currently compatible with Python 3.8 through 3.12.1. To install the latest release, run the following in a compatible environment of choice:

::::{tab-set}

:::{tab-item} `pip`

```bash
pip install unique-uncertainty
```
:::

:::{tab-item} `conda`

```bash
conda install -c conda-forge unique-uncertainty
```
:::

:::{tab-item} `mamba`

```bash
mamba install -c conda-forge unique-uncertainty
```
:::

::::

:::{tip}
To create a dedicated virtual environment for `UNIQUE` using `conda`/`mamba` with all the required and compatible dependencies, check out: [For Developers](#for-developers).
:::

## For Developers

 {{license_badge}} {{precommit}} {{codestyle}}

:::{seealso}
If you wish to work on the codebase itself, check first [how to best contribute to `UNIQUE`](./development/contributing.md).
:::

:::{warning}
The following steps are recommended only for expert/power-users.
:::

First, clone the repository and check into the project folder.

```bash
git clone https://github.com/Novartis/UNIQUE.git ./unique
cd unique
```

The project uses [`conda`](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and/or [`mamba`](https://mamba.readthedocs.io/en/latest/index.html) for dependencies management.

Install first the `conda` dependencies and the Jupyter kernel needed to run the examples:

```bash
# Install conda environment and jupyter kernel locally
make env && make jupyter-kernel
conda activate .conda/unique
```

Next, enable the pre-commit hooks for automatic code formatting/linting:

```bash
# Setup precommit hooks
make pre-commit
```

Lastly, install `UNIQUE` from source:

```bash
pip install -e .
# Use `pip install -e .[dev]` to also install optional dependencies
```

In this way, you will have access to the `UNIQUE` codebase and be able to make local modifications to the source code, within the `./.conda/unique` local environment that contains all the required dependencies.

Additionally, if you use Jupyter Notebooks, the `unique` kernel will be available in the "Select kernel" menu of the JupyterLab/JupyterNotebook UI.

Finally, when using `git` for code versioning, the predefined `pre-commit` hooks will be run against the commited files for automatic formatting and syntax checks.

You can find out more about custom Jupyter kernels [here](https://ipython.readthedocs.io/en/stable/install/kernel_install.html) and `pre-commit` hooks [here](https://pre-commit.com/).
