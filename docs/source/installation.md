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

{{python_versions}}

`UNIQUE` is currently compatible with Python 3.8 through 3.12.1. To install the latest release and use the package as is, run the following in a compatible environment of choice:

```bash
pip install git+https://github.com/Novartis/UNIQUE.git
```

:::{tip}
To create a dedicated virtual environment for `UNIQUE` using `conda`/`mamba` with all the required and compatible dependencies, check out: [For Developers](#for-developers).
:::

## For Developers

 {{precommit}} {{codestyle}}

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

The project uses [`mamba`](https://mamba.readthedocs.io/en/latest/index.html) for dependencies management, which is a faster drop-in replacement for [`conda`](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

:::{tip}
If you still wish to use `conda`, you can change the backend solver by adding `--solver=libmamba` to your `conda install` standard command ([check out the docs](https://conda.github.io/conda-libmamba-solver/user-guide/#try-it-once)).
:::

To set the project up, run:

```bash
# Install conda environment and jupyter kernel locally
make env && make jupyter-kernel
conda activate .conda/unique

# Setup precommit hooks
make pre-commit

# Install UNIQUE
pip install -e .
# Use `pip install -e .[dev]` to also install optional dependencies
```

In this way, you will have access to the `UNIQUE` codebase and be able to make local modifications to the source code, within the `.conda/unique` environment that contains all the required dependencies.

Additionally, if you use Jupyter Notebooks, the `unique` kernel will be available in the "Select kernel" menu of the JupyterLab/JupyterNotebook UI.

Finally, when using `git` for code versioning, the predefined `pre-commit` hooks will be run against the commited files for automatic formatting and syntax checks.

You can find out more about custom Jupyter kernels [here](https://ipython.readthedocs.io/en/stable/install/kernel_install.html) and `pre-commit` hooks [here](https://pre-commit.com/).