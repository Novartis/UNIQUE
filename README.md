<figure>
  <img src="https://github.com/Novartis/UNIQUE/raw/main/docs/source/_static/unique_logo_blue.png" alt="UNIQUE Logo">
  <figcaption align=center><u><b>UN</b></u>certa<u><b>I</b></u>nty <u><b>QU</b></u>antification b<u><b>E</b></u>nchmark: a Python library for benchmarking uncertainty estimation and quantification methods for Machine Learning models predictions.</figcaption>
</figure>

[![Python](https://img.shields.io/pypi/pyversions/unique-uncertainty?label=Python)](https://pypi.org/project/unique-uncertainty/)
[![PyPI version](https://img.shields.io/pypi/v/unique-uncertainty?color=green&label=PyPI)](https://pypi.org/project/unique-uncertainty/)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/unique-uncertainty?color=green&label=conda-forge)](https://anaconda.org/conda-forge/unique-uncertainty)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-red)](https://opensource.org/licenses/BSD-3-Clause)
[![chemRxiv](https://img.shields.io/badge/chemRxiv-10.26434%2Fchemrxiv--2024--fmbgk-yellow)](https://doi.org/10.26434/chemrxiv-2024-fmbgk)
[![PyPI downloads](https://img.shields.io/pypi/dm/unique-uncertainty?color=yellowgreen&label=PyPI%20downloads)](https://pypi.org/project/unique-uncertainty/)
[![Conda downloads](https://img.shields.io/conda/dn/conda-forge/unique-uncertainty?color=yellowgreen&label=conda%20downloads)](https://anaconda.org/conda-forge/unique-uncertainty)
[![Docs](https://github.com/Novartis/UNIQUE/actions/workflows/docs.yml/badge.svg?branch=main)](https://opensource.nibr.com/UNIQUE/)
[![Build](https://github.com/Novartis/UNIQUE/actions/workflows/build.yml/badge.svg?branch=main)](https://pypi.org/project/unique-uncertainty/)


## Introduction

`UNIQUE` provides methods for quantifying and evaluating the uncertainty of Machine Learning (ML) models predictions. The library allows to combine and benchmark multiple uncertainty quantification (UQ) methods simultaneously, generates intuitive visualizations, evaluates the goodness of the UQ methods against established metrics, and in general enables the users to get a comprehensive overview of their ML model's performances from an uncertainty quantification perspective.

`UNIQUE` is a model-agnostic tool, meaning that it does not depend on any specific ML model building platform or provides any  ML model training functionality. It is lightweight, because it only requires the user to input their model's inputs and predictions.

<figure>
  <img src="https://github.com/Novartis/UNIQUE/raw/main/docs/source/_static/schema_high_level.png" alt="UNIQUE High Level Schema">
  <figcaption align=center>High-level schema of <code>UNIQUE</code>'s components.</figcaption>
</figure>


## Installation

[![Python](https://img.shields.io/pypi/pyversions/unique-uncertainty?label=Python)](https://pypi.org/project/unique-uncertainty/)
[![PyPI version](https://img.shields.io/pypi/v/unique-uncertainty?color=green&label=PyPI)](https://pypi.org/project/unique-uncertainty/)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/unique-uncertainty?color=green&label=conda-forge)](https://anaconda.org/conda-forge/unique-uncertainty)
[![PyPI downloads](https://img.shields.io/pypi/dm/unique-uncertainty?color=yellowgreen&label=PyPI%20downloads)](https://pypi.org/project/unique-uncertainty/)
[![Conda downloads](https://img.shields.io/conda/dn/conda-forge/unique-uncertainty?color=yellowgreen&label=conda%20downloads)](https://anaconda.org/conda-forge/unique-uncertainty)
[![Build](https://github.com/Novartis/UNIQUE/actions/workflows/build.yml/badge.svg?branch=main)](https://pypi.org/project/unique-uncertainty/)

`UNIQUE` is currently compatible with Python 3.8 through 3.12.1. To install the latest release and use the package as is, run the following in a compatible environment of choice:

```bash
pip install unique-uncertainty
```

or:

```bash
conda install -c conda-forge unique-uncertainty
# mamba install -c conda-forge unique-uncertainty
```

Check out the [docs](https://opensource.nibr.com/UNIQUE/installation.html#installation) for more installation instructions.


## Getting Started

Check out the [docs](https://opensource.nibr.com/UNIQUE/getting_started/index.html#getting-started) for a complete set of instructions on how to prepare your data and the possible configurations offered by `UNIQUE`.


## Usage

Finally, once the data and configuration files have been prepared, you can run `UNIQUE` in the following way:

```python
from unique import Pipeline

# Prepare UNIQUE pipeline
pipeline = Pipeline.from_config("/path/to/config.yaml")

# Run UNIQUE pipeline
uq_methods_outputs, uq_evaluation_outputs = pipeline.fit()
# Returns: (Dict[str, np.ndarray], Dict[str, pd.DataFrame])
```

Fitting the `Pipeline` will return two dictionaries:

- `uq_methods_outputs`: contains each UQ method's name (as in "UQ_Method_Name[Input_Name(s)]") and computed UQ values.
- `uq_evaluation_outputs`: contains, for each evaluation type (ranking-based, proper scoring rules, and calibration-based), the evaluation metrics outputs for all the corresponding UQ methods organized in `pd.DataFrame`.

Additionally, `UNIQUE` also generates graphical outputs in the form of tables and evaluation plots (if `display_outputs` is enabled and the code is running in a JupyterNotebook cell).


### Examples

For more hands-on examples and detailed usage, check out some of the examples in the [docs](https://opensource.nibr.com/UNIQUE/examples/index.html#examples).


## Deep Dive

Check out the [docs](https://opensource.nibr.com/UNIQUE/indepth/index.html#deep-dive) for an in-depth overview of `UNIQUE`'s concepts, functionalities, outputs, and references.


## Contributing

Any and all contributions and suggestions from the community are more than welcome and highly appreciated. If you wish to help us out in making `UNIQUE` even better, please check out our [contributing guidelines](./CONTRIBUTING.md).

Please note that we have a [Code of Conduct](./CODE_OF_CONDUCT.md) in place to ensure a positive and inclusive community environment. By participating in this project, you agree to abide by its terms.


## License

[![License](https://img.shields.io/badge/License-BSD_3--Clause-red)](https://opensource.org/licenses/BSD-3-Clause)

`UNIQUE` is licensed under the BSD 3-Clause License. See the [LICENSE](./LICENSE.md) file.


## Cite Us

[![chemRxiv](https://img.shields.io/badge/chemRxiv-10.26434%2Fchemrxiv--2024--fmbgk-yellow)](https://doi.org/10.26434/chemrxiv-2024-fmbgk)

If you find `UNIQUE` helpful for your work and/or research, please consider citing our work:

```bibtex
@misc{lanini2024unique,
  title={UNIQUE: A Framework for Uncertainty Quantification Benchmarking},
  author={Lanini, Jessica and Huynh, Minh Tam Davide and Scebba, Gaetano and Schneider, Nadine and Rodr{\'\i}guez-P{\'e}rez, Raquel},
  year={2024},
  doi={https://doi.org/10.26434/chemrxiv-2024-fmbgk},
}
```


## Contacts & Acknowledgements

For any questions or further details about the project, please get in touch with any of the following contacts:

* **[Jessica Lanini](mailto:jessica.lanini@novartis.com?subject=UNIQUE)**
* **[Minh Tam Davide Huynh](https://github.com/mtdhuynh)**
* **[Gaetano Scebba](mailto:gaetano.scebba@novartis.com?subject=UNIQUE)**
* **[Nadine Schneider](mailto:nadine-1.schneider@novartis.com?subject=UNIQUE)**
* **[Raquel Rodríguez-Pérez](mailto:raquel.rodriguez_perez@novartis.com?subject=UNIQUE)**


![Novartis Logo](https://github.com/Novartis/UNIQUE/raw/main/docs/source/_static/novartis_logo.png)
