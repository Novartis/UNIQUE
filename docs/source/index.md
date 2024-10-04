---
myst:
   substitutions:
      description: "<u><b>UN</b></u>certa<u><b>I</b></u>nty <u><b>QU</b></u>antification b<u><b>E</b></u>nchmark: a Python library for benchmarking uncertainty estimation and quantification methods for Machine Learning models predictions."
      logo_light: |
         :::{figure} _static/unique_logo_blue.png
         :alt: UNIQUE Logo (light-theme)
         :class: only-light
         :align: center

         {{description}}
         :::
      logo_dark: |
         :::{figure} _static/unique_logo_dark_blue.png
         :alt: UNIQUE Logo (dark-theme)
         :class: only-dark
         :align: center

         {{description}}
         :::
      high_level_schema: |
         :::{figure} _static/schema_high_level.png
         :target: _images/schema_high_level.png
         :alt: UNIQUE High-Level Schema
         :align: center
         :class: dark-light

         High-level schema of `UNIQUE`'s components.
         :::
---

# Welcome to `UNIQUE`'s documentation!

{{python_versions_badge}} {{pypi_version_badge}} {{conda_version_badge}} {{license_badge}} {{chemrxiv_badge}} {{pypi_downloads_badge}} {{conda_downloads_badge}} {{docs_badge}} {{pypi_build_badge}}

{{logo_light}} {{logo_dark}}

## Introduction

`UNIQUE` provides methods for quantifying and evaluating the uncertainty of Machine Learning (ML) models predictions. The library allows to:
* combine and benchmark multiple uncertainty quantification (UQ) methods simultaneously;
* evaluate the goodness of UQ methods against established metrics;
* generate intuiti ve visualizations to qualitatively assess how well the computed UQ methods represent the actual model uncertainty;
* enable the users to get a comprehensive overview of their ML model's performances from an uncertainty quantification perspective.

`UNIQUE` is a model-agnostic tool, meaning that it does not depend on any specific ML model building platform nor provides any ML model training functionality. It only requires the user to input their model's inputs and predictions.

{{high_level_schema}}


Check out [Installation](./installation.md) to get started!

## Cite Us

{{chemrxiv_badge}}

If you find `UNIQUE` helpful for your work and/or research, please consider citing our work:

```bibtex
@misc{lanini2024unique,
  title={UNIQUE: A Framework for Uncertainty Quantification Benchmarking},
  author={Lanini, Jessica and Huynh, Minh Tam Davide and Scebba, Gaetano and Schneider, Nadine and Rodr{\'\i}guez-P{\'e}rez, Raquel},
  year={2024},
  doi={https://doi.org/10.26434/chemrxiv-2024-fmbgk},
}
```

To request more information, check out [Contacts & Acknowledgements](development/contacts.md#contacts--acknowledgements).


---

# Table of Contents

:::{toctree}
:maxdepth: 2
installation
getting_started/index
examples/index
indepth/index
development/index
:::

:::{image} ./_static/novartis_logo.png
:alt: Novartis logo
:align: center
:class: dark-light
:::
