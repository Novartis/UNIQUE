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
      version_badge: "![Version badge](https://img.shields.io/badge/Version-vUNIQUE-green)"
      # See: https://github.com/executablebooks/MyST-Parser/issues/279#issuecomment-752948379
      license_badge: |
         :::{image} https://img.shields.io/badge/License-BSD_3--Clause-red.svg
         :alt: License
         :target: https://opensource.org/licenses/BSD-3-Clause
         :::
      high_level_schema: |
         :::{figure} _static/schema_high_level.png
         :alt: UNIQUE High-Level Schema
         :align: center
         :class: dark-light

         High-level schema of `UNIQUE`'s components.
         :::
---

# Welcome to `UNIQUE`'s documentation!

{{python_versions}} {{version_badge | replace("UNIQUE", version)}} {{license_badge}}

{{logo_light}} {{logo_dark}}

`UNIQUE` provides methods for quantifying and evaluating the uncertainty of Machine Learning (ML) models predictions. The library allows to:
* combine and benchmark multiple uncertainty quantification (UQ) methods simultaneously;
* evaluate the goodness of UQ methods against established metrics;
* generate intuitive visualizations to qualitatively assess how well the computed UQ methods represent the actual model uncertainty;
* enable the users to get a comprehensive overview of their ML model's performances from an uncertainty quantification perspective.

`UNIQUE` is a model-agnostic tool, meaning that it does not depend on any specific ML model building platform nor provides any ML model training functionality. It only requires the user to input their model's inputs and predictions.

{{high_level_schema}}

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

# API Reference

:::{toctree}
:maxdepth: 4
apidocs/index
:::

:::{image} ./_static/novartis_logo.png
:alt: Novartis logo
:align: center
:class: dark-light
:::
