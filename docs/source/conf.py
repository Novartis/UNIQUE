# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import unique

project = "UNIQUE"
copyright = "2024, Novartis Pharma AG"
author = "Novartis Pharma AG"
# author = 'Minh Tam Davide Huynh, Gaetano Scebba, Jessica Lanini, Raquel Rodriguez-Perez'

pypi_version_badge_url = (
    "https://img.shields.io/pypi/v/unique-uncertainty?color=yellowgreen&label=PyPI"
)

conda_version_badge_url = "https://img.shields.io/conda/vn/conda-forge/unique-uncertainty?color=yellowgreen&label=conda-forge"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # "autoapi.extension",
    # "autodoc2",  # Instead of 'sphinx.ext.autodoc',
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
    # "sphinx.ext.autodoc",
    # "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- MyST configuration ------------------------------------------------------

# myst_gfm_only = False
# myst_commonmark_only = False

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "linkify",
    "substitution",
]

myst_heading_anchors = 3

myst_substitutions = {
    "conda_downloads_badge": "[![Conda downloads](https://img.shields.io/conda/dn/conda-forge/unique-uncertainty?color=brightgreen&label=conda%20downloads)](https://anaconda.org/conda-forge/unique-uncertainty)",
    "conda_version_badge": "[![Conda version](https://img.shields.io/conda/vn/conda-forge/unique-uncertainty?color=yellowgreen&label=conda-forge)](https://anaconda.org/conda-forge/unique-uncertainty)",
    "pypi_build_badge": "[![PyPI build](https://github.com/Novartis/UNIQUE/actions/workflows/build.yml/badge.svg?branch=main)](https://pypi.org/project/unique-uncertainty)",
    "pypi_downloads_badge": "[![PyPI downloads](https://img.shields.io/pypi/dm/unique-uncertainty?color=brightgreen&label=PyPI%20downloads)](https://pypi.org/project/unique-uncertainty)",
    "pypi_version_badge": "[![PyPI version](https://img.shields.io/pypi/v/unique-uncertainty?color=yellowgreen&label=PyPI)](https://pypi.org/project/unique-uncertainty)",
    "python_versions_badge": "![Python versions](https://img.shields.io/pypi/pyversions/unique-uncertainty?label=Python)",
}

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

myst_admonition_enable = True
myst_amsmath_enable = True
myst_footnote_transition = False
myst_html_img_enable = True
myst_url_schemes = ("http", "https", "mailto")

# -- Myst NB Configuration ---------------------------------------------------
nb_execution_mode = "cache"  # "auto"
nb_execution_timeout = -1
nb_execution_cache_path = "docs/.jupyter_cache"
nb_merge_streams = True

# -- Sphinx AutoAPI ----------------------------------------------------------
# autoapi_dirs = ["../../unique"]
# autoapi_add_toctree_entry = False
# autoapi_generate_api_docs = False
# autoapi_add_objects_to_toctree = False

# -- Sphinx AutoDoc2 Config --------------------------------------------------
# autodoc2_module_all_regexes = [
#     "unique.error_models",
# 	"unique.evaluation",
# 	"unique.input_type",
# 	"unique.uncertainty",
# 	"unique.uq_metric_factory",
# 	"unique.utils",
# 	"unique",
# ]
# autodoc2_packages = [
#     {
#         "path": "../../unique",
#         "auto_mode": True,
#     },
# ]
# autodoc2_hidden_objects = ["dunder", "private", "inherited"]
# autodoc2_index_template = """API Reference
# =============

# These sections contain auto-generated API reference documentation.

# .. toctree::
#    :titlesonly:
# {% for package in top_level %}
#    {{ package }}
# {%- endfor %}

# """

# autodoc2_render_plugin = (
#     "md"  # Comment to allow for correct parsing of Google-style docstring
# )

# autodoc_typehints = "both"
# autodoc_typehints_description_target = "documented_params"
# autodoc_default_options = {
#     "members": True,
#     "member-order": "groupwise",
#     "private-members": True,
#     "special-members": "__init__",
#     "undoc-members": True,
# }

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_favicon = "_static/nvs_favicon.svg"
html_logo = "_static/unique_logo_blue.png"
html_static_path = ["_static"]
html_theme = "sphinx_book_theme"
html_title = "UNIQUE"
html_last_updated_fmt = ""

html_theme_options = {
    "icon_links": [
        {
            "name": "PyPI Version",
            "url": "https://pypi.org/project/unique-uncertainty",
            "icon": pypi_version_badge_url,
            "type": "url",
        },
        {
            "name": "Conda Version",
            "url": "https://anaconda.org/conda-forge/unique-uncertainty",
            "icon": conda_version_badge_url,
            "type": "url",
        },
    ],
    "logo": {
        "alt_text": "UNIQUE's Documentation - Home",
        "image_dark": "_static/unique_logo_dark_blue.png",
    },
    "repository_branch": "main",
    "repository_provider": "github",
    "repository_url": "https://github.com/Novartis/UNIQUE",
    "path_to_docs": "./docs/source",
    "show_navbar_depth": 1,
    "toc_title": "On this page",
    "use_edit_page_button": True,
    "use_repository_button": True,
    "use_source_button": True,
}
