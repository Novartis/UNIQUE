# Declare env variables and executables
CONDA_ENV := ${PWD}/.conda/unique
PYTHON = ${CONDA_ENV}/bin/python

# Docs
SPHINXOPTS ?= "--jobs=auto"
SPHINXBUILD ?= sphinx-build
SOURCEDIR = ${PWD}/docs/source
BUILDDIR = ${PWD}/docs/build
DOCS_SOURCES := $(wildcard ${SOURCEDIR}/*.* ${SOURCEDIR}/**/*.*)
PY_SOURCES := $(wildcard ${PWD}/unique/*.py ${PWD}/unique/**/*.py ${PWD}/unique/**/**/*.py)
IPYNB_SOURCES := $(wildcard ${PWD}/notebooks/**/*.ipynb ${PWD}/notebooks/**/*.yaml ${PWD}/notebooks/**/*.py)

# Build docs only if there has been changes in the source code or source docs
docs: ${DOCS_SOURCES} ${PY_SOURCES} ${IPYNB_SOURCES}
	ln -sfn ${PWD}/notebooks ${SOURCEDIR}/examples/notebooks
	${SPHINXBUILD} -M clean "${SOURCEDIR}" "${BUILDDIR}"
	${SPHINXBUILD} -M html "${SOURCEDIR}" "${BUILDDIR}" ${SPHINXOPTS} ${0}

# Retrieve package information from pyproject.toml
info: print-info

print-info:
	@echo \### Description: `${PYTHON} -c 'import tomli; print(tomli.load(open("pyproject.toml", "rb"))["project"]["description"])'`
	@echo \### Version: `${PYTHON} -c 'import unique; print(unique.__version__)'`
	@echo \### License: `${PYTHON} -c 'import tomli; print(tomli.load(open("pyproject.toml", "rb"))["project"]["license"]["file"])'`
	@echo \### Main Author: `${PYTHON} -c 'import tomli; author=tomli.load(open("pyproject.toml", "rb"))["project"]["authors"][0]; name=author["name"]; email=author["email"]; print(f"{name} | {email}")'`
	@echo \### Home: `${PYTHON} -c 'import tomli; print(tomli.load(open("pyproject.toml", "rb"))["project"]["urls"]["Source"])'`

# Install environment
env:
	conda env create -f unique-environment.yml -p ${CONDA_ENV}

jupyter-kernel:
	${PYTHON} -m ipykernel install --sys-prefix --name unique --display-name unique

# Delete environment
clean-env:
	conda env remove -p ${CONDA_ENV}
	conda clean --all --yes
	rm -rf ${CONDA_ENV}

# Setup pre-commit
pre-commit:
	pre-commit install
	pre-commit run --all-files

# Tests
tests:
	pytest --cov=speakit --cov-report=xml:./.github/coverage.xml
	genbadge coverage -i ./.github/coverage.xml -o ./.github/coverage-badge.svg

# Targets without file dependency
.PHONY: clean-env env info jupyter-kernel pre-commit print-info tests
