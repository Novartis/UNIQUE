# Declare env variables and executables
CONDA_ENV := ${PWD}/.conda/unique
PYTHON = ${CONDA_ENV}/bin/python

# Retrieve package information from pyproject.toml
info: print-info

print-info:
	@echo \### Description: `${PYTHON} -c 'import tomli; print(tomli.load(open("pyproject.toml", "rb"))["project"]["description"])'`
	@echo \### Version: `${PYTHON} -c 'import tomli; print(tomli.load(open("pyproject.toml", "rb"))["project"]["version"])'`
	@echo \### License: `${PYTHON} -c 'import tomli; print(tomli.load(open("pyproject.toml", "rb"))["project"]["license"]["file"])'`
	@echo \### Main Author: `${PYTHON} -c 'import tomli; author=tomli.load(open("pyproject.toml", "rb"))["project"]["authors"][0]; name=author["name"]; email=author["email"]; print(f"{name} | {email}")'`
	@echo \### Home: `${PYTHON} -c 'import tomli; print(tomli.load(open("pyproject.toml", "rb"))["project"]["urls"]["Source"])'`

# Install environment
env:
	mamba env create -f unique-environment.yml -p ${CONDA_ENV}
	mamba config --append envs_dirs ${CONDA_ENV}

jupyter-kernel:
	${PYTHON} -m ipykernel install --sys-prefix --name unique --display-name unique

# Delete environment
clean-env:
	mamba env remove -p ${CONDA_ENV}
	mamba clean --all --yes
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
.PHONY: info print-info env clean-env jupyter-kernel pre-commit tests
