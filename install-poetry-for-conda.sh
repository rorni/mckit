#!/bin/bash
#
# See https://github.com/python-poetry/poetry/issues/105
# https://github.com/biotite-dev/biotite/issues/214
#
# Configure the Poetry's Virtualenv location.
# Use `conda info` and https://stedolan.github.io/jq/download/
CONDA_ENV_PATH=$(conda info --json | jq '.envs_dirs[0]')  # set the variable explicitly in Windows
poetry config settings.virtualenvs.path $CONDA_ENV_PATH
poetry config settings.virtualenvs.create 0
