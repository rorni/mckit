#!/bin/bash

# See nbdime – diffing and merging of Jupyter Notebooks
# https://nbdime.readthedocs.io/en/latest/

conda install --yes nbdime && \
nbdime config-git --enable --global
