#/usr/bin/env sh

python -m pip install --upgrade pip

# This installs the latest poetry for current user global scope.
# After installation issue the command:
#    source ~/.poetry/env
# to use it.
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

# pip install --user --upgrade pre-commit

# vim: set ts=4 sw=0 tw=79 ss=0 et ai :