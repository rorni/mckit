#!/usr/bin/env bash

OS="$(uname)"

if [[ "$OS" == "Linux" ]]; then

    #
    # install prerequisites
    #
    sudo apt update && sudo apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

    #
    # upgrade pip
    #
    python -m pip install --upgrade pip

    if [[ ! -e poetry ]]; then
        #
        # This installs the latest poetry for current user global scope.
        # See https://python-poetry.org/docs/#installation#
        #
        curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
        #
        #    ATTENTION!
        #    =========
        #
        #    Make sure that pyenv is before poetry in PATH.
        #
        ln -s $HOME/.poetry/bin/poetry $HOME/bin/poetry
    fi

    if [[ ! -e pyenv ]]; then
        #
        # install pyenv
        #
        curl https://pyenv.run | bash
        # Add the following lines to your ~/.bashrc or ~/.zshrc:
        #   export PATH="~/.pyenv/bin:$PATH"
        #   eval "$(pyenv init -)"
        #   eval "$(pyenv virtualenv-init -)"
        #
        # In Zsh use pyenv-zsh plugin.
    fi

    #
    # pre-commit
    # see https://cjolowicz.github.io/posts/hypermodern-python-03-linting/#managing-git-hooks-with-precommit
    #
    pip install --upgrade pre-commit
    pre-commit install
    pre-commit run -all-files

else
    echo "ERROR: Install build tools is not implemented for $OS"
    echo "       See the comment in ${0}"
    # There's Windows version for pyenv: https://pypi.org/project/pyenv-win/1.1.2/"
    # To install poetry on Windos use:
    #  (Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
    # Then setup PATH: remember pyenv should appear in the PATH before poetry.
    # In that case poetry will use the local virtual environment created by pyenv.
    #
fi

# vim: set ts=4 sw=0: tw=79 ss=0 ft=sh et ai :
