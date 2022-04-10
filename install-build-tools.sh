#!/bin/bash

#
#  Install prerequisites for python, pyenv, python itself, poetry.
#
#  dvp Apr 2022
#
#  Be patient: this script has been changed since last usage and not tested after that.
#

OS="$(uname)"

install_linux_prerequisites() {
    sudo apt update && sudo apt install -y make build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
        libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
}

install_poetry() {
    if [[ ! -e poetry ]]; then
        #
        # This installs the latest poetry for current user global scope.
        # See https://python-poetry.org/docs/#installation#
        #
        curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
        ln -s "$HOME/.poetry/bin/poetry" "$HOME/bin/poetry"
    fi
}

install_linux_pyenv() {
    export PATH="$HOME/.pyenv/bin:$PATH"
    if [[ ! -e pyenv ]]; then
        #
        # install pyenv
        #
        curl https://pyenv.run | bash
        export PATH="$HOME/.pyenv/bin:$PATH"
        # also install tools
        git clone https://github.com/pyenv/pyenv-virtualenv.git "$(pyenv root)/plugins/pyenv-virtualenv"
        git clone git://github.com/pyenv/pyenv-doctor.git "$(pyenv root)/plugins/pyenv-doctor"
        # Add the following lines to your ~/.bashrc or ~/.zshrc:
        eval "$(pyenv init -)"
        eval "$(pyenv virtualenv-init -)"
        #
        # In Zsh you can use pyenv-zsh plugin.
        #
    fi
}

install_python() {
    local version="${1:-3.10.4}"
    pyenv install "$version"
    python="python$version"
    $python -m pip install --upgrade pip setuptools wheel
    pyenv local "$version"
}

# pre-commit
# see https://cjolowicz.github.io/posts/hypermodern-python-03-linting/#managing-git-hooks-with-precommit
#
# We install it with poetry as important development dependency.
#
#install_precommit() {
#    python -m pip install --upgrade pre-commit
#}
#setup_precommit() {
#    pre-commit install
#    pre-commit run --all-files
#}

install_all() {
    [[ "$OS" == "Linux" ]] || echo "ERROR: Install build tools is not implemented for $OS" && return 1

    install_linux_prerquisites
    #
    #    ATTENTION!
    #    =========
    #
    #    Make sure that pyenv is before poetry in PATH.
    #
    export PATH="$HOME/.pyenv/bin:$HOME/.poetry/bin:$PATH"
    install_linux_pyenv && \
    install_python  "$@" && \
    install_poetry
}

install_all "$@"

# vim: set ts=4 sw=0: tw=79 ss=0 ft=sh et ai :
