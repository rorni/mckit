#!/usr/bin/env bash

OS="$(uname)"

function install_linux_prerequisites() {
    sudo apt update && sudo apt install -y make build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
        libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
}

function install_poetry() {
    if [[ ! -e poetry ]]; then
        #
        # This installs the latest poetry for current user global scope.
        # See https://python-poetry.org/docs/#installation#
        #
        curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
        ln -s $HOME/.poetry/bin/poetry $HOME/bin/poetry
    fi
}

function instal_linux_pyenv() {
    export PATH="~/.pyenv/bin:$PATH"
    if [[ ! -e pyenv ]]; then
        #
        # install pyenv
        #
        curl https://pyenv.run | bash
        export PATH="$HOME/.pyenv/bin:$PATH"
        # also install tools
        git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
        git clone git://github.com/pyenv/pyenv-doctor.git $(pyenv root)/plugins/pyenv-doctor
        # Add the following lines to your ~/.bashrc or ~/.zshrc:
        eval "$(pyenv init -)"
        eval "$(pyenv virtualenv-init -)"
        #
        # In Zsh use pyenv-zsh plugin.
        
    fi
}

function install_python() {
    local version="${1:-3.8.5}"
    pyenv install $version
    python = "python$version"
    $python -m pip install --upgrade pip setuptools wheel
    pyenv local $version
}

# pre-commit
# see https://cjolowicz.github.io/posts/hypermodern-python-03-linting/#managing-git-hooks-with-precommit
#
function install_precommit() {
    pythom -m pip install --upgrade pre-commit
}

function install_pyenv_win() {
    git clone https://github.com/pyenv-win/pyenv-win.git "$HOME/.pyenv"
}

function setup_precommit() {
    pre-commit install
    pre-commit run -all-files
}

function install_all() {
    if [[ "$OS" == "Linux" ]]; then
        install_linux_prerquisites
        #
        #    ATTENTION!
        #    =========
        #
        #    Make sure that pyenv is before poetry in PATH.
        #
        export PATH="$HOME/.pyenv/bin:$HOME/.poetry/bin:$PATH"
        install_linux_pyenv && \
        install_python      && \
        install_poetry
    elif [[ "$OS" == "MINGW64_NT-6.3-9600" ]]; then
        echo "Use power-shell script install-pyenv-win.ps1"
        return
        # export PATH="$HOME/.pyenv/bin:$HOME/.poetry/bin:$PATH"
        # install_pyenv_win   && \
        # install_python      && \
        # install_poetry
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
}


# vim: set ts=4 sw=0: tw=79 ss=0 ft=sh et ai :
