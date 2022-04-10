#!/bin/bash
#
# Prepare pyenv environment for mckit development on Linux.
#
# dvp, Apr 2022
#

default_python_version=3.10.4

usage() {
  cat <<- EndOfMessage

Reset PyEnv environment.

Usage
-----

reset_pyenv <env> <python_version>

All the parameters are optional.

Parameters (and defaults)
-------------------------
  env               - pyenv environment name (mckit)
  python_version    - ($default_python_version)

EndOfMessage

}

get_args() {

  [ "$1" == "--help" ] &&  usage && exit 0

  mckit="${1:-mckit}"
  shift


  python_version=${1:-$default_python_version}
  shift

  echo "Installing PyEnv environment $mckit with python $python_version"
}

reset_env() {
  unset LD_PRELOAD
  pyenv local "$python_version"
  pyenv virtualenv-delete -f "$mckit" > /dev/nul
  pyenv virtualenv "$python_version" "$mckit"
  pyenv local "$mckit" "3.9.12" "3.8.12"

  # pip is obsolete almost always
  python -m pip install --upgrade pip

#  source ./setenv.rc  reset

  poetry config --local virtualenvs.create false
  poetry install  && \
  pyenv rehash && \
  mckit --version || echo "ERROR: failed to install mckit" && return 1

  echo
  echo "SUCCESS: mckit has been installed"
  echo
}

check_environment() {
  poetry run pytest -m "not slow" &&  echo "SUCCESS: pytest is passed OK"

  poetry run nox --list
  poetry run nox -s safety
  poetry run nox -s tests -p "$python_version" -- -m "not slow" --cov

  tools/create-jk.sh "$mckit" || echo "ERROR: failed to create Jupyter Kernel for $mckit environment" &&  return 1

  echo
  echo SUCCESS!
  echo --------
  echo
  echo Usage:
  echo
  mckit --help
  echo
  echo "PyEnv environment $mckit is all clear."
  echo "Set your IDE to use $(pyenv which python)"
  echo
  echo Enjoy!
  echo
}

main() {
  echo "Running: $0 $*"
  get_args "$@" && reset_env && check_environment
}

main "$@"
