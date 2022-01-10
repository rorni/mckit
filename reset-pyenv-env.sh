#!/usr/bin/env bash
#
# Prepare pyenv environment for mckit development on Linux.
#
# dvp, Dec 2020
#

default_python_version=3.9.9

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

  poetry "--version"
  if [[ ! $?  ]]; then
      echo "ERROR: Poetry is not available"
      return 1
  fi

  python_version=${1:-$default_python_version}
  shift

  echo "Installing PyEnv environment $mckit"
}

reset_env() {
  unset LD_PRELOAD
  pyenv local "$python_version"
  pyenv virtualenv-delete -f "$mckit"
  pyenv virtualenv "$python_version" "$mckit"
  pyenv local "$mckit" "3.8.12"

  # pip is obsolete almost always
  python -m pip install --upgrade pip

  # Fix LD_LIBRARY_PATH and so on
  source ./setenv.rc  reset

  poetry install
#  poetry install --extra

  mckit --version
  if [[ 0 != $? ]]; then
      echo "ERROR: failed to install mckit"
      return 1
  fi

  echo
  echo "SUCCESS: mckit has been installed"
  echo
}

check_environment() {
  poetry run pytest -m "not slow"
  if [[ 0 == $? ]]; then
      echo
      echo "SUCCESS: pytest is passed OK"
      echo
  else
      echo "ERROR: failed to run tests"
      return 1
  fi

  poetry run nox --list
  poetry run nox -s safety
  poetry run nox -s tests -p 3.9 -- -m "not slow" --cov

  ./create-jk.sh "$mckit"
  if [[ 0 != $? ]]; then
      return 1
  fi

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
