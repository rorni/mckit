# noxfile.py
"""
    Nox sessions.

    See `Cjolowicz's article <https://cjolowicz.github.io/posts/hypermodern-python-03-linting>`_
"""
from typing import Any, Generator, List

import os
import platform
import tempfile

from contextlib import contextmanager
from glob import glob
from pathlib import Path

import nox

from nox.sessions import Session

# TODO dvp: uncomment when code and docs are more mature
nox.options.sessions = (
    "safety",
    "isort",
    "black",
    # "lint",
    # "mypy",
    "xdoctest",
    "tests",
    # "codecov",
    # "docs",
)

locations = "mckit", "tests", "noxfile.py", "docs/source/conf.py"

supported_pythons = "3.9 3.8 3.7".split()
black_pythons = "3.9"
mypy_pythons = "3.9"
lint_pythons = "3.9"

on_windows = platform.system() == "Windows"


@contextmanager
def collect_dev_requirements(session: Session) -> Generator[str, None, None]:
    req_path = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
    try:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--without-hashes",
            "--format=requirements.txt",
            f"--output={req_path}",
            external=True,
        )
        yield req_path
    finally:
        os.unlink(req_path)


# see https://stackoverflow.com/questions/59768651/how-to-use-nox-with-poetry
def install_with_constraints(session: Session, *args: str, **kwargs: Any) -> None:
    """
    Install packages constrained by Poetry's lock file.

    This function is a wrapper for nox.sessions.Session.install. It
    invokes pip to install packages inside of the session's virtualenv.
    Additionally, pip is passed a constraints file generated from
    Poetry's lock file, to ensure that the packages are pinned to the
    versions specified in poetry.lock. This allows you to manage the
    packages as Poetry development dependencies.

    Arguments:
        session: The Session object.
        args: Command-line arguments for pip.
        kwargs: Additional keyword arguments for Session.install.
    """
    with collect_dev_requirements(session) as req_path:
        session.install(f"--constraint={req_path}", *args, **kwargs)


@nox.session(python=supported_pythons, venv_backend="venv")
def tests(session: Session) -> None:
    """Run the test suite."""
    args = session.posargs or ["--cov", "-m", "not e2e"]
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(session, "pytest", "pytest-cov", "pytest-mock", "coverage")
    path = Path(session.bin).parent
    if on_windows:
        session.bin_paths.insert(
            0, str(path / "Library/bin")
        )  # here all the DLLs should be installed
    session.log(f"Session path: {session.bin_paths}")
    session.run("pytest", env={"LD_LIBRARY_PATH": str(path / "lib")}, *args)
    if "--cov" in args:
        session.run("coverage", "report", "--show-missing", "--skip-covered")
        session.run("coverage", "html")


@nox.session(python=lint_pythons)
def lint(session: Session) -> None:
    """Lint using flake8."""
    args = session.posargs or locations
    install_with_constraints(
        session,
        "flake8",
        "flake8-annotations",
        "flake8-bandit",
        "flake8-black",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-import-order",
        "darglint",
    )
    session.run("flake8", *args)


@nox.session(python=black_pythons)  # TODO dvp: this doesn't work with 3.8 so far
def black(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or locations
    install_with_constraints(session, "black")
    session.run("black", *args)


@nox.session(python="3.9")
def safety(session: Session) -> None:
    """Scan dependencies for insecure packages."""
    with collect_dev_requirements(session) as req_path:
        install_with_constraints(session, "safety")
        session.run("safety", "check", f"--file={req_path}", "--full-report")


#  This dangerous on ill complex project: may cause cyclic dependency
#  on partial imports ( from ... import).
#  Uncomment when proper imports or noorder directive is applied in sensitive files.
#  Always test after reorganizing ill projects.
#
@nox.session(python="3.9")
def isort(session: Session) -> None:
    """Organize imports"""

    install_with_constraints(session, "isort")
    search_patterns = [
        "*.py",
        "mckit/*.py",
        "tests/*.py",
        "benchmarks/*.py",
        "profiles/*.py",
        #        "adhoc/*.py",
    ]
    files_to_process: List[str] = sum(
        map(lambda p: glob(p, recursive=True), search_patterns), []
    )
    session.run(
        "isort",
        "--check",
        "--diff",
        *files_to_process,
        external=True,
    )


@nox.session(python=mypy_pythons)
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or locations
    install_with_constraints(session, "mypy")
    session.run(
        "mypy",
        # "--config",
        # "mypy.ini",  # TODO dvp: compute path to ini-file from test environment: maybe search upward.
        *args,
    )


@nox.session(python=supported_pythons)
def xdoctest(session: Session) -> None:
    """Run examples with xdoctest."""
    args = session.posargs or ["mckit"]
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(session, "xdoctest")
    session.run("python", "-m", "xdoctest", *args)


@nox.session(python="3.8")
def docs(session: Session) -> None:
    """Build the documentation."""
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(
        session,
        "sphinx",
        "sphinx-autobuild",
        "numpydoc",
        "sphinxcontrib-htmlhelp",
        "sphinxcontrib-jsmath",
        "sphinxcontrib-napoleon",
        "sphinxcontrib-qthelp",
        "sphinx-autodoc-typehints",
        "sphinx_autorun",
        "sphinx-rtd-theme",
    )
    if session.interactive:
        session.run(
            "sphinx-autobuild",
            "--port=0",
            "--open-browser",
            "docs/source",
            "docs/_build/html",
        )
    else:
        session.run("sphinx-build", "docs/source", "docs/_build")


@nox.session(python="3.8")
def codecov(session: Session) -> None:
    """Upload coverage data."""
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(
        session,
        "coverage[toml]",
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "coverage",
        "codecov",
    )
    # install_with_constraints(session, "coverage[toml]", "codecov")
    session.run("coverage", "xml", "--fail-under=0")
    session.run("codecov", *session.posargs)


@nox.session(python="3.9", venv_backend="venv")
def test_nox(session: Session) -> None:
    path = Path(session.bin)
    print("bin", path.parent)
    session.run("pip", "install", ".")
