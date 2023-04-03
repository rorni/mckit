"""Nox sessions."""
from __future__ import annotations

from typing import Final

import re
import shutil
import sys

from glob import glob
from pathlib import Path
from textwrap import dedent

import nox

from nox import Session, session

nox.options.sessions = (
    # "safety",
    "pre-commit",
    "tests",
    "docs-build",
)


NAME_RGX = re.compile(r'name\s*=\s*"(?P<package>[-_a-zA-Z0-9]+)"')


def find_my_name() -> str:
    """Find this package name.

    Search the first row in pyproject.toml in format

        name = "<package>"

    and returns the <package> value.
    This allows to reuse the noxfile.py in similar projects
    without any changes.

    Returns:
        str: Current package found in pyproject.toml

    Raises:
        ValueError: if the pattern is not found.
    """
    with Path("pyproject.toml").open() as fid:
        for line in fid:
            res = NAME_RGX.match(line)
            if res is not None:
                return res["package"].replace("-", "_")
    msg = "Cannot find package name"
    raise ValueError(msg)


package: Final = find_my_name()
locations: Final = f"src/{package}", "tests", "./noxfile.py", "docs/source/conf.py"

supported_pythons: Final = "3.8", "3.9", "3.10", "3.11"


def _update_hook(hook: Path, virtualenv: str, s: Session) -> None:
    text = hook.read_text()
    bin_dir = repr(s.bin)[1:-1]  # strip quotes
    if Path("A") == Path("a") and bin_dir.lower() in text.lower() or bin_dir in text:
        lines = text.splitlines()
        if lines[0].startswith("#!") and "python" in lines[0].lower():
            header = dedent(
                f"""\
                import os
                os.environ["VIRTUAL_ENV"] = {virtualenv!r}
                os.environ["PATH"] = os.pathsep.join((
                    {s.bin!r},
                    os.environ.get("PATH", ""),
                ))
                """,
            )
            lines.insert(1, header)
            hook.write_text("\n".join(lines))


def activate_virtualenv_in_precommit_hooks(s: Session) -> None:
    """Activate virtualenv in hooks installed by pre-commit.

    This function patches git hooks installed by pre-commit to activate the
    session's virtual environment. This allows pre-commit to locate hooks in
    that environment when invoked from git.

    Args:
        s: The Session object.
    """
    virtualenv = s.env.get("VIRTUAL_ENV")
    if virtualenv is None:
        return

    hook_dir = Path(".git") / "hooks"
    if not hook_dir.is_dir():
        return

    for hook in filter(
        lambda x: not x.name.endswith(".sample") and x.is_file(), hook_dir.iterdir()
    ):
        _update_hook(hook, virtualenv, s)


@session(name="pre-commit")
def precommit(s: Session) -> None:
    """Lint using pre-commit."""
    args = s.posargs or ["run", "--all-files", "--show-diff-on-failure"]
    s.run(
        "poetry",
        "install",
        "--no-root",
        "--only",
        "pre_commit,style,isort,black,ruff",
        external=True,
    )
    s.run("pre-commit", *args)
    if args and args[0] == "install":
        activate_virtualenv_in_precommit_hooks(s)


# @session
# def safety(s: Session) -> None:
#     """Scan dependencies for insecure packages."""
#     requirements = f"{s.virtualenv.location}/safety-requirements.txt"
#     s.run(
#         "poetry",
#         "export",
#         "-f",
#         "requirements.txt",
#         "-o",
#         requirements,
#         "--only",
#         "main",
#         external=True,
#     )
#     s.install("safety")
#     s.run("safety", "check", "--full-report", f"--file={requirements}", *s.posargs)


@session(python=supported_pythons)
def tests(s: Session) -> None:
    """Run the test suite."""
    s.run(
        "poetry",
        "install",
        "--only",
        "main,test,xdoctest,coverage",
        external=True,
    )
    try:
        s.run("coverage", "run", "--parallel", "-m", "pytest", *s.posargs)
    finally:
        if s.interactive and "--no-cov" not in s.posargs:
            s.notify("coverage", posargs=[])


@session
def coverage(s: Session) -> None:
    """Produce the coverage report.

    To obtain html report run
        nox -rs coverage -- html
    """
    args = s.posargs or ["report"]

    s.run(
        "poetry",
        "install",
        "--no-root",
        "--only",
        "coverage",
        external=True,
    )

    if not s.posargs and any(Path().glob(".coverage.*")):
        s.run("coverage", "combine")

    s.run("coverage", *args)


@session
def typeguard(s: Session) -> None:
    """Runtime type checking using Typeguard."""
    s.run(
        "poetry",
        "install",
        "--only",
        "main,test,typeguard",
        external=True,
    )
    s.run("pytest", "--typeguard-packages=src", *s.posargs, external=True)


@session
def isort(s: Session) -> None:
    """Organize imports."""
    search_patterns = [
        "*.py",
        f"src/{package}/*.py",
        "tests/*.py",
        "benchmarks/*.py",
        "profiles/*.py",
        "adhoc/*.py",
    ]
    files_to_process: list[str] = sum((glob(p, recursive=True) for p in search_patterns), [])
    if files_to_process:
        s.run(
            "poetry",
            "install",
            "--no-root",
            "--only",
            "isort",
            external=True,
        )
        s.run(
            "pycln",
            "--check",
            "--diff",
            *files_to_process,
            external=True,
        )
        s.run(
            "isort",
            "--check",
            "--diff",
            *files_to_process,
            external=True,
        )


@session
def black(s: Session) -> None:
    """Run black code formatter."""
    args = s.posargs or locations
    s.run(
        "poetry",
        "install",
        "--no-root",
        "--only",
        "black",
        external=True,
    )
    s.run("black", *args)


@session
def lint(s: Session) -> None:
    """Lint using flake8."""
    args = s.posargs or locations
    s.run(
        "poetry",
        "install",
        "--no-root",
        "--only",
        "flake8",
        external=True,
    )
    s.run("flake8", *args)


@session
def mypy(s: Session) -> None:
    """Type-check using mypy."""
    args = s.posargs or ["src", "docs/source/conf.py"]
    s.run(
        "poetry",
        "install",
        "--no-root",
        "--only",
        "main,mypy",
        external=True,
    )
    s.run("mypy", *args)

    # special case for noxfile.py: need to find `nox` itself in session
    if not s.posargs:
        s.run("mypy", f"--python-executable={sys.executable}", "noxfile.py")


@session(python="3.11")
def xdoctest(s: Session) -> None:
    """Run examples with xdoctest."""
    args = s.posargs or ["--quiet", "-m", f"src/{package}"]
    s.run(
        "poetry",
        "install",
        "--no-root",
        "--only",
        "main,xdoctest",
        external=True,
    )
    s.run("python", "-m", "xdoctest", *args)


@session(python="3.11")
def ruff(s: Session) -> None:
    """Run ruff linter."""
    args = s.posargs or ["src", "tests"]
    s.run(
        "poetry",
        "install",
        "--no-root",
        "--only",
        "main,ruff",
        external=True,
    )
    s.run("ruff", *args)


@session(name="docs-build", python="3.11")
def docs_build(s: Session) -> None:
    """Build the documentation."""
    args = s.posargs or ["docs/source", "docs/_build"]
    s.run(
        "poetry",
        "install",
        "--only",
        "main,docs",
        external=True,
    )
    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    s.run("sphinx-build", *args)


@session(python="3.11")
def docs(s: Session) -> None:
    """Build and serve the documentation with live reloading on file changes."""
    args = s.posargs or ["--open-browser", "docs/source", "docs/_build"]
    s.run(
        "poetry",
        "install",
        "--only",
        "main,docs,docs_auto",
        external=True,
    )
    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    s.run("sphinx-autobuild", *args)
