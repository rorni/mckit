"""Nox sessions.

See `Cjolowicz's article <https://cjolowicz.github.io/posts/hypermodern-python-03-linting>`_
"""
from typing import List

import platform
import shutil
import sys

from glob import glob
from pathlib import Path
from textwrap import dedent

import nox

try:
    from nox_poetry import Session, session  # mypy: ignore
except ImportError:
    message = f"""\
    Nox failed to import the 'nox-poetry' package.

    Please install it using the following command:

    {sys.executable} -m pip install nox-poetry"""
    raise SystemExit(dedent(message)) from None

# TODO dvp: uncomment when code and docs are more mature
nox.options.sessions = (
    "safety",
    # "isort",   isort and black are included to precommit
    # "black",
    "pre-commit",
    # TODO dvp: enable default runs with  lint and mypy when code matures and
    #           if these checks are not already enabled in pre-commit
    # "lint",
    # "mypy",
    # "xdoctest",  # TODO dvp: uncomment when doctests appear in the code (check with: xdoctest -c list mckit)
    "tests",
    # "docs-build",
)

package = "mckit"
locations = [package, "tests", "noxfile.py", "docs/source/conf.py"]

supported_pythons = ["3.8", "3.9", "3.10"]
black_pythons = "3.10"
mypy_pythons = "3.10"
lint_pythons = "3.10"

on_windows = platform.system() == "Windows"


def activate_virtualenv_in_precommit_hooks(s: Session) -> None:
    """Activate virtualenv in hooks installed by pre-commit.

    This function patches git hooks installed by pre-commit to activate the
    session's virtual environment. This allows pre-commit to locate hooks in
    that environment when invoked from git.

    Args:
        s: The Session object.
    """
    assert s.bin is not None  # noqa: S101

    virtualenv = s.env.get("VIRTUAL_ENV")
    if virtualenv is None:
        return

    hook_dir = Path(".git") / "hooks"
    if not hook_dir.is_dir():
        return

    for hook in hook_dir.iterdir():
        if hook.name.endswith(".sample") or not hook.is_file():
            continue

        text = hook.read_text()
        bin_dir = repr(s.bin)[1:-1]  # strip quotes
        if not (
            Path("A") == Path("a")
            and bin_dir.lower() in text.lower()
            or bin_dir in text
        ):
            continue

        lines = text.splitlines()
        if not (lines[0].startswith("#!") and "python" in lines[0].lower()):
            continue

        header = dedent(
            f"""\
            import os
            os.environ["VIRTUAL_ENV"] = {virtualenv!r}
            os.environ["PATH"] = os.pathsep.join((
                {s.bin!r},
                os.environ.get("PATH", ""),
            ))
            """
        )

        lines.insert(1, header)
        hook.write_text("\n".join(lines))


@session(name="pre-commit", python="3.10")
def precommit(s: Session) -> None:
    """Lint using pre-commit."""
    args = s.posargs or ["run", "--all-files", "--show-diff-on-failure"]
    s.install(
        "black",
        "darglint",
        "flake8",
        "flake8-annotations",
        "flake8-bandit",
        "flake8-black",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-rst-docstrings",
        "pep8-naming",
        "pre-commit",
        "pre-commit-hooks",
        "isort",
        "mypy",
        "types-setuptools",
    )
    s.run("pre-commit", *args)
    if args and args[0] == "install":
        activate_virtualenv_in_precommit_hooks(s)


@session(python="3.10")
def safety(s: Session) -> None:
    """Scan dependencies for insecure packages."""
    # args = s.posargs or ["--ignore", "44715"]  - was used because of old numpy issues, no issues now
    args = s.posargs
    requirements = s.poetry.export_requirements()
    s.install("safety")
    s.run("safety", "check", "--full-report", f"--file={requirements}", *args)


@session(python=supported_pythons)
def tests(s: Session) -> None:
    """Run the test suite."""
    s.run(
        "poetry",
        "install",
        "--no-dev",
        external=True,
    )
    s.install("coverage[toml]", "pytest", "pygments")
    try:
        s.run("coverage", "run", "--parallel", "-m", "pytest", *s.posargs)
    finally:
        if s.interactive:
            s.notify("coverage", posargs=[])


@session
def coverage(s: Session) -> None:
    """Produce the coverage report.

    To obtain html report run
        nox -rs coverage -- html
    """
    args = s.posargs or ["report"]

    s.install("coverage[toml]")

    if not s.posargs and any(Path().glob(".coverage.*")):
        s.run("coverage", "combine")

    s.run("coverage", *args)


@session(python=supported_pythons)
def typeguard(s: Session) -> None:
    """Runtime type checking using Typeguard."""
    s.run(
        "poetry",
        "install",
        "--no-dev",
        external=True,
    )
    s.install("pytest", "typeguard", "pygments")
    s.run("pytest", f"--typeguard-packages={package}", *s.posargs)


@session(python="3.10")
def isort(s: Session) -> None:
    """Organize imports."""
    s.install("isort")
    search_patterns = [
        # "*.py",
        "mckit/*.py",
        "tests/*.py",
        "benchmarks/*.py",
        "profiles/*.py",
        #        "adhoc/*.py",
    ]
    # exclude = ["setup-generated.py", "setup-bak.py", "setup"]
    #
    # def skip(path: str) -> bool:
    #     return not ("example" in path or path.startswith("setup") or path in exclude)

    # files_to_process: List[str] = filter(
    #     skip, sum(map(lambda p: glob(p, recursive=True), search_patterns), [])
    # )
    files_to_process: List[str] = sum(
        map(lambda p: glob(p, recursive=True), search_patterns), []
    )

    s.run(
        "isort",
        "--check",
        "--diff",
        *files_to_process,
        external=True,
    )


@session(python=black_pythons)
def black(s: Session) -> None:
    """Run black code formatter."""
    args = s.posargs or locations
    s.install("black")
    s.run("black", *args)


@session(python=lint_pythons)
def lint(s: Session) -> None:
    """Lint using flake8."""
    args = s.posargs or locations
    s.install(
        "flake8",
        "flake8-annotations",
        "flake8-bandit",
        "flake8-black",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-rst-docstrings",
        "flake8-import-order",
        "darglint",
    )
    s.run("flake8", *args)


@session(python=mypy_pythons)
def mypy(s: Session) -> None:
    """Type-check using mypy."""
    args = s.posargs or ["mckit", "tests", "docs/source/conf.py"]
    s.run(
        "poetry",
        "install",
        "--no-dev",
        external=True,
    )
    s.install("mypy", "pytest", "types-setuptools")
    s.run("mypy", *args)
    if not s.posargs:
        s.run("mypy", f"--python-executable={sys.executable}", "noxfile.py")


@session(python=supported_pythons)
def wheels(s: Session) -> None:
    """Build wheels and install from wheels."""
    s.run(
        "poetry",
        "install",
        "--no-dev",
        "--no-root",
        external=True,
    )
    dist_dir = Path("dist")  # noqa
    if dist_dir.exists():
        shutil.rmtree(str(dist_dir))
    s.run(
        "poetry",
        "build",
        "--format",
        "wheel",
        external=True,
    )
    if not dist_dir.exists():
        s.error("'dist' directory is not created on poetry build")
        return
    wheel_path = next(dist_dir.glob("*.whl")).absolute()
    s.log(f"Installing mckit from wheel {wheel_path}")
    s.run(
        "python",
        "-m",
        "pip",
        "install",
        str(wheel_path),
        external=True,
    )
    s.run(
        "mckit",
        "--version",
        external=True,
    )


@session(python=supported_pythons)
def xdoctest(s: Session) -> None:
    """Run examples with xdoctest."""
    args = s.posargs or ["all"]
    s.run(
        "poetry",
        "install",
        "--no-dev",
        external=True,
    )
    s.install("xdoctest[colors]")
    s.run("python", "-m", "xdoctest", package, *args)


@session(name="docs-build", python="3.9")
def docs_build(s: Session) -> None:
    """Build the documentation."""
    args = s.posargs or ["docs/source", "docs/_build"]
    s.run(
        "poetry",
        "install",
        "--no-dev",
        external=True,
    )
    s.install(
        "sphinx",
        "sphinx-click",
        "sphinx-rtd-theme",
        # "sphinxcontrib-htmlhelp",
        # "sphinxcontrib-jsmath",
        "sphinxcontrib-napoleon",
        # "sphinxcontrib-qthelp",
        "sphinx-autodoc-typehints",
        # "sphinx_autorun",
        "numpydoc",
    )

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    s.run("sphinx-build", *args)


@session(python="3.10")
def docs(s: Session) -> None:
    """Build and serve the documentation with live reloading on file changes."""
    args = s.posargs or ["--open-browser", "docs/source", "docs/_build"]
    s.run(
        "poetry",
        "install",
        "--no-dev",
        external=True,
    )
    s.install(
        "sphinx",
        "sphinx-autobuild",
        "sphinx-click",
        "sphinx-rtd-theme",
        # "sphinxcontrib-htmlhelp",
        # "sphinxcontrib-jsmath",
        # "sphinxcontrib-napoleon",
        # "sphinxcontrib-qthelp",
        # "sphinx-autodoc-typehints",
        # "sphinx_autorun",
    )

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    s.run("sphinx-autobuild", *args)
