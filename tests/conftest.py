from __future__ import annotations

import os

import pytest


@pytest.fixture()
def cd_tmpdir(tmpdir):  # noqa: PT004
    """Temporarily switch to temp directory.

    Args:
        tmpdir: pytest fixture for temp directory

    Yields:
        None
    """
    old_dir = tmpdir.chdir()
    try:
        yield
    finally:
        os.chdir(old_dir)
