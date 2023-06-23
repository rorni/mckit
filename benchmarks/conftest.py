from __future__ import annotations

from typing import TYPE_CHECKING

from zipfile import ZipFile

import pytest

from mckit.constants import MCNP_ENCODING
from mckit.parser import from_text
from mckit.utils import path_resolver

if TYPE_CHECKING:
    from mckit import Universe


@pytest.fixture(scope="session")
def data():
    """Benchmarks data folder."""
    return path_resolver("benchmarks")


@pytest.fixture(scope="session")
def clite_text(data) -> str:
    """C-lite model text.

    Returns:
        Loaded text of a C-lite model.
    """
    with ZipFile(data("data/4M.zip")) as data_archive:
        return data_archive.read("clite.i").decode(encoding=MCNP_ENCODING)


@pytest.fixture(scope="session")
def clite_model(clite_text) -> Universe:
    """Load c-lite model.

    Args:
        clite_text: c-lite model text

    Returns:
        loaded universe
    """
    return from_text(clite_text)
