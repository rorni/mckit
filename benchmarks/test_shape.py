"""Benchmark computationally intensive Shape class methods."""
from __future__ import annotations

from typing import TYPE_CHECKING

from zipfile import ZipFile

import pytest

from mckit import Shape, Universe
from mckit.box import Box
from mckit.constants import MCNP_ENCODING
from mckit.parser import from_text
from mckit.utils import path_resolver

if TYPE_CHECKING:
    from mckit import Body


data = path_resolver("benchmarks")
with ZipFile(data("data/4M.zip")) as data_archive:
    clite_text = data_archive.read("clite.i").decode(encoding=MCNP_ENCODING)
clite_model = from_text(clite_text).universe


def sample_by_complexity(model: Universe, size=1000, step=1) -> list[tuple[int, Body]]:
    complex_cells = sorted(model, key=lambda x: x.shape.complexity(), reverse=True)[
        : size * step : step
    ]

    def _mapper(cell):
        complexity = cell.shape.complexity()
        return pytest.param(complexity, cell, id=f"{complexity}-{cell.name()}")

    return [_mapper(cell) for cell in complex_cells]


@pytest.mark.parametrize("complexity, cell", sample_by_complexity(clite_model))
def test_bounding_box(benchmark, complexity, cell) -> None:
    gb = Box([692.0, 27.0, -313.0], 3700.0, 1400.0, 4500.0)
    benchmark.extra_info["complexity"] = complexity
    benchmark.extra_info["cell"] = cell.name()
    shape = cell.shape
    benchmark.pedantic(Shape.bounding_box, args=(shape,), kwargs={"box": gb, "tol": 10.0})


def test_universe_bounding_box(benchmark) -> None:
    gb = Box([692.0, 27.0, -313.0], 3700.0, 1400.0, 4500.0)
    box = benchmark.pedantic(
        Universe.bounding_box, args=(clite_model,), kwargs={"box": gb, "tol": 20.0}
    )
    assert box.center == pytest.approx([692.13867188, 27.46582031, -312.5])
    assert box.dimensions == pytest.approx([3615.72265625, 1351.318359375, 4375.0])


if __name__ == "__main__":
    pytest.main(["--benchmark-enable", "--benchmark-autosave"])
