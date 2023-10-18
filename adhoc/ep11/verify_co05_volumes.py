#!python
# noinspection GrazieInspection
"""Find ratio of a volume fo steel with Co0.5 fraction to other steel volume.

The model pc11_shields8_Co05.i is variant where the specific material was applied for
frame parts, which are in the integrator scope - fraction of Co was increased to 0.5%.
The material's id is 207302.
All other parts (diagnostics) use the same materials as in baseline model.

We want to show that the portion of the specific steel is much less than of other steel
and cannot affect SDDR value too much.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import os
import sqlite3 as sq
import sys

from pathlib import Path

import dotenv
import numpy as np

from loguru import logger
from tqdm import tqdm

from mckit.geometry import EX, EY, EZ, Box
from mckit.parser import ParseResult, from_file

if TYPE_CHECKING:
    from sqlite3 import Connection

    from mckit import Material, Universe

dotenv.load_dotenv()


DEBUG = True
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)


def init_logger(level="INFO", _format=LOG_FORMAT, log_path="model-info.log"):
    logger.remove()
    logger.add(
        sys.stderr,
        format=_format,
        level=level,
        backtrace=True,
        diagnose=True,
    )
    """Initialize logging to stderr and file."""
    logger.add(log_path, backtrace=True, diagnose=True)


WRK_DIR = Path(os.getenv("MCKIT_MODEL", "~/dev/mcnp/ep11/wrk/pc/Co05-volumes")).expanduser()
if not WRK_DIR.is_dir():
    raise FileNotFoundError(WRK_DIR)

MODEL_PATH = WRK_DIR / "pc11_shields8_Co05.i"
if not MODEL_PATH.is_file():
    raise FileNotFoundError(MODEL_PATH)


def from_bounds(
    minx: float, maxx: float, miny: float, maxy: float, minz: float, maxz: float
) -> Box:
    min_vals = np.array([minx, miny, minz])
    max_vals = np.array([maxx, maxy, maxz])
    if not np.all(min_vals < max_vals):
        raise ValueError("Unsorted boundaries values")
    center = 0.5 * (min_vals + max_vals)
    widths = max_vals - min_vals
    return Box(center, *widths, ex=EX, ey=EY, ez=EZ)


def create_global_box():
    """Define the box surrounding the model.

    This cell 196000 with zero importance.
    Surfaces:
    206001:-206002:206003:206005:-206004:-206000
    206000 PX 830
    206001 PX 3350
    206002 PY -1200
    206003 PY 1200
    206004 PZ -620
    206005 PZ 740
    """
    global_box = from_bounds(830.0, 3350, -1200.0, 1200.0, -620, 740)
    logger.info("Global box: {}", global_box)
    logger.info("Global box volume: {}", global_box.volume)
    return global_box


def load_mcnp_model(model_path: Path = MODEL_PATH) -> ParseResult:
    """Load MCNP model.

    Args:
        model_path: path to the model

    Returns:
        ParseResult: Universe and indexes for cells, surfaces transformations and materials
    """
    logger.info("Loading model from {}", model_path)
    model_info = from_file(MODEL_PATH)
    model = model_info.universe
    logger.info("Loaded model with {} cells", len(model))
    return model_info


def init_data_base(_conn: Connection) -> None:
    _conn.executescript(
        """
        drop table if exists cell_geometry;
        drop table if exists cell_material;
        """
    )
    _conn.executescript(
        """
        create table cell_geometry (
            cell int primary key,
            box_min_x real,
            box_max_x real,
            box_min_y real,
            box_max_y real,
            box_min_z real,
            box_max_z real,
            volume real,
            min_vol real
        );
        create table cell_material (
            cell int primary key,
            material int,
            density real
        );
        """
    )
    _conn.executescript(
        """
        drop view if exists material_mass;
        create view material_mass as
        select
            a.material,
            sum(volume * density) as mass
        from
            cell_material a
            inner join cell_geometry b on a.cell = b.cell
        group by
            a.material
        """
    )


def collect_model_info(model: Universe, _conn: Connection, global_box: Box, min_vol: float = 0.25):
    for c in tqdm(model):
        m: Material = c.material()
        if m is not None:
            cell = c.name()
            material = m.composition.name()
            density = m.density
            _conn.execute(
                """
                insert into cell_material (
                    cell, material, density
                )
                values (?, ?, ?)
                """,
                (cell, material, density),
            )
            bounding_box = c.shape.bounding_box(box=global_box, tol=1.0)
            (
                (box_min_x, box_max_x),
                (box_min_y, box_max_y),
                (box_min_z, box_max_z),
            ) = bounding_box.bounds
            volume = c.shape.volume(bounding_box, min_vol)
            _conn.execute(
                """
                insert into cell_geometry (
                    cell, box_min_x, box_max_x, box_min_y, box_max_y, box_min_z, box_max_z, volume, min_vol
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cell,
                    box_min_x,
                    box_max_x,
                    box_min_y,
                    box_max_y,
                    box_min_z,
                    box_max_z,
                    volume,
                    min_vol,
                ),
            )


def fill_data_base(_conn: Connection, model_path: Path) -> None:
    """Collect information on a MCNP model."""
    global_box = create_global_box()
    model_parse = load_mcnp_model(model_path)
    model = model_parse.universe
    collect_model_info(model, _conn, global_box)


def analyze(_conn: Connection) -> None:
    logger.info("Analyzing the MCNP model volumes.")
    init_data_base(_conn)
    _conn.executescript(
        """
        drop view if exists material_mass;
        create view material_mass as
        select
            a.material,
            sum(volume * density) as mass
        from
            cell_material a
            inner join cell_geometry b on a.cell = b.cell
        group by
            a.material
    """
    )

    total_mass = (
        _conn.execute(
            """
        select sum(mass) from material_mass
    """
        ).fetchone()[0]
        * 0.001
    )
    """Total mass of materials in kg."""
    logger.info("Total mass of materials {:3g}kg", total_mass)

    mass_207302 = (
        _conn.execute(
            """
        select sum(mass) from material_mass where material == 207302
    """
        ).fetchone()[0]
        * 0.001
    )
    """Mass of m207302 in kg."""
    logger.info("Mass of material 207302 {:3g}kg", mass_207302)

    ratio = mass_207302 / total_mass

    logger.info("Ratio is {:3g}%", ratio * 100)


def main():
    init_logger(log_path=WRK_DIR / "model-info.log")
    db_path = WRK_DIR / "model-info.sqlite"
    logger.info("Creating model info database {}", db_path)
    conn = sq.connect(db_path)
    try:
        fill_data_base(conn, MODEL_PATH)
        analyze(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
