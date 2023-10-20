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

from typing import TYPE_CHECKING, Annotated

import os
import sqlite3 as sq
import sys

from pathlib import Path
from time import time_ns

import dotenv
import numpy as np

from loguru import logger
from tqdm import tqdm

import typer

from mckit.geometry import EX, EY, EZ, Box
from mckit.parser import ParseResult, from_file

if TYPE_CHECKING:
    from sqlite3 import Connection

    from mckit import Material, Universe

dotenv.load_dotenv()

__version__ = "0.0.1"

DEBUG = os.getenv("MCKIT_DEBUG") is not None

LOG_FORMAT = os.getenv(
    "MCKIT_LOG_FORMAT",
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>",
)


def _init_logger(level="INFO", _format=LOG_FORMAT, log_path=None) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format=_format,
        level=level,
        backtrace=True,
        diagnose=True,
    )
    """Initialize logging to stderr and file."""
    if log_path:
        logger.add(log_path, backtrace=True, diagnose=True)


WRK_DIR = Path(os.getenv("MCKIT_MODEL", "~/dev/mcnp/ep11/wrk/pc/Co05-volumes")).expanduser()
if not WRK_DIR.is_dir():
    raise FileNotFoundError(WRK_DIR)

MODEL_PATH = WRK_DIR / "pc11_shields8_Co05.i"
if not MODEL_PATH.is_file():
    raise FileNotFoundError(MODEL_PATH)


def _from_bounds(
    minx: float, maxx: float, miny: float, maxy: float, minz: float, maxz: float
) -> Box:
    min_vals = np.array([minx, miny, minz])
    max_vals = np.array([maxx, maxy, maxz])
    if not np.all(min_vals < max_vals):
        raise ValueError("Unsorted boundaries values")
    center = 0.5 * (min_vals + max_vals)
    widths = max_vals - min_vals
    return Box(center, *widths, ex=EX, ey=EY, ez=EZ)


def create_global_box() -> Box:
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

    Returns:
        Box: box with the given boundaries
    """
    global_box = _from_bounds(830.0, 3350, -1200.0, 1200.0, -620, 740)
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


def _init_data_base(con: Connection) -> None:
    logger.info("Deleting the old tables")
    con.executescript(
        """
        drop table if exists cell_geometry;
        drop table if exists cell_material;
        """
    )
    logger.info("Creating tables cell_geometry and cell_material")
    con.executescript(
        """
        create table cell_geometry (
            cell int primary key,
            box_min_x real,
            box_max_x real,
            box_min_y real,
            box_max_y real,
            box_min_z real,
            box_max_z real,
            box_time  int,
            volume real,
            min_vol real,
            vol_time int
        );
        create table cell_material (
            cell int primary key,
            material int,
            density real
        );
        """
    )
    logger.info("Creating view material_mass")
    con.executescript(
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
    logger.info("Database is initialized")


def _collect_model_info(model: Universe, con: Connection, global_box: Box, min_vol: float = 0.25):
    _init_data_base(con)
    for i, c in enumerate(tqdm(model)):
        m: Material = c.material()
        if m is not None:
            cell = c.name()
            material = m.composition.name()
            density = m.density
            con.execute(
                """
                insert into cell_material (
                    cell, material, density
                )
                values (?, ?, ?)
                """,
                (cell, material, density),
            )
            box_time = -time_ns()
            bounding_box = c.shape.bounding_box(box=global_box, tol=2.0)
            box_time += time_ns()
            (
                (box_min_x, box_max_x),
                (box_min_y, box_max_y),
                (box_min_z, box_max_z),
            ) = bounding_box.bounds
            logger.debug(
                "Added box for cell {}: {} {} {}, {}, {}",
                cell,
                box_min_x,
                box_max_x,
                box_min_y,
                box_max_y,
                box_min_z,
                box_max_z,
            )
            vol_time = -time_ns()
            volume = c.shape.volume(bounding_box, min_vol)
            vol_time += time_ns()
            con.execute(
                """
                insert into cell_geometry (
                    cell,
                    box_min_x, box_max_x, box_min_y, box_max_y, box_min_z, box_max_z, box_time,
                    volume, min_vol, vol_time
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cell,
                    box_min_x,
                    box_max_x,
                    box_min_y,
                    box_max_y,
                    box_min_z,
                    box_max_z,
                    box_time,
                    volume,
                    min_vol,
                    vol_time,
                ),
            )
            logger.debug(
                "Added volume for cell {}: {}, rel: {}",
                cell,
                volume,
                min_vol / volume if volume > 0 else 1.0,
            )
            con.commit()
            if DEBUG and i > 2:
                break


app = typer.Typer()


@app.command()
def fill(
    db_path: Annotated[
        Path,
        typer.Option(
            help="Path to the sqlite database to be created",
        ),
    ] = WRK_DIR
    / "model-info.sqlite",
    model_path: Annotated[
        Path,
        typer.Option(
            help="Path to the input MCNP model",
        ),
    ] = WRK_DIR
    / "pc11_shields8_Co05.i",
) -> None:
    """Collect information from an MCNP model to database."""
    logger.info("Creating model info database {}", db_path)
    con = sq.connect(db_path)
    try:
        global_box = create_global_box()
        model_parse = load_mcnp_model(model_path)
        model = model_parse.universe
        _collect_model_info(model, con, global_box)
    finally:
        con.close()
    logger.success("The database {} is created", db_path)


@app.command()
def add_materials(
    db_path: Annotated[
        Path,
        typer.Option(
            help="Path to the sqlite database to be created",
        ),
    ] = WRK_DIR
    / "model-info.sqlite",
    model_path: Annotated[
        Path,
        typer.Option(
            help="Path to the input MCNP model",
        ),
    ] = WRK_DIR
    / "pc11_shields8_Co05.i",
) -> None:
    """Load material compositions to the database."""
    logger.info("Adding  materials info database {}", db_path)
    con = sq.connect(db_path)
    try:
        con.executescript(
            """
            drop table if exists compositions;
            create table compositions(
                material_id int primary key,
                charge int,
                mass_number int,
                lib text,
                isomer int,
                fraction real
            );
        """
        )
        model_parse = load_mcnp_model(model_path)
        model = model_parse.universe
        compositions = model.get_compositions()

        for composition in compositions:
            material_id = composition.name()
            if material_id is None:
                raise ValueError("Material is not specified")
            for element, fraction in composition:
                con.execute(
                    """
                    insert into compositions (
                        material_id,
                        charge,
                        mass_number,
                        lib,
                        isomer,
                        fraction
                    ) values (
                        ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        material_id,
                        element.charge,
                        element.mass_number,
                        element.lib,
                        element.isomer,
                        fraction,
                    ),
                )

    finally:
        con.close()
    logger.success("The database {} is created", db_path)


@app.command()
def analyze(
    db_path: Annotated[
        Path,
        typer.Option(
            help="Path to the sqlite database to be created",
        ),
    ] = WRK_DIR
    / "model-info.sqlite",
    suspicious: Annotated[
        int, typer.Option(help="Material id to find the mass ratio to total")
    ] = 207302,
) -> None:
    """Analyze the information on model.

    - Define ratio of a given "suspicious" material to total mass.
    """
    logger.info("Analyzing the MCNP model volumes.")
    con = sq.connect(db_path)
    try:
        total_mass = (
            con.execute(
                """
                select sum(mass) from material_mass
                """
            ).fetchone()[0]
            * 0.001
        )
        """Total mass of materials in kg."""
        logger.info("Total mass of materials {:3g}kg", total_mass)

        mass_207302_fetch = con.execute(
            """
            select sum(mass) from material_mass where material = ?
            """,
            (suspicious,),
        ).fetchone()
        if mass_207302_fetch:
            mass_207302 = mass_207302_fetch[0] * 0.001
            """Mass of m207302 in kg."""
            logger.info("Mass of material 207302 {:3g}kg", mass_207302)

            ratio = mass_207302 / total_mass

            logger.info("Ratio is {:3g}%", ratio * 100)
        else:
            logger.warning("Cannot find material {}")
    finally:
        con.close()
    logger.success("The database {} is analyzed", db_path)


@app.callback()
def start(
    log_path: Annotated[
        Path,
        typer.Option(help="Path to log file."),
    ] = WRK_DIR
    / "model-info.log"
) -> None:
    """Present information on MCNP cells volumes and masses."""
    _init_logger(log_path=log_path)
    logger.info("{}, v{}", Path(__file__).name, __version__)


if __name__ == "__main__":
    app()
