"""Extract lost particles from MCNP output.

Create tables:
    - cell_fail, surface, cell_in, x, y, z, u, v, w, out_file, lp_no, hist_no
    - cell_fail, count  -  sorted by count in descending order

Create comin from MCNP plot by template replacing coordinates with coordinates of the most frequent
cell fail.
"""
from __future__ import annotations

from typing import cast

import re
import sqlite3 as sq
import sys

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

DESCRIPTION_START_RE = re.compile(r"^1\s+lost particle no.\s*(?P<lp_no>\d+)")
HISTORY_NO_RE = re.compile(r"history no.\s+(?P<hist_no>\d+)$")


# noinspection SqlDialectInspection,SqlNoDataSourceInspection
def setup_db(con: sq.Cursor) -> None:
    """Setup data base."""
    con.executescript(
        """
        drop table if exists lost_particles;
        create table lost_particles (
            cell_fail integer not null,
            surface integer not null,
            cell_in integer not null,
            x real not null,
            y real not null,
            z real not null,
            u real not null,
            v real not null,
            w real not null,
            lp_no integer not null,
            hist_no integer not null,
            out_file text not null
        );
        """
    )


def extract_descriptions(p: Path) -> Iterator[list[str]]:
    """Extract lost particles descriptions from MCNP output file.

    Args:
        p: path to MCNP output file

    Yields:
        portion of lines corresponding to a lost particle
    """
    wait_description = True
    description = []
    with p.open() as fid:
        for line in fid:
            if wait_description:
                if line.startswith("1   lost particle no."):
                    description.append(line)
                    wait_description = False
            else:
                description.append(line)
                if len(description) > 15:
                    wait_description = True
                    yield description
                    description = []


@dataclass
class _Description:
    cell_fail: int
    surface: int
    cell_in: int
    point: bool
    x: float
    y: float
    z: float
    u: float
    v: float
    w: float
    lp_no: int
    hist_no: int


def _parse_description(lines: list[str]) -> _Description:
    match = DESCRIPTION_START_RE.search(lines[0])
    if match is None:
        raise ValueError("Cannot find lost particle number")
    lp_no = int(match["lp_no"])
    match = HISTORY_NO_RE.search(lines[0])
    if match is None:
        raise ValueError("Cannot find lost history number")
    hist_no = int(match["hist_no"])
    surface = int(lines[2].split()[-3])
    cell_fail = int(lines[3].rsplit(maxsplit=2)[-1])
    # line[6] may contain one of the following
    # point (x,y,z) is in cell     1439
    # the neutron  is in cell     1344.
    line6 = lines[6].strip()
    point = line6.startswith("point")
    cell_in = int(lines[6].rsplit(maxsplit=2)[-1].rstrip("."))
    x, y, z = map(float, lines[8].split()[-3:])
    u, v, w = map(float, lines[9].split()[-3:])
    return _Description(cell_fail, surface, cell_in, point, x, y, z, u, v, w, lp_no, hist_no)


def _process_file(p: Path, cur: sq.Cursor) -> None:
    with open("lp-details.txt", "w") as fid:
        out_file_name = str(p)
        for lines in extract_descriptions(p):
            print("-" * 20, file=fid)
            for line in lines:
                print(line, file=fid, end="")
            description = _parse_description(lines)
            cur.execute(
                """
                insert into lost_particles (
                    cell_fail,
                    surface,
                    cell_in,
                    x,
                    y,

                    z,
                    u,
                    v,
                    w,
                    lp_no,

                    hist_no,
                    out_file
                )
                values (?,?,?,?,?, ?,?,?,?,?, ?,?)
            """,
                (
                    description.cell_fail,
                    description.surface,
                    description.cell_in,
                    description.x,
                    description.y,
                    description.z,
                    description.u,
                    description.v,
                    description.w,
                    description.lp_no,
                    description.hist_no,
                    out_file_name,
                ),
            )


def main() -> None:
    """Workflow implementation."""
    db = "lost-particles.sqlite"
    _collect_lost_particles(db)
    _analyze(db)


def _analyze(db) -> None:
    with sq.connect(db) as con:
        cur = con.cursor()
        cell_fail_counts = cur.execute(
            """
            select
                cell_fail,
                count(*) cnt
            from lost_particles
            group by cell_fail
            order by cnt desc
            """
        ).fetchall()
        with open("lp-cell-fails-count.csv", "w") as fid:
            for t in cell_fail_counts:
                print(*t, sep=",", file=fid)
        lost_coordinates = cur.execute(
            """
            select
                cell_fail,
                surface,
                cell_in,
                x, y, z,
                u, v, w
            from lost_particles
            order by
                cell_fail,
                surface,
                cell_in,
                x, y, z,
                u, v, w
            """
        ).fetchall()
        with open("lp-coordinates.csv", "w") as fid:
            for t in lost_coordinates:
                print(*t, sep=",", file=fid)

        coordinates_to_work = cur.execute(
            """
                select
                    x, y, z
                from lost_particles
                where
                    cell_fail = ?
                limit 1
            """,
            (cell_fail_counts[0][0],),
        ).fetchone()
        coordinates_text = " ".join(map(str, coordinates_to_work))
        origin_text = "origin " + coordinates_text + " &"
        comin_path = Path("comin")
        if comin_path.exists():
            comin_text = Path("comin").read_text()
            comin_lines = comin_text.split("\n")
            comin_lines[0] = origin_text
            new_comin_text = "\n".join(comin_lines)
        else:
            new_comin_text = origin_text[:-1]
        with open("comin", "w") as fid:
            print(new_comin_text, file=fid)


def _collect_lost_particles(db: str) -> None:
    files = _collect_files()
    with sq.connect(db) as con:
        cur = con.cursor()
        setup_db(cur)
        for f in files:
            _process_file(f, cur)


def _collect_files() -> Iterator[Path]:
    args = sys.argv[1:]
    if args:
        files = cast(Iterator[Path], map(Path, args))
    else:
        files = cast(Iterator[Path], Path.cwd().glob("*.o"))
    return files


if __name__ == "__main__":
    main()
