#!python
"""Try typer CLI."""
from __future__ import annotations

from typing import Annotated, Optional

import typer

from rich.console import Console

app = typer.Typer(rich_markup_mode="rich")
console = Console(stderr=True)

__version__ = "0.1.0"


def version_callback(value: bool):
    if value:
        print(f"Awesome CLI Version: {__version__}")
        raise typer.Exit()


@app.command()
def main(
    name: Annotated[str, typer.Option()] = "World",
    version: Annotated[Optional[bool], typer.Option("--version", callback=version_callback)] = None,
):
    print(f"Hello {name}")


if __name__ == "__main__":
    typer.run(main)

if __name__ == "__main__":
    app()
