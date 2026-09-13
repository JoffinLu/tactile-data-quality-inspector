"""Command-line interface for tactile-qc."""

from __future__ import annotations

import click


@click.group()
@click.version_option(package_name="tactile-qc")
def cli() -> None:
    """tactile-qc — statistical quality assessment for robotic tactile data."""


@cli.command()
@click.argument("path", type=click.Path(exists=True))
def inspect(path: str) -> None:
    """Inspect the RCT data file at PATH."""
    click.echo(f"Inspecting: {path}")
    click.echo("Not implemented yet — arrives in stage 2.")


@cli.command()
def report() -> None:
    """Generate a quality report for a dataset."""
    click.echo("Not implemented yet — arrives in a later stage.")


if __name__ == "__main__":
    cli()
