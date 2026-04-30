"""Typer-based CLI for microGPT name generation."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from microgpt_name_gen.main import MicroGPTConfig, run_microgpt

app = typer.Typer(
    help="Train a minimal character-level GPT on a name corpus and sample new names.",
    invoke_without_command=True,
    rich_markup_mode="rich",
)
console = Console()


@app.callback()
def _train(
    ctx: typer.Context,
    data_path: Path = typer.Option(
        Path("input.txt"),
        "--data",
        "-d",
        help="Path to the training corpus (one name per line).",
        exists=False,
    ),
    no_download: bool = typer.Option(
        False,
        "--no-download",
        help="Do not download the default Karpathy names corpus if data file is missing.",
    ),
    steps: int = typer.Option(
        1000, "--steps", "-s", help="Number of training steps.", min=1
    ),
    samples: int = typer.Option(
        20, "--samples", "-n", help="Number of names to generate after training.", min=1
    ),
    temperature: float = typer.Option(
        0.5,
        "--temperature",
        "-t",
        help='Sampling "creativity" in (0, 1]; lower is more conservative.',
        min=0.01,
        max=1.0,
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    learning_rate: float = typer.Option(0.01, "--lr", help="Base learning rate."),
    n_layer: int = typer.Option(1, "--layers", help="Transformer depth.", min=1),
    n_embd: int = typer.Option(16, "--embd", help="Embedding width.", min=1),
    block_size: int = typer.Option(16, "--block", help="Context length.", min=1),
    n_head: int = typer.Option(4, "--heads", help="Attention heads.", min=1),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="No progress bar; minimal output."
    ),
) -> None:
    """Train on [bold]data_path[/bold] and print generated names."""
    if ctx.invoked_subcommand is not None:
        return

    config = MicroGPTConfig(
        seed=seed,
        data_path=data_path,
        download_if_missing=not no_download,
        num_steps=steps,
        num_samples=samples,
        temperature=temperature,
        learning_rate=learning_rate,
        n_layer=n_layer,
        n_embd=n_embd,
        block_size=block_size,
        n_head=n_head,
    )

    if n_embd % n_head != 0:
        console.print(
            "[red]Error:[/red] --embd must be divisible by --heads "
            f"(got {n_embd} and {n_head})."
        )
        raise typer.Exit(code=1)

    if not quiet:
        info = Table.grid(padding=(0, 2))
        info.add_column(style="bold cyan", justify="right")
        info.add_column()
        info.add_row("Data", str(config.data_path))
        info.add_row("Steps", str(config.num_steps))
        info.add_row("Samples", str(config.num_samples))
        info.add_row("Temperature", str(config.temperature))
        info.add_row("Seed", str(config.seed))
        console.print(Panel(info, title="Run configuration", border_style="blue"))

    if quiet:
        generated = run_microgpt(config)
    else:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Training", total=config.num_steps)

            def on_step(step: int, _total: int, loss: float) -> None:
                progress.update(
                    task_id,
                    completed=step,
                    description=f"[cyan]loss[/] {loss:.4f}",
                )

            generated = run_microgpt(config, on_step=on_step)

    table = Table(title="Generated names", show_header=False, border_style="green")
    for i, name in enumerate(generated, start=1):
        table.add_row(str(i), name)
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
