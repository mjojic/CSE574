"""Command-line interface for PyDPOCL."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from pydpocl import __version__
from pydpocl.domain.compiler import compile_domain_and_problem
from pydpocl.planning.llm_policy import DEFAULT_MODEL, LLMConfig
from pydpocl.planning.planner import DPOCLPlanner

console = Console()


@click.group()
@click.version_option(version=__version__)
@click.option(
    "--verbose", "-v", is_flag=True, help="Enable verbose output"
)
@click.option(
    "--quiet", "-q", is_flag=True, help="Suppress non-essential output"
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool) -> None:
    """PyDPOCL - A modern Python implementation of Decompositional Partial Order Causal-Link Planning."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet

    if not quiet:
        console.print(f"[bold blue]PyDPOCL {__version__}[/bold blue]")


@cli.command()
@click.argument("domain_file", type=click.Path(exists=True, path_type=Path))
@click.argument("problem_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--max-solutions", "-k", default=1, help="Maximum number of solutions to find"
)
@click.option(
    "--timeout", "-t", default=300.0, help="Timeout in seconds"
)
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), help="Output file for solutions"
)
@click.option(
    "--heuristic", default="add",
    type=click.Choice(["oc", "fc", "tc", "ps", "add", "max", "ff", "llm"]),
    help=(
        "POCL heuristic — structural: 'oc' (open conditions), 'fc' (flaws+threats),"
        " 'tc' (threats only), 'ps' (plan-size penalty);"
        " state-based: 'add' (h_add, default), 'max' (h_max, admissible),"
        " 'ff' (h_FF, strongest); 'llm' (LLM picks next frontier node)"
    )
)
@click.option(
    "--flaw-selection-strat", default="lcfr",
    type=click.Choice(["lcfr", "zlifo", "lifo", "fifo", "random", "llm"]),
    help=(
        "Open-condition flaw ordering: 'lcfr' (least-constrained-first / fail-first),"
        " 'zlifo', 'lifo' (most-recently-introduced), 'fifo' (oldest-first),"
        " 'random', or 'llm' (delegate flaw choice to the LLM)"
    )
)
@click.option(
    "--resolver-strat", default="enumerate",
    type=click.Choice(["enumerate", "llm"]),
    help=(
        "'enumerate' pushes every consistent resolver successor onto the frontier."
        " 'llm' picks exactly one resolver per expansion and reprompts on dead ends."
    )
)
@click.option(
    "--llm-model", default=None,
    help=(
        "Model identifier (default: $OPENAI_MODEL / $LLM_MODEL or"
        f" {DEFAULT_MODEL}). For vLLM, pass the model id served by the"
        " endpoint, e.g. 'Qwen/Qwen3-32B-FP8'."
    )
)
@click.option(
    "--llm-base-url", default=None,
    help=(
        "OpenAI-compatible base URL (default: $OPENAI_BASE_URL / $LLM_BASE_URL)."
        " For vLLM use e.g. 'http://localhost:8000/v1'."
    )
)
@click.option(
    "--llm-api-key", default=None,
    help=(
        "Override $OPENAI_API_KEY. Local servers usually ignore this; a"
        " placeholder is sent automatically when --llm-base-url is set"
        " without a key."
    )
)
@click.option(
    "--llm-response-format",
    type=click.Choice(["json_schema", "json_object", "none"]),
    default="json_schema",
    help=(
        "How to constrain LLM outputs: 'json_schema' (strict OpenAI-style"
        " structured outputs), 'json_object' (looser; works on most vLLM"
        " builds), or 'none' (rely on the prompt). The policy auto-falls"
        " back to 'json_object' if the server rejects 'json_schema'."
    )
)
@click.pass_context
def solve(
    ctx: click.Context,
    domain_file: Path,
    problem_file: Path,
    max_solutions: int,
    timeout: float,
    output: Path | None,
    heuristic: str,
    flaw_selection_strat: str,
    resolver_strat: str,
    llm_model: str | None,
    llm_base_url: str | None,
    llm_api_key: str | None,
    llm_response_format: str,
) -> None:
    """Solve a planning problem."""
    verbose = ctx.obj["verbose"]
    quiet = ctx.obj["quiet"]

    needs_llm = (
        heuristic == "llm"
        or flaw_selection_strat == "llm"
        or resolver_strat == "llm"
    )

    try:
        if not quiet:
            console.print(f"Domain: [cyan]{domain_file}[/cyan]")
            console.print(f"Problem: [cyan]{problem_file}[/cyan]")
            console.print(f"Heuristic: [yellow]{heuristic}[/yellow]")
            console.print(f"Flaw selection: [yellow]{flaw_selection_strat}[/yellow]")
            console.print(f"Resolver strategy: [yellow]{resolver_strat}[/yellow]")
            if needs_llm:
                console.print(
                    "LLM model: [cyan]"
                    + (llm_model or "env/default")
                    + "[/cyan]"
                    + (f"  base_url: [cyan]{llm_base_url}[/cyan]" if llm_base_url else "")
                )
            console.print()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=quiet,
        ) as progress:
            task = progress.add_task("Compiling domain and problem...", total=None)

            try:
                problem = compile_domain_and_problem(domain_file, problem_file)
                progress.update(
                    task,
                    description=f"Compiled {len(problem.operators)} ground steps",
                )
            except Exception as e:
                console.print(f"[red]Error compiling domain/problem:[/red] {e}")
                sys.exit(1)

        llm_config: LLMConfig | None = None
        if needs_llm:
            llm_config = LLMConfig(
                response_format=llm_response_format,
            )
            if llm_model:
                llm_config.model = llm_model
            if llm_base_url:
                llm_config.base_url = llm_base_url
            if llm_api_key:
                llm_config.api_key = llm_api_key

        planner = DPOCLPlanner(
            heuristic=heuristic,
            verbose=verbose,
            flaw_selection_strat=flaw_selection_strat,
            resolver_strat=resolver_strat,
            llm_config=llm_config,
        )

        if not quiet:
            console.print(f"Searching for up to {max_solutions} solutions...")
            console.print()

        solutions = []
        stats = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=quiet,
        ) as progress:
            task = progress.add_task("Planning...", total=None)

            try:
                solutions = list(
                    planner.solve(problem, max_solutions=max_solutions, timeout=timeout)
                )
                stats = planner.get_statistics()

                progress.update(
                    task,
                    description=f"Found {len(solutions)} solution(s)",
                )

            except Exception as e:
                console.print(f"[red]Planning error:[/red] {e}")
                sys.exit(1)

        if solutions:
            if not quiet:
                console.print(f"[green]Found {len(solutions)} solution(s)![/green]")
                console.print()

            for i, solution in enumerate(solutions, 1):
                if not quiet:
                    console.print(f"[bold]Solution {i}:[/bold]")

                    steps = solution.to_execution_sequence()
                    if steps:
                        table = Table(title=f"Solution {i} Steps")
                        table.add_column("Step", style="cyan")
                        table.add_column("Action", style="white")

                        for j, step in enumerate(steps, 1):
                            table.add_row(str(j), step.signature)

                        console.print(table)
                    else:
                        console.print("Empty plan (no actions needed)")

                    console.print()

            if verbose and stats:
                stats_table = Table(title="Planning Statistics")
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Value", style="white")

                for key, value in stats.items():
                    stats_table.add_row(key.replace("_", " ").title(), str(value))

                console.print(stats_table)

            if output:
                try:
                    with open(output, "w") as f:
                        f.write(f"# PyDPOCL Solutions for {problem_file.name}\n\n")
                        for i, solution in enumerate(solutions, 1):
                            f.write(f"## Solution {i}\n")
                            steps = solution.to_execution_sequence()
                            for j, step in enumerate(steps, 1):
                                f.write(f"{j}. {step.signature}\n")
                            f.write("\n")

                    console.print(f"Solutions saved to [cyan]{output}[/cyan]")

                except Exception as e:
                    console.print(f"[red]Error saving output:[/red] {e}")

        else:
            console.print("[yellow]No solutions found within the time limit.[/yellow]")
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Planning interrupted by user.[/yellow]")
        sys.exit(1)


@cli.command()
@click.argument("domain_file", type=click.Path(exists=True, path_type=Path))
@click.argument("problem_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def validate(
    ctx: click.Context,
    domain_file: Path,
    problem_file: Path,
) -> None:
    """Validate a domain and problem file."""
    quiet = ctx.obj["quiet"]

    try:
        if not quiet:
            console.print("Validating domain and problem files...")

        try:
            problem = compile_domain_and_problem(domain_file, problem_file)

            if not quiet:
                console.print(
                    f"[green]✓[/green] Successfully compiled "
                    f"{len(problem.operators)} ground steps"
                )
                console.print("[green]✓[/green] Domain and problem files are valid")

        except Exception as e:
            console.print(f"[red]✗[/red] Validation failed: {e}")
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]Error during validation:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("domain_file", type=click.Path(exists=True, path_type=Path))
@click.argument("problem_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), help="Output file for ground steps"
)
@click.pass_context
def compile(
    ctx: click.Context,
    domain_file: Path,
    problem_file: Path,
    output: Path | None,
) -> None:
    """Compile domain and problem into ground steps."""
    quiet = ctx.obj["quiet"]

    try:
        if not quiet:
            console.print("Compiling domain and problem...")

        problem = compile_domain_and_problem(domain_file, problem_file)
        ground_steps = problem.operators

        if not quiet:
            console.print(f"[green]Successfully compiled {len(ground_steps)} ground steps[/green]")

        if output:
            try:
                with open(output, "w") as f:
                    f.write(f"# Ground steps for {problem_file.name}\n")
                    f.write(f"# Total: {len(ground_steps)} steps\n\n")

                    for i, step in enumerate(ground_steps):
                        f.write(f"Step {i}: {step.signature}\n")
                        f.write(f"  Preconditions: {list(step.preconditions)}\n")
                        f.write(f"  Effects: {list(step.effects)}\n")
                        f.write("\n")

                console.print(f"Ground steps saved to [cyan]{output}[/cyan]")

            except Exception as e:
                console.print(f"[red]Error saving output:[/red] {e}")

        elif not quiet:
            table = Table(title="Ground Steps Summary")
            table.add_column("Step", style="cyan")
            table.add_column("Operator", style="white")
            table.add_column("Parameters", style="yellow")

            for i, step in enumerate(ground_steps[:10]):
                table.add_row(
                    str(i),
                    str(step.name),
                    ", ".join(step.parameters) if step.parameters else "None"
                )

            if len(ground_steps) > 10:
                table.add_row("...", "...", "...")

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error during compilation:[/red] {e}")
        sys.exit(1)


@cli.command()
def examples() -> None:
    """Show usage examples."""
    console.print("[bold]PyDPOCL Usage Examples[/bold]\n")

    examples_list = [
        ("Basic planning (h_add, lcfr)", "pydpocl solve domain.pddl problem.pddl"),
        ("Find multiple solutions", "pydpocl solve domain.pddl problem.pddl -k 5"),
        ("Use h_FF heuristic", "pydpocl solve domain.pddl problem.pddl --heuristic ff"),
        ("Use h_max heuristic (admissible)", "pydpocl solve domain.pddl problem.pddl --heuristic max"),
        ("Open-condition count heuristic + zlifo", "pydpocl solve domain.pddl problem.pddl --heuristic oc --flaw-selection-strat zlifo"),
        ("LLM node selection only", "pydpocl solve domain.pddl problem.pddl --heuristic llm"),
        ("LLM flaw selection only", "pydpocl solve domain.pddl problem.pddl --flaw-selection-strat llm"),
        ("Full LLM control", "pydpocl solve domain.pddl problem.pddl --heuristic llm --flaw-selection-strat llm --resolver-strat llm"),
        ("Save solutions to file", "pydpocl solve domain.pddl problem.pddl -o solutions.txt"),
        ("Validate domain/problem", "pydpocl validate domain.pddl problem.pddl"),
        ("Compile to ground steps", "pydpocl compile domain.pddl problem.pddl -o ground_steps.txt"),
    ]

    for description, command in examples_list:
        console.print(f"[cyan]{description}:[/cyan]")
        console.print(f"  {command}")
        console.print()


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
