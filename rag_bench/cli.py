"""CLI entry point for rag-bench."""

import asyncio
import json
from pathlib import Path

import click

from rag_bench import __version__


@click.group()
@click.version_option(__version__)
def cli():
    """rag-bench — Benchmark for Code RAG MCP servers."""


@cli.command()
@click.option("--preset", help="Preset name (e.g. nova-rag, mcp-local-rag)")
@click.option("--config", type=click.Path(exists=True), help="Custom server config JSON")
@click.option("--command", help="Server launch command (auto-detect mode)")
@click.option("--transport", default="stdio", type=click.Choice(["stdio", "sse"]))
@click.option("--repo", help="Run only on specific repo (flask, fastapi, express)")
@click.option("--ab-baseline", is_flag=True, help="Run A/B comparison vs Grep/Glob baseline")
@click.option("--top-k", default=10, help="Number of results to request from RAG")
@click.option("--output", "-o", type=click.Path(), help="Output JSON path")
def run(preset, config, command, transport, repo, ab_baseline, top_k, output):
    """Run benchmark on a RAG MCP server."""
    from rag_bench.runner import run_benchmark

    server_config = _resolve_server_config(preset, config, command, transport)
    if not server_config:
        raise click.ClickException("Provide --preset, --config, or --command")

    result = asyncio.run(run_benchmark(
        server_config=server_config,
        repo_filter=repo,
        ab_baseline=ab_baseline,
        top_k=top_k,
    ))

    output_path = Path(output) if output else Path(f"results/run_{result['run_id']}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    click.echo(f"\nResults saved to {output_path}")


@cli.command()
@click.option("--presets", required=True, help="Comma-separated preset names")
@click.option("--repo", help="Run only on specific repo")
@click.option("--top-k", default=10)
def compare(presets, repo, top_k):
    """Compare multiple RAG MCP servers."""
    from rag_bench.runner import run_benchmark

    preset_names = [p.strip() for p in presets.split(",")]
    results = []
    for name in preset_names:
        server_config = _load_preset(name)
        if not server_config:
            click.echo(f"Preset '{name}' not found, skipping")
            continue
        result = asyncio.run(run_benchmark(
            server_config=server_config,
            repo_filter=repo,
            top_k=top_k,
        ))
        results.append(result)

    if results:
        from rag_bench.report import print_comparison_table
        print_comparison_table(results)


@cli.command()
@click.argument("result_files", nargs=-1, type=click.Path(exists=True))
@click.option("--server-url", default="http://localhost:8080", help="Leaderboard server URL")
@click.option("--git-url", required=True, help="Git repository URL")
@click.option("--git-user", required=True, help="Git username")
def submit(result_files, server_url, git_url, git_user):
    """Submit benchmark results to leaderboard server."""
    from rag_bench.submit import submit_results

    for path in result_files:
        data = json.loads(Path(path).read_text())
        data.setdefault("server", {})["git_url"] = git_url
        data["server"]["git_user"] = git_user
        asyncio.run(submit_results(server_url, data))
        click.echo(f"Submitted {path}")


@cli.command()
@click.option("--port", default=8080)
@click.option("--host", default="0.0.0.0")
def serve(port, host):
    """Start the leaderboard web server."""
    import uvicorn
    from server.app import app

    uvicorn.run(app, host=host, port=port)


def _resolve_server_config(preset, config, command, transport):
    if config:
        return json.loads(Path(config).read_text())
    if preset:
        return _load_preset(preset)
    if command:
        return {
            "name": "custom",
            "command": command,
            "transport": transport,
        }
    return None


def _load_preset(name):
    preset_dir = Path(__file__).parent / "presets"
    path = preset_dir / f"{name.replace('-', '_')}.json"
    if not path.exists():
        # try with dashes
        path = preset_dir / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


if __name__ == "__main__":
    cli()
