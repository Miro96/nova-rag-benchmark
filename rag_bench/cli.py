"""CLI entry point for rag-bench."""

import asyncio
import json
import logging
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
@click.option(
    "--replicates",
    default=3,
    type=click.IntRange(min=1),
    help="Number of times to run the benchmark query set; reported metrics are the median across replicates.",
)
@click.option(
    "--clean-index",
    is_flag=True,
    help="Wipe any existing nova-rag-style index directory before ingest for a clean reproducible run.",
)
@click.option("--output", "-o", type=click.Path(), help="Output JSON path")
def run(preset, config, command, transport, repo, ab_baseline, top_k,
        replicates, clean_index, output):
    """Run benchmark on a RAG MCP server."""
    from rag_bench.runner import run_benchmark

    server_config = _resolve_server_config(preset, config, command, transport)

    result = asyncio.run(run_benchmark(
        server_config=server_config,
        repo_filter=repo,
        ab_baseline=ab_baseline,
        top_k=top_k,
        replicates=replicates,
        clean_index=clean_index,
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


@cli.command()
@click.option("--preset", help="Preset name to validate (e.g. nova-rag).")
@click.option("--config", type=click.Path(exists=True), help="Custom server config JSON.")
@click.option("--command", help="Server launch command (auto-detect mode).")
@click.option("--transport", default="stdio", type=click.Choice(["stdio", "sse"]))
@click.option(
    "--config-only",
    is_flag=True,
    help="Only validate preset structure; don't launch the MCP server.",
)
def validate(preset, config, command, transport, config_only):
    """Dry-run a preset: load it, optionally start the server, and report tool compatibility."""
    server_config = _resolve_server_config(preset, config, command, transport)

    click.echo(f"Preset: {server_config.get('name', 'custom')}")
    click.echo(f"Command: {server_config.get('command', '')}")
    click.echo(f"Transport: {server_config.get('transport', 'stdio')}")

    mapping = server_config.get("tool_mapping", {}) or {}
    if mapping:
        click.echo("Configured tool mapping:")
        for slot in ("ingest", "query", "clear", "status"):
            if slot in mapping:
                tool_name = mapping[slot].get("tool", "?")
                params = mapping[slot].get("params", {})
                click.echo(f"  {slot}: {tool_name}  params={params}")
    else:
        click.echo("No preset tool_mapping; will rely on auto-detection.")

    if config_only:
        click.echo("Config-only check passed.")
        return

    asyncio.run(_validate_against_server(server_config))


async def _validate_against_server(server_config):
    from rag_bench.adapter import RAGAdapter
    from rag_bench.mcp_client import create_client

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    client = create_client(server_config)
    async with client:
        adapter = RAGAdapter(client, server_config)
        detected = await adapter.detect_tools()
        click.echo("\nDetected tools:")
        for slot, name in detected.items():
            status = "OK" if name else "MISSING"
            click.echo(f"  {slot}: {name or '(none)'}  [{status}]")

        if not detected.get("ingest"):
            raise click.ClickException("No ingest tool detected on server.")
        if not detected.get("query"):
            raise click.ClickException("No query tool detected on server.")

        click.echo("\nResolved param mappings:")
        click.echo(f"  ingest params: {adapter._ingest_params}")
        click.echo(f"  query params:  {adapter._query_params}")
        click.echo("\nValidation passed.")


def _resolve_server_config(preset, config, command, transport):
    if not (preset or config or command):
        raise click.ClickException("Provide --preset, --config, or --command")
    if config:
        return json.loads(Path(config).read_text())
    if preset:
        loaded = _load_preset(preset)
        if loaded is None:
            available = sorted(_available_presets())
            raise click.ClickException(
                f"Preset '{preset}' not found. Available presets: "
                + (", ".join(available) if available else "(none)")
            )
        return loaded
    return {
        "name": "custom",
        "command": command,
        "transport": transport,
    }


def _preset_dir() -> Path:
    return Path(__file__).parent / "presets"


def _available_presets() -> list[str]:
    return [p.stem.replace("_", "-") for p in _preset_dir().glob("*.json")]


def _load_preset(name):
    preset_dir = _preset_dir()
    path = preset_dir / f"{name.replace('-', '_')}.json"
    if not path.exists():
        path = preset_dir / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


if __name__ == "__main__":
    cli()
