"""Report generation: terminal tables and JSON output."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag_bench.metrics import BenchmarkMetrics

console = Console()


def print_results_table(server_name: str, m: BenchmarkMetrics) -> None:
    """Print benchmark results as a rich table."""
    console.print()
    console.print(Panel(f"[bold]rag-bench results: {server_name}[/bold]"))

    # Main metrics
    table = Table(title="Retrieval Quality", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Hit@1", f"{m.hit_at_1:.1%}")
    table.add_row("Hit@3", f"{m.hit_at_3:.1%}")
    table.add_row("Hit@5", f"{m.hit_at_5:.1%}")
    table.add_row("Hit@10", f"{m.hit_at_10:.1%}")
    table.add_row("Symbol Hit@5", f"{m.symbol_hit_at_5:.1%}")
    table.add_row("MRR", f"{m.mrr:.4f}")
    console.print(table)

    # Latency
    lat_table = Table(title="Latency", show_header=True)
    lat_table.add_column("Metric", style="cyan")
    lat_table.add_column("Value", style="green", justify="right")

    lat_table.add_row("p50", f"{m.query_latency_p50_ms:.0f} ms")
    lat_table.add_row("p95", f"{m.query_latency_p95_ms:.0f} ms")
    lat_table.add_row("p99", f"{m.query_latency_p99_ms:.0f} ms")
    lat_table.add_row("mean", f"{m.query_latency_mean_ms:.0f} ms")
    console.print(lat_table)

    # Ingest
    ing_table = Table(title="Ingest", show_header=True)
    ing_table.add_column("Metric", style="cyan")
    ing_table.add_column("Value", style="green", justify="right")

    ing_table.add_row("Files", str(m.ingest_total_files))
    ing_table.add_row("Time", f"{m.ingest_total_sec:.1f} s")
    ing_table.add_row("Speed", f"{m.ingest_files_per_sec:.1f} files/s")
    ing_table.add_row("Index Size", f"{m.index_size_mb:.1f} MB")
    ing_table.add_row("RAM Peak", f"{m.ram_peak_mb:.1f} MB")
    console.print(ing_table)

    # Breakdown by difficulty
    if m.by_difficulty:
        diff_table = Table(title="By Difficulty", show_header=True)
        diff_table.add_column("Difficulty", style="cyan")
        diff_table.add_column("Count", justify="right")
        diff_table.add_column("Hit@5", justify="right")
        diff_table.add_column("MRR", justify="right")
        diff_table.add_column("p50 ms", justify="right")

        for diff in ["easy", "medium", "hard"]:
            if diff in m.by_difficulty:
                d = m.by_difficulty[diff]
                diff_table.add_row(
                    diff,
                    str(d["count"]),
                    f"{d['hit_at_5']:.1%}",
                    f"{d['mrr']:.4f}",
                    f"{d['latency_p50_ms']:.0f}",
                )
        console.print(diff_table)

    # Breakdown by type
    if m.by_type:
        type_table = Table(title="By Query Type", show_header=True)
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Count", justify="right")
        type_table.add_column("Hit@5", justify="right")
        type_table.add_column("MRR", justify="right")

        for qtype in ["locate", "callers", "explain", "impact"]:
            if qtype in m.by_type:
                d = m.by_type[qtype]
                type_table.add_row(
                    qtype,
                    str(d["count"]),
                    f"{d['hit_at_5']:.1%}",
                    f"{d['mrr']:.4f}",
                )
        console.print(type_table)

    # Composite score
    console.print(
        f"\n[bold]Composite Score: [green]{m.composite_score:.4f}[/green][/bold]\n"
    )


def print_comparison_table(results: list[dict]) -> None:
    """Print side-by-side comparison of multiple benchmark results."""
    table = Table(title="RAG Server Comparison", show_header=True)
    table.add_column("Metric", style="cyan")

    for r in results:
        table.add_column(r["server"]["name"], style="green", justify="right")

    rows = [
        ("Hit@1", lambda r: f"{r['retrieval']['hit_at_1']:.1%}"),
        ("Hit@5", lambda r: f"{r['retrieval']['hit_at_5']:.1%}"),
        ("Symbol Hit@5", lambda r: f"{r['retrieval']['symbol_hit_at_5']:.1%}"),
        ("MRR", lambda r: f"{r['retrieval']['mrr']:.4f}"),
        ("Latency p50", lambda r: f"{r['retrieval']['latency']['p50_ms']:.0f} ms"),
        ("Latency p95", lambda r: f"{r['retrieval']['latency']['p95_ms']:.0f} ms"),
        ("Ingest Time", lambda r: f"{r['ingest']['total_sec']:.1f} s"),
        ("Ingest Speed", lambda r: f"{r['ingest']['files_per_sec']:.1f} f/s"),
        ("RAM Peak", lambda r: f"{r['ingest']['ram_peak_mb']:.1f} MB"),
        ("Score", lambda r: f"{r['composite_score']:.4f}"),
    ]

    for label, fn in rows:
        values = []
        for r in results:
            try:
                values.append(fn(r))
            except (KeyError, TypeError):
                values.append("N/A")
        table.add_row(label, *values)

    console.print(table)
