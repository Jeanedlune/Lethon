"""
lethon/cli/main.py
Command-Line Interface (CLI) for Lethon.
Supports:
- `lethon mcp` -> Launches the MCP server for Claude Code / Cursor / Antigravity
- `lethon prune <file>` -> Prunes a specific file or text input
- `lethon cot <file>` -> Compresses a reasoning trace
"""

import sys
import json
from pathlib import Path
import click

from lethon.core.pipeline import LethonPipeline
from lethon.mcp.server import main as run_mcp_server

pipeline = LethonPipeline()


@click.group()
@click.version_option(version="0.1.0", prog_name="lethon")
def cli():
    """Lethon: Spectral-RL Hybrid Context Compression Engine."""
    pass


@cli.command(name="mcp")
def start_mcp():
    """Starts the Model Context Protocol (MCP) server over stdio."""
    run_mcp_server()


@cli.command(name="prune")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--ratio", "-r", default=0.5, type=float, help="Retention ratio (e.g. 0.4 keeps ~40% tokens).")
@click.option("--mode", "-m", default="BALANCED", type=click.Choice(["BALANCED", "SPECTRAL_FAST", "AGGRESSIVE_COT"]), help="Compression preservation mode.")
@click.option("--output", "-o", type=click.Path(dir_okay=False), default=None, help="Output file path (default: stdout).")
@click.option("--json-metrics", is_flag=True, default=False, help="Print metrics as raw JSON.")
def prune_file(file_path: str, ratio: float, mode: str, output: str, json_metrics: bool):
    """Prunes a codebase, log file, or document to save tokens."""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    
    compressed_text, metrics = pipeline.prune_context(
        context_text=text,
        compression_ratio=ratio,
        preservation_mode=mode
    )

    if output:
        Path(output).write_text(compressed_text, encoding="utf-8")
        if json_metrics:
            click.echo(json.dumps(metrics))
        else:
            click.echo(f"Output written to: {output}")
            click.echo(f"Original tokens: {metrics['orig_tokens']}")
            click.echo(f"Compressed tokens: {metrics['compressed_tokens']}")
            click.echo(f"Savings: {metrics['savings_pct']}%")
            click.echo(f"Latency: {metrics['latency_ms']} ms")
    else:
        if json_metrics:
            click.echo(json.dumps({"compressed_text": compressed_text, "metrics": metrics}))
        else:
            click.echo(compressed_text)
            click.echo(f"--- Metrics: tokens {metrics['orig_tokens']} -> {metrics['compressed_tokens']} ({metrics['savings_pct']}% savings in {metrics['latency_ms']} ms) ---", err=True)


@cli.command(name="cot")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--limit", "-l", default=None, type=int, help="Optional maximum token budget.")
@click.option("--output", "-o", type=click.Path(dir_okay=False), default=None, help="Output file path.")
@click.option("--json-metrics", is_flag=True, default=False, help="Print metrics as raw JSON.")
def prune_cot(file_path: str, limit: int, output: str, json_metrics: bool):
    """Compresses a Chain-of-Thought (CoT) reasoning trace."""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    
    compressed_cot, metrics = pipeline.compress_cot(
        cot_text=text,
        target_token_limit=limit
    )

    if output:
        Path(output).write_text(compressed_cot, encoding="utf-8")
        if json_metrics:
            click.echo(json.dumps(metrics))
        else:
            click.echo(f"Output written to: {output}")
            click.echo(f"Original tokens: {metrics['orig_tokens']}")
            click.echo(f"Compressed tokens: {metrics['compressed_tokens']}")
            click.echo(f"Savings: {metrics['savings_pct']}%")
            click.echo(f"Latency: {metrics['latency_ms']} ms")
    else:
        if json_metrics:
            click.echo(json.dumps({"compressed_cot": compressed_cot, "metrics": metrics}))
        else:
            click.echo(compressed_cot)
            click.echo(f"--- Metrics: tokens {metrics['orig_tokens']} -> {metrics['compressed_tokens']} ({metrics['savings_pct']}% savings in {metrics['latency_ms']} ms) ---", err=True)


if __name__ == "__main__":
    cli()
