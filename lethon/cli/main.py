"""
lethon/cli/main.py
Command-Line Interface (CLI) for Lethon.
Specialized commands for AI coding workflows:
- `lethon mcp` -> Starts MCP server for Claude Code / Cursor / Antigravity
- `lethon code <file>` -> Prunes source code
- `lethon logs <file>` -> Prunes logs and extracts error traces
- `lethon diff <file>` -> Cleans git diffs
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
    """Lethon: Spectral Context Compression Engine for AI Coding Agents."""
    pass


@cli.command(name="mcp")
def start_mcp():
    """Starts the Model Context Protocol (MCP) server over stdio."""
    run_mcp_server()


@cli.command(name="code")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--ratio", "-r", default=0.5, type=float, help="Retention ratio (default: 0.5).")
@click.option("--output", "-o", type=click.Path(dir_okay=False), default=None, help="Output file path.")
@click.option("--json-metrics", is_flag=True, default=False, help="Print metrics as JSON.")
def prune_code_cmd(file_path: str, ratio: float, output: str, json_metrics: bool):
    """Prunes large source code files while protecting syntax and signatures."""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    compressed, metrics = pipeline.prune_code(text, compression_ratio=ratio)
    
    if output:
        Path(output).write_text(compressed, encoding="utf-8")
        click.echo(f"Saved: {output}")
    else:
        click.echo(compressed)
    
    if json_metrics:
        click.echo(json.dumps(metrics))
    else:
        click.echo(f"--- Tokens: {metrics['orig_tokens']} -> {metrics['compressed_tokens']} ({metrics['savings_pct']}% savings in {metrics['latency_ms']} ms) ---", err=True)


@cli.command(name="logs")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--context", "-c", default=3, type=int, help="Context lines around errors.")
@click.option("--output", "-o", type=click.Path(dir_okay=False), default=None, help="Output file path.")
@click.option("--json-metrics", is_flag=True, default=False, help="Print metrics as JSON.")
def prune_logs_cmd(file_path: str, context: int, output: str, json_metrics: bool):
    """Prunes terminal logs and retains error stack traces."""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    compressed, metrics = pipeline.prune_logs(text, context_window_lines=context)
    
    if output:
        Path(output).write_text(compressed, encoding="utf-8")
        click.echo(f"Saved: {output}")
    else:
        click.echo(compressed)
        
    if json_metrics:
        click.echo(json.dumps(metrics))
    else:
        click.echo(f"--- Tokens: {metrics['orig_tokens']} -> {metrics['compressed_tokens']} ({metrics['savings_pct']}% savings in {metrics['latency_ms']} ms) ---", err=True)


@cli.command(name="diff")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", type=click.Path(dir_okay=False), default=None, help="Output file path.")
@click.option("--json-metrics", is_flag=True, default=False, help="Print metrics as JSON.")
def prune_diff_cmd(file_path: str, output: str, json_metrics: bool):
    """Prunes git diff files by removing lockfiles and binary assets."""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    compressed, metrics = pipeline.prune_diff(text)
    
    if output:
        Path(output).write_text(compressed, encoding="utf-8")
        click.echo(f"Saved: {output}")
    else:
        click.echo(compressed)
        
    if json_metrics:
        click.echo(json.dumps(metrics))
    else:
        click.echo(f"--- Tokens: {metrics['orig_tokens']} -> {metrics['compressed_tokens']} ({metrics['savings_pct']}% savings in {metrics['latency_ms']} ms) ---", err=True)


if __name__ == "__main__":
    cli()
