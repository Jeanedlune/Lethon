"""
lethon/mcp/server.py
Model Context Protocol (MCP) Server for Lethon.
Exposes specialized context pruning tools for AI Coding Agents:
- prune_code: Prunes large source code files while preserving syntax & function signatures
- prune_logs: Cuts repetitive logs and isolates root-cause stack traces
- prune_diff: Cleans git diffs by discarding auto-generated lockfiles and assets
- prune_context: Universal router
"""

import json
from typing import Optional

try:
    from mcp.server.mcpserver import MCPServer
    mcp = MCPServer(
        name="lethon",
        instructions="Lethon: Spectral Context Compression Engine for AI Coding Agents. Prunes large code, logs, and git diffs."
    )
except ImportError:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP(
        name="lethon",
        instructions="Lethon: Spectral Context Compression Engine for AI Coding Agents. Prunes large code, logs, and git diffs."
    )

from lethon.core.pipeline import LethonPipeline

pipeline = LethonPipeline()


@mcp.tool()
def prune_code(
    code_text: str,
    compression_ratio: float = 0.5
) -> str:
    """
    Compresses large source code files before sending to LLM context window.
    Preserves all function signatures, classes, imports, and syntax invariants.
    
    Args:
        code_text: The full source code string.
        compression_ratio: Target retention ratio between 0.1 and 0.9 (default: 0.5, saves ~50% tokens).
        
    Returns:
        JSON string containing 'compressed_code' and 'metrics'.
    """
    compressed_text, metrics = pipeline.prune_code(
        code_text=code_text,
        compression_ratio=compression_ratio
    )
    return json.dumps({
        "compressed_code": compressed_text,
        "metrics": metrics
    }, indent=2)


@mcp.tool()
def prune_logs(
    log_text: str,
    context_lines: int = 3
) -> str:
    """
    Compresses massive terminal or build logs (e.g. cargo build, npm test, pytest).
    Truncates repetitive INFO/DEBUG messages and retains exact error stack traces.
    
    Args:
        log_text: Raw terminal log string.
        context_lines: Number of surrounding lines to keep around each error (default: 3).
        
    Returns:
        JSON string containing 'compressed_logs' and 'metrics'.
    """
    compressed_text, metrics = pipeline.prune_logs(
        log_text=log_text,
        context_window_lines=context_lines
    )
    return json.dumps({
        "compressed_logs": compressed_text,
        "metrics": metrics
    }, indent=2)


@mcp.tool()
def prune_diff(
    diff_text: str
) -> str:
    """
    Compresses large git diffs by stripping noisy lockfiles (package-lock, Cargo.lock)
    and compiled assets, keeping only core logic changes.
    
    Args:
        diff_text: The output from `git diff`.
        
    Returns:
        JSON string containing 'compressed_diff' and 'metrics'.
    """
    compressed_text, metrics = pipeline.prune_diff(diff_text=diff_text)
    return json.dumps({
        "compressed_diff": compressed_text,
        "metrics": metrics
    }, indent=2)


@mcp.tool()
def prune_context(
    context_text: str,
    compression_ratio: float = 0.5
) -> str:
    """
    Universal context pruner. Automatically routes to code or log pruning.
    
    Args:
        context_text: Text content to compress.
        compression_ratio: Target retention ratio.
        
    Returns:
        JSON string containing 'compressed_text' and 'metrics'.
    """
    compressed_text, metrics = pipeline.prune_context(
        context_text=context_text,
        compression_ratio=compression_ratio
    )
    return json.dumps({
        "compressed_text": compressed_text,
        "metrics": metrics
    }, indent=2)


def main():
    """Runs the MCP server over standard stdio transport."""
    mcp.run()


if __name__ == "__main__":
    main()
