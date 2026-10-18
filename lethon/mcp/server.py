"""
lethon/mcp/server.py
Model Context Protocol (MCP) Server for Lethon.
Compatible with MCP 2.x and MCP 1.x.
Allows Claude Code, Cursor, Windsurf, Codex, and Google Antigravity
to prune long codebases, error traces, and Chain-of-Thought logs on demand.
"""

import sys
import json
from typing import Optional

try:
    # MCP 2.x standard
    from mcp.server.mcpserver import MCPServer
    mcp = MCPServer(
        name="lethon",
        instructions="Lethon: Spectral-RL Hybrid Context Compression Engine. Cuts token volume by 50%-66% with <0.7% accuracy loss."
    )
except ImportError:
    # MCP 1.x fallback
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP(
        name="lethon",
        instructions="Lethon: Spectral-RL Hybrid Context Compression Engine. Cuts token volume by 50%-66% with <0.7% accuracy loss."
    )

from lethon.core.pipeline import LethonPipeline

pipeline = LethonPipeline()


@mcp.tool()
def prune_context(
    context_text: str,
    compression_ratio: float = 0.5,
    preservation_mode: str = "BALANCED"
) -> str:
    """
    Compresses large source code, documents, or terminal error logs before adding to context window.
    
    Args:
        context_text: The long text, code, or logs to be compressed.
        compression_ratio: Target retention ratio between 0.1 and 0.9 (e.g. 0.4 keeps ~40% tokens, saving 60%). Default is 0.5.
        preservation_mode: 'BALANCED' (default), 'SPECTRAL_FAST', or 'AGGRESSIVE_COT'.
        
    Returns:
        JSON string containing 'compressed_text' and 'metrics' (tokens saved, savings %, latency).
    """
    compressed_text, metrics = pipeline.prune_context(
        context_text=context_text,
        compression_ratio=compression_ratio,
        preservation_mode=preservation_mode
    )
    return json.dumps({
        "compressed_text": compressed_text,
        "metrics": metrics
    }, indent=2)


@mcp.tool()
def compress_cot(
    reasoning_trace: str,
    target_token_limit: Optional[int] = None
) -> str:
    """
    Cleans and condenses Chain-of-Thought (CoT) reasoning traces from models like DeepSeek-R1 / o1 / o3.
    Removes repetitive hesitations and loops while preserving essential logic steps.
    
    Args:
        reasoning_trace: The raw Chain-of-Thought text (e.g. inside <think>...</think>).
        target_token_limit: Optional maximum token ceiling for reasoning budget.
        
    Returns:
        JSON string containing 'compressed_cot' and 'metrics'.
    """
    compressed_cot, metrics = pipeline.compress_cot(
        cot_text=reasoning_trace,
        target_token_limit=target_token_limit
    )
    return json.dumps({
        "compressed_cot": compressed_cot,
        "metrics": metrics
    }, indent=2)


def main():
    """Runs the MCP server over standard stdio transport."""
    mcp.run()


if __name__ == "__main__":
    main()
