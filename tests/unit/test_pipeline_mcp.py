"""
Unit tests for LethonPipeline, MCP tools, and CLI endpoints.
"""

import json
import pytest
from lethon.core.pipeline import LethonPipeline
from lethon.mcp.server import prune_context, compress_cot


@pytest.fixture
def pipeline():
    return LethonPipeline()


def test_pipeline_prune_context_code(pipeline):
    code = """
import os
import sys

class DataProcessor:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.active = True

    def process(self, data):
        # some repetitive processing lines
        val = data * 2
        val = val + 1
        val = val - 1
        val = val * 2
        val = val + 1
        val = val - 1
        val = val * 2
        val = val + 1
        val = val - 1
        return val

    def cleanup(self):
        self.count = 0
        self.active = False
""" * 5
    compressed, metrics = pipeline.prune_context(code, compression_ratio=0.4)
    
    assert metrics["orig_tokens"] > metrics["compressed_tokens"]
    assert metrics["savings_pct"] > 0.0
    assert metrics["latency_ms"] < 25.0
    # Must preserve essential anchor keywords
    assert "class DataProcessor:" in compressed
    assert "def process(self, data):" in compressed
    assert "import os" in compressed


def test_pipeline_prune_logs_with_errors(pipeline):
    logs = """
[2026-10-18 10:00:01] [INFO] Server started on port 8080
[2026-10-18 10:00:02] [INFO] Healthcheck ok
[2026-10-18 10:00:03] [INFO] Healthcheck ok
[2026-10-18 10:00:04] [INFO] Healthcheck ok
[2026-10-18 10:00:05] [INFO] Healthcheck ok
[2026-10-18 10:00:06] [ERROR] Database connection timeout: Connection refused!
Traceback (most recent call last):
  File "db.py", line 42, in connect
    raise TimeoutError("DB failed")
TimeoutError: DB failed
[2026-10-18 10:00:07] [INFO] Healthcheck ok
[2026-10-18 10:00:08] [INFO] Healthcheck ok
[2026-10-18 10:00:09] [INFO] Healthcheck ok
[2026-10-18 10:00:10] [INFO] Healthcheck ok
""" * 3
    compressed, metrics = pipeline.prune_context(logs, compression_ratio=0.5)
    
    assert metrics["savings_pct"] > 0.0
    # Critical error anchor must be kept
    assert "Database connection timeout" in compressed
    assert "TimeoutError: DB failed" in compressed


def test_pipeline_compress_cot(pipeline):
    raw_cot = """
<think>
Wait, let me rethink the problem.
Hold on, wait... let me check again.
The equation is 3x + 9 = 24.
Let me compute: 3x = 24 - 9 = 15.
Therefore x = 5.
Double check: 3(5) + 9 = 15 + 9 = 24. Yes correct!
</think>
<answer>5</answer>
"""
    compressed, metrics = pipeline.compress_cot(raw_cot)
    
    assert "<answer>5</answer>" in compressed
    assert "Wait, let me rethink" not in compressed
    assert "3x = 15" in compressed or "x = 5" in compressed


def test_mcp_tools_json_interface():
    # Test MCP tool prune_context
    raw_code = "def foo():\n    return 42\n" * 40
    res_json = prune_context(raw_code, compression_ratio=0.5)
    data = json.loads(res_json)
    
    assert "compressed_text" in data
    assert "metrics" in data
    assert data["metrics"]["orig_tokens"] > data["metrics"]["compressed_tokens"]

    # Test MCP tool compress_cot
    raw_cot = "<think>Wait, let me rethink. 2+2=4.</think><answer>4</answer>"
    res_cot_json = compress_cot(raw_cot)
    cot_data = json.loads(res_cot_json)
    
    assert "compressed_cot" in cot_data
    assert "metrics" in cot_data
