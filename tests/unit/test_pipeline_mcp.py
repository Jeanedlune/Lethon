"""
Unit tests for specialized LethonPipeline and MCP tools (Code, Logs, Diff).
"""

import json
import pytest
from lethon.core.pipeline import LethonPipeline
from lethon.mcp.server import prune_code, prune_logs, prune_diff, prune_context


@pytest.fixture
def pipeline():
    return LethonPipeline()


def test_prune_code_preserves_syntax(pipeline):
    code = """
import os
import sys

pub struct CircuitValidator {
    pub root: [u8; 32],
    pub leaf_count: u64,
}

impl CircuitValidator {
    pub fn new(root: [u8; 32]) -> Self {
        Self { root, leaf_count: 0 }
    }

    pub fn compute_constraint(&self, a: u64, b: u64) -> u64 {
        let x = a + b;
        let y = x * 2;
        let z = y - 1;
        let w = z + 10;
        let v = w / 2;
        v
    }
}
""" * 10
    compressed, metrics = pipeline.prune_code(code, compression_ratio=0.4)
    
    assert metrics["orig_tokens"] > metrics["compressed_tokens"]
    assert metrics["savings_pct"] > 0.0
    assert "pub struct CircuitValidator" in compressed
    assert "pub fn compute_constraint" in compressed
    assert "import os" in compressed


def test_prune_logs_captures_tracebacks(pipeline):
    logs = """
[INFO] Application starting...
[INFO] Loading modules...
[INFO] Worker 1 ready
[INFO] Worker 2 ready
[INFO] Processing batch 1...
[INFO] Processing batch 2...
[ERROR] Database connection failed: connection refused on 127.0.0.1:5432
Traceback (most recent call last):
  File "server.py", line 124, in handle_request
    db.execute("SELECT 1")
ConnectionRefusedError: Port 5432 unreachable
[INFO] Retrying connection in 5s...
[INFO] Worker 1 idle
[INFO] Worker 2 idle
[INFO] Heartbeat OK
""" * 5
    compressed, metrics = pipeline.prune_logs(logs, context_window_lines=2)
    
    assert metrics["savings_pct"] > 0.0
    assert "Database connection failed" in compressed
    assert "ConnectionRefusedError: Port 5432 unreachable" in compressed


def test_prune_diff_filters_lockfiles(pipeline):
    diff = """
diff --git a/src/main.rs b/src/main.rs
index abc..def 100644
--- a/src/main.rs
+++ b/src/main.rs
@@ -10,3 +10,4 @@
 fn main() {
+    println!("Hello Lethon");
 }
diff --git a/Cargo.lock b/Cargo.lock
index 111..222 100644
--- a/Cargo.lock
+++ b/Cargo.lock
@@ -1,500 +1,500 @@
+ massive lockfile noise lines...
+ massive lockfile noise lines...
+ massive lockfile noise lines...
"""
    compressed, metrics = pipeline.prune_diff(diff)
    
    assert "println!(\"Hello Lethon\");" in compressed
    assert "[Lethon: lockfile/asset diff pruned]" in compressed
    assert metrics["ignored_files"] == 1


def test_mcp_tools_specialized():
    code_res = json.loads(prune_code("fn test() { let x = 1; }\n" * 30, compression_ratio=0.4))
    assert "compressed_code" in code_res
    assert code_res["metrics"]["savings_pct"] > 0

    log_res = json.loads(prune_logs("[INFO] ok\n" * 30 + "[ERROR] bug found!\n" + "[INFO] ok\n" * 30))
    assert "compressed_logs" in log_res
    assert "bug found!" in log_res["compressed_logs"]

    diff_res = json.loads(prune_diff("diff --git a/app.js b/app.js\n+console.log(1);\n"))
    assert "compressed_diff" in diff_res
