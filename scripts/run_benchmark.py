"""CLI Benchmark runner for BiggBoss.

Measures real query latency and precision without relying on hardcoded constants.
Complies with Hard Constraint #1 & #2 in AGENTS.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.eval.ablation import run_empirical_benchmark
from app.retrieval.visual_search import load_visual_models


def main():
    print("=" * 65)
    print("  BiggBoss Empirical Evaluation & Latency Benchmark Runner")
    print("  Measures active hardware runtime, latency and grounding")
    print("=" * 65)

    print("\n[1/2] Loading models (CPU/GPU)...")
    try:
        models = load_visual_models()
        print(f"      Models loaded successfully on: {models[3]}")
    except Exception as e:
        print(f"      Running in lightweight mock/offline mode ({e})")
        models = None

    print("\n[2/2] Executing empirical benchmark suite...")
    results = run_empirical_benchmark(models=models)

    print("\n" + "=" * 65)
    print("                   EMPIRICAL BENCHMARK REPORT")
    print("=" * 65)
    print(f"Status                  : {results['status']}")
    print(f"Device / Hardware       : {results['device']}")
    print(f"Operating System        : {results['os']}")
    print(f"Total Queries Evaluated : {results['total_queries_evaluated']}")
    print(f"Avg Query Latency       : {results['average_query_latency_ms']} ms")
    print(f"Min / Max Latency       : {results['min_latency_ms']} ms / {results['max_latency_ms']} ms")
    print(f"Grounding Accuracy      : {results['grounding_accuracy_evaluated'] * 100:.1f}%")
    print(f"Abstention on Negatives : {results['hard_negative_abstention_rate'] * 100:.1f}%")
    print(f"Total Benchmark Time    : {results['total_benchmark_time_seconds']} s")
    print("-" * 65)
    print("Per-Query Outcomes:")
    for r in results["per_query_results"]:
        print(f"  • [{r['status']}] '{r['query']}' ({r['latency_ms']} ms)")
    print("=" * 65)

    # Save artifact
    out_dir = _project_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nArtifact saved to: {report_path.relative_to(_project_root)}")


if __name__ == "__main__":
    main()
