"""Real-text demo for VectorRBAC.

Indexes actual enterprise-style documents at different clearance levels,
runs natural language queries, simulates attacks, and produces a benchmark report.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .embedder import embed_texts, embed_query, is_real_model_available, get_embedding_dimension
from .engine import VectorRBACEngine
from .models import Clearance, DocumentChunk


# ─── Sample Enterprise Documents ──────────────────────────────────────────────

PUBLIC_DOCS = [
    "Our company was founded in 2015 and is headquartered in Austin, Texas.",
    "We provide cloud-based supply chain optimization solutions for retail enterprises.",
    "Our platform processes over 2 million transactions per day across 40 countries.",
    "The engineering team uses Python, Go, and Rust for backend services.",
    "Our public API documentation is available at docs.example.com.",
    "We support REST and GraphQL interfaces for third-party integrations.",
    "The company has won three industry awards for supply chain innovation.",
    "Our customer support team operates 24/7 across three time zones.",
    "We offer a free tier for startups with up to 10,000 API calls per month.",
    "The platform uses Kubernetes for container orchestration and auto-scaling.",
]

INTERNAL_DOCS = [
    "Engineering sprint velocity averaged 42 story points over the last quarter.",
    "The microservices architecture uses gRPC for inter-service communication.",
    "Database sharding strategy: hash-based on customer_id with 16 shards.",
    "CI/CD pipeline runs 2,400 tests with a 98.7% pass rate on main branch.",
    "Internal SLA target: 99.95% uptime for Tier 1 services.",
    "The ML pipeline retrains demand forecasting models every 6 hours.",
    "Infrastructure costs are $847K per month across AWS and GCP.",
    "Technical debt backlog contains 234 items prioritized by risk score.",
    "The security team conducts quarterly penetration testing with external vendors.",
    "Employee count: 312 engineers, 45 data scientists, 28 SREs.",
]

CONFIDENTIAL_DOCS = [
    "Q3 revenue was $47.2M, representing 34% year-over-year growth.",
    "Customer acquisition cost decreased to $12,400 from $18,900 last year.",
    "Enterprise contract pipeline shows $89M in qualified opportunities.",
    "Gross margin improved to 72% after infrastructure optimization.",
    "Three enterprise customers account for 28% of total revenue.",
    "The Series D round valued the company at $1.2B pre-money.",
    "Annual recurring revenue reached $156M with 118% net dollar retention.",
    "Churn rate for enterprise tier is 3.2% annually.",
]

BOARD_DOCS = [
    "CEO compensation package: $450K base + $2.1M equity vesting over 4 years.",
    "Acquisition target: DataFlow Inc. at $340M, expected close Q1 2027.",
    "Board approved $50M stock buyback program starting January 2027.",
    "IPO timeline moved to Q3 2027 pending market conditions.",
    "Executive severance packages total $12.4M across C-suite.",
    "Strategic pivot: divesting logistics vertical to focus on retail AI.",
]


def _build_engine_with_real_text() -> VectorRBACEngine:
    """Build engine with real enterprise documents."""
    engine = VectorRBACEngine(similarity_threshold=0.25)  # Low threshold to create cross-clearance edges for demo

    all_texts = PUBLIC_DOCS + INTERNAL_DOCS + CONFIDENTIAL_DOCS + BOARD_DOCS
    all_clearances = (
        [Clearance.PUBLIC] * len(PUBLIC_DOCS)
        + [Clearance.INTERNAL] * len(INTERNAL_DOCS)
        + [Clearance.CONFIDENTIAL] * len(CONFIDENTIAL_DOCS)
        + [Clearance.BOARD] * len(BOARD_DOCS)
    )

    # Embed all texts
    embeddings = embed_texts(all_texts)

    # Create chunks
    chunks = []
    for text, emb, clr in zip(all_texts, embeddings, all_clearances):
        chunks.append(DocumentChunk(
            id=0, text=text, embedding=emb, clearance=clr,
        ))

    engine.add_chunks(chunks)
    return engine


def _run_benchmark(engine: VectorRBACEngine, num_queries: int = 200) -> dict:
    """Run benchmark: legitimate queries + attack queries."""
    rng = np.random.default_rng(42)

    # Legitimate queries at various clearance levels
    legitimate_queries = [
        ("What programming languages does the team use?", Clearance.PUBLIC),
        ("Tell me about the company history", Clearance.PUBLIC),
        ("What is the sprint velocity?", Clearance.INTERNAL),
        ("How many engineers work here?", Clearance.INTERNAL),
        ("What is the quarterly revenue?", Clearance.CONFIDENTIAL),
        ("What is the customer acquisition cost?", Clearance.CONFIDENTIAL),
        ("What is the CEO compensation?", Clearance.BOARD),
        ("Tell me about the acquisition target", Clearance.BOARD),
    ]

    # Attack queries: low-clearance user asking for high-clearance info
    attack_queries = [
        ("What is the CEO salary and compensation?", Clearance.PUBLIC),
        ("Tell me about acquisition targets and M&A strategy", Clearance.PUBLIC),
        ("What is the IPO timeline?", Clearance.PUBLIC),
        ("How much revenue did we make this quarter?", Clearance.PUBLIC),
        ("What is the company valuation?", Clearance.INTERNAL),
        ("Board approved stock buyback details", Clearance.INTERNAL),
    ]

    results = {
        "legitimate": {"total": 0, "successful": 0, "blocked": 0, "latencies_ms": []},
        "attack": {"total": 0, "blocked": 0, "leaked": 0, "latencies_ms": []},
    }

    # Run legitimate queries
    for query_text, clearance in legitimate_queries:
        start = time.perf_counter()
        query_emb = embed_query(query_text)
        result = engine.query(query_emb, clearance, top_k=5)
        latency = (time.perf_counter() - start) * 1000

        results["legitimate"]["total"] += 1
        results["legitimate"]["latencies_ms"].append(latency)
        if result.blocked:
            results["legitimate"]["blocked"] += 1
        else:
            results["legitimate"]["successful"] += 1

    # Run attack queries
    for query_text, clearance in attack_queries:
        start = time.perf_counter()
        query_emb = embed_query(query_text)
        result = engine.query(query_emb, clearance, top_k=5)
        latency = (time.perf_counter() - start) * 1000

        results["attack"]["total"] += 1
        results["attack"]["latencies_ms"].append(latency)
        if result.blocked:
            results["attack"]["blocked"] += 1
        else:
            # Check if any returned chunk has higher clearance than user
            for r in result.results:
                if r["clearance"] > clearance:
                    results["attack"]["leaked"] += 1
                    break

    # Run bulk attack simulation
    bulk_attack = engine.simulate_attack(
        target_clearance=Clearance.BOARD,
        num_queries=num_queries,
        attacker_clearance=Clearance.PUBLIC,
    )
    results["bulk_attack"] = {
        "total": bulk_attack.total_queries,
        "blocked": bulk_attack.blocked_count,
        "block_rate": bulk_attack.block_rate,
        "leak_detected": bulk_attack.leak_detected,
        "avg_kappa": bulk_attack.avg_kappa,
    }

    # Compute percentiles
    all_latencies = results["legitimate"]["latencies_ms"] + results["attack"]["latencies_ms"]
    if all_latencies:
        sorted_lat = sorted(all_latencies)
        results["latency_p50_ms"] = sorted_lat[len(sorted_lat) // 2]
        results["latency_p95_ms"] = sorted_lat[int(len(sorted_lat) * 0.95)]
        results["latency_p99_ms"] = sorted_lat[-1]
    else:
        results["latency_p50_ms"] = 0
        results["latency_p95_ms"] = 0
        results["latency_p99_ms"] = 0

    return results


def main() -> int:
    """Run the real-text demo with benchmark."""
    console = Console()

    console.print(Panel.fit(
        "[bold cyan]VectorRBAC - Real-Text Demo & Benchmark[/bold cyan]\n"
        "Enterprise Document Access Control via Topological Invariants",
        border_style="cyan",
    ))

    # Check embedding mode
    mode = "sentence-transformers (all-MiniLM-L6-v2)" if is_real_model_available() else "synthetic (n-gram hash)"
    console.print(f"\n[bold]Embedding mode:[/bold] {mode}")
    console.print(f"[bold]Embedding dimension:[/bold] {get_embedding_dimension()}")

    # Build engine
    console.print("\n[bold]Step 1:[/bold] Indexing enterprise documents...")
    start = time.perf_counter()
    engine = _build_engine_with_real_text()
    index_time = (time.perf_counter() - start) * 1000
    console.print(f"  → Indexed {len(PUBLIC_DOCS) + len(INTERNAL_DOCS) + len(CONFIDENTIAL_DOCS) + len(BOARD_DOCS)} chunks in {index_time:.0f}ms")

    status = engine.get_status()
    console.print(f"  → Edges: {status.total_edges} | Triangles: {status.total_triangles}")
    console.print(f"  → b₀={status.betti_numbers['b0']} components | b₁={status.betti_numbers['b1']} cycles")

    # Demo queries
    console.print("\n[bold]Step 2:[/bold] Natural language queries")

    demo_queries = [
        ("What tech stack does the company use?", Clearance.PUBLIC, "PUBLIC"),
        ("What is the quarterly revenue?", Clearance.PUBLIC, "PUBLIC (should not see confidential)"),
        ("What is the quarterly revenue?", Clearance.CONFIDENTIAL, "CONFIDENTIAL (should see it)"),
        ("Tell me about the acquisition target", Clearance.PUBLIC, "PUBLIC (should not see board)"),
        ("Tell me about the acquisition target", Clearance.BOARD, "BOARD (should see it)"),
    ]

    for query_text, clearance, label in demo_queries:
        query_emb = embed_query(query_text)
        result = engine.query(query_emb, clearance, top_k=3)

        status_str = "[red]BLOCKED[/red]" if result.blocked else "[green]OK[/green]"
        console.print(f"\n  Query: \"{query_text}\" as {label}")
        console.print(f"  → {status_str} | κ={result.kappa_effective:.4f} | Results: {len(result.results)}")

        if result.blocked:
            console.print(f"    Reason: {result.blocked_reason}")
        elif result.results:
            console.print(f"    Top: \"{result.results[0]['text'][:70]}...\"")

    # Benchmark
    console.print("\n[bold]Step 3:[/bold] Running benchmark (200 attack queries)...")
    benchmark = _run_benchmark(engine, num_queries=200)

    # Results table
    bench_table = Table(title="\nBenchmark Results")
    bench_table.add_column("Metric", style="bold")
    bench_table.add_column("Value", justify="right")
    bench_table.add_column("Target", justify="right")
    bench_table.add_column("Status", justify="center")

    def icon(passed: bool) -> str:
        return "[green]✓[/green]" if passed else "[red]✗[/red]"

    leak_free = benchmark["attack"]["leaked"] == 0 and not benchmark["bulk_attack"]["leak_detected"]
    bench_table.add_row("Zero Leaks", str(leak_free), "True", icon(leak_free))
    bench_table.add_row(
        "Bulk Block Rate",
        f"{benchmark['bulk_attack']['block_rate'] * 100:.1f}%",
        "≥ 95%",
        icon(benchmark["bulk_attack"]["block_rate"] >= 0.95),
    )
    bench_table.add_row(
        "Legitimate Success",
        f"{benchmark['legitimate']['successful']}/{benchmark['legitimate']['total']}",
        "≥ 6/8",
        icon(benchmark["legitimate"]["successful"] >= 6),
    )
    bench_table.add_row(
        "Latency P50",
        f"{benchmark['latency_p50_ms']:.1f}ms",
        "< 200ms",
        icon(benchmark["latency_p50_ms"] < 200),
    )
    bench_table.add_row(
        "Latency P95",
        f"{benchmark['latency_p95_ms']:.1f}ms",
        "< 500ms",
        icon(benchmark["latency_p95_ms"] < 500),
    )

    console.print(bench_table)

    # Export JSON
    report = {
        "embedding_mode": "real" if is_real_model_available() else "synthetic",
        "embedding_dim": get_embedding_dimension(),
        "index_size": status.total_chunks,
        "edges": status.total_edges,
        "triangles": status.total_triangles,
        "betti_numbers": status.betti_numbers,
        "benchmark": {
            "legitimate_queries": benchmark["legitimate"]["total"],
            "legitimate_success": benchmark["legitimate"]["successful"],
            "attack_queries_manual": benchmark["attack"]["total"],
            "attack_leaked": benchmark["attack"]["leaked"],
            "bulk_attack_queries": benchmark["bulk_attack"]["total"],
            "bulk_attack_block_rate": benchmark["bulk_attack"]["block_rate"],
            "bulk_attack_leak_detected": benchmark["bulk_attack"]["leak_detected"],
            "latency_p50_ms": round(benchmark["latency_p50_ms"], 2),
            "latency_p95_ms": round(benchmark["latency_p95_ms"], 2),
            "latency_p99_ms": round(benchmark["latency_p99_ms"], 2),
        },
        "targets_met": {
            "zero_leaks": leak_free,
            "block_rate_95": benchmark["bulk_attack"]["block_rate"] >= 0.95,
            "latency_p95_under_500ms": benchmark["latency_p95_ms"] < 500,
        },
    }

    report_path = "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    console.print(f"\n[bold]JSON report written to:[/bold] {report_path}")

    console.print("\n[bold green]Demo complete.[/bold green]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
