"""CLI demo for VectorRBAC.

Generates synthetic document chunks with mixed clearance levels,
builds the topological complex, runs queries, and simulates an attack.
"""

from __future__ import annotations

import sys

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .engine import VectorRBACEngine
from .models import Clearance, DocumentChunk


def _generate_synthetic_data(seed: int = 42) -> list[DocumentChunk]:
    """Generate synthetic document chunks with embeddings and clearance levels."""
    rng = np.random.default_rng(seed)
    chunks: list[DocumentChunk] = []
    dim = 64  # Embedding dimension

    # Public documents (clearance 0) - cluster around center A
    center_a = rng.normal(0, 0.3, size=dim).astype(np.float32)
    for i in range(30):
        emb = center_a + rng.normal(0, 0.15, size=dim).astype(np.float32)
        emb = emb / np.linalg.norm(emb)  # Normalize
        chunks.append(DocumentChunk(
            id=0, text=f"Public doc chunk {i}: general company info",
            embedding=emb.tolist(), clearance=Clearance.PUBLIC,
            source_doc="public_handbook.pdf",
        ))

    # Internal documents (clearance 1) - cluster around center B (near A)
    center_b = center_a + rng.normal(0, 0.2, size=dim).astype(np.float32)
    for i in range(25):
        emb = center_b + rng.normal(0, 0.12, size=dim).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        chunks.append(DocumentChunk(
            id=0, text=f"Internal doc chunk {i}: engineering specs",
            embedding=emb.tolist(), clearance=Clearance.INTERNAL,
            source_doc="engineering_specs.pdf",
        ))

    # Confidential documents (clearance 2) - cluster around center C
    center_c = rng.normal(0.5, 0.3, size=dim).astype(np.float32)
    for i in range(20):
        emb = center_c + rng.normal(0, 0.1, size=dim).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        chunks.append(DocumentChunk(
            id=0, text=f"Confidential chunk {i}: financial projections",
            embedding=emb.tolist(), clearance=Clearance.CONFIDENTIAL,
            source_doc="financial_projections.pdf",
        ))

    # Board documents (clearance 3) - cluster around center D (near C)
    center_d = center_c + rng.normal(0, 0.15, size=dim).astype(np.float32)
    for i in range(10):
        emb = center_d + rng.normal(0, 0.08, size=dim).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        chunks.append(DocumentChunk(
            id=0, text=f"Board chunk {i}: M&A strategy and CEO compensation",
            embedding=emb.tolist(), clearance=Clearance.BOARD,
            source_doc="board_minutes.pdf",
        ))

    return chunks


def main() -> int:
    """Run the VectorRBAC demo."""
    console = Console()

    console.print(Panel.fit(
        "[bold cyan]VectorRBAC - Topological Access Control[/bold cyan]\n"
        "Preventing Latent-Space Privilege Escalation in RAG Systems",
        border_style="cyan",
    ))

    # Generate data
    console.print("\n[bold]Step 1:[/bold] Generating synthetic document corpus...")
    chunks = _generate_synthetic_data()
    console.print(f"  → {len(chunks)} chunks generated (Public: 30, Internal: 25, Confidential: 20, Board: 10)")

    # Build engine
    console.print("\n[bold]Step 2:[/bold] Building topological complex...")
    engine = VectorRBACEngine(similarity_threshold=0.65)
    engine.add_chunks(chunks)

    status = engine.get_status()
    console.print(f"  → Chunks: {status.total_chunks} | Edges: {status.total_edges} | Triangles: {status.total_triangles}")
    console.print(f"  → b₀={status.betti_numbers['b0']} | b₁={status.betti_numbers['b1']} | b₂={status.betti_numbers['b2']}")

    # Query as public user
    console.print("\n[bold]Step 3:[/bold] Query as PUBLIC user (clearance 0)")
    rng = np.random.default_rng(99)
    query_emb = rng.normal(0, 0.3, size=64).astype(np.float32)
    query_emb = query_emb / np.linalg.norm(query_emb)

    result = engine.query(query_emb.tolist(), Clearance.PUBLIC, top_k=5)
    console.print(f"  → Blocked: {result.blocked} | κ={result.kappa_effective:.4f} | Results: {len(result.results)}")
    if result.results:
        console.print(f"  → Top result: \"{result.results[0]['text'][:60]}...\"")

    # Query as internal user
    console.print("\n[bold]Step 4:[/bold] Query as INTERNAL user (clearance 1)")
    result = engine.query(query_emb.tolist(), Clearance.INTERNAL, top_k=5)
    console.print(f"  → Blocked: {result.blocked} | κ={result.kappa_effective:.4f} | Results: {len(result.results)}")
    if result.cup_product_filtered > 0:
        console.print(f"  → [yellow]Cup product filtered {result.cup_product_filtered} chunks (emergent super-permission)[/yellow]")

    # Simulate attack
    console.print("\n[bold]Step 5:[/bold] Simulating boundary probing attack (100 queries)")
    console.print("  → Attacker: PUBLIC (clearance 0) | Target: BOARD (clearance 3)")
    attack = engine.simulate_attack(
        target_clearance=Clearance.BOARD,
        num_queries=100,
        attacker_clearance=Clearance.PUBLIC,
    )

    attack_table = Table(title="Attack Simulation Results")
    attack_table.add_column("Metric", style="bold")
    attack_table.add_column("Value", justify="right")
    attack_table.add_column("Target", justify="right")
    attack_table.add_column("Status", justify="center")

    attack_table.add_row(
        "Block Rate", f"{attack.block_rate * 100:.1f}%", "≥ 95%",
        "[green]✓[/green]" if attack.block_rate >= 0.95 else "[red]✗[/red]"
    )
    attack_table.add_row(
        "Leak Detected", str(attack.leak_detected), "False",
        "[green]✓[/green]" if not attack.leak_detected else "[red]✗[/red]"
    )
    attack_table.add_row(
        "Avg κ_effective", f"{attack.avg_kappa:.4f}", "< 0.5",
        "[green]✓[/green]" if attack.avg_kappa < 0.5 else "[yellow]~[/yellow]"
    )
    attack_table.add_row(
        "Queries Blocked", f"{attack.blocked_count}/{attack.total_queries}", "≥ 95",
        "[green]✓[/green]" if attack.blocked_count >= 95 else "[red]✗[/red]"
    )

    console.print(attack_table)

    # Final status
    console.print("\n[bold]Step 6:[/bold] Final system status")
    status = engine.get_status()
    status_table = Table(title="System Coherence Invariants")
    status_table.add_column("Metric", style="bold")
    status_table.add_column("Value", justify="right")
    status_table.add_row("b₀ (components)", str(status.betti_numbers["b0"]))
    status_table.add_row("b₁ (cycles)", str(status.betti_numbers["b1"]))
    status_table.add_row("κ threshold", f"{status.kappa_effective_threshold:.2f}")
    status_table.add_row("Probe rate", f"{status.probe_rate:.0f}/min")
    console.print(status_table)

    console.print("\n[bold green]Demo complete.[/bold green]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
