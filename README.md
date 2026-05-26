# VectorRBAC - Topological Access Control for RAG Systems

**Preventing latent-space privilege escalation in vector databases using simplicial homology, impedance matching, and cup product cohomology.**

## The Problem: Latent-Space Privilege Escalation

Vector embeddings lose rigid RBAC boundaries when chunks from enterprise documents are blended into an index. A query can reconstruct restricted information from latent weights even if no single chunk is directly accessible — because embeddings encode semantic relationships *across* clearance boundaries.

Standard metadata filtering fails because:
- Similarity search returns chunks near the query in latent space, regardless of clearance
- Two low-clearance chunks can combine to reveal high-clearance information (emergent super-permissions)
- Boundary probing attacks systematically exploit the latent-space proximity between clearance levels

## The Solution: Topological Access Control

Model the document index as a **simplicial complex** and enforce access via topological invariants:

| Concept | Topological Mapping |
|---------|-------------------|
| Document chunk | 0-simplex (vertex) with embedding + clearance |
| Semantic similarity | 1-simplex (edge) if cosine sim > θ |
| Clearance boundary | Connected component boundary in subcomplex |
| Unauthorized retrieval | Path crossing component boundary |
| Emergent super-permission | Non-zero cup product of two low-clearance edges |
| Throttling | κ_effective = 2Z_query / (Z_query + Z_boundary) |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VectorRBAC Engine                              │
├──────────────────┬──────────────────┬───────────────────────────┤
│  Simplicial      │  Access Control  │  Runtime Protection        │
│  Complex         │                  │                           │
│                  │  • Clearance     │  • Impedance κ_effective  │
│  • Embeddings    │    Subcomplexes  │  • Metabolic Fuse         │
│  • Similarity    │    (L0-L3)       │    (adaptive threshold)   │
│    Edges         │  • Component-    │  • Cup Product Filter     │
│  • Clique        │    Bounded       │    (emergent detection)   │
│    Triangles     │    Retrieval     │  • Probe Rate Tracking    │
│  • Betti Numbers │  • BFS Components│                           │
└──────────────────┴──────────────────┴───────────────────────────┘
```

### Key Invariants

- **b₀** (connected components): Clearance boundaries. Queries cannot cross component boundaries.
- **κ_effective**: Impedance matching coefficient. Below 0.3 → access denied (boundary probe detected).
- **Cup product**: Detects when two accessible chunks combine into restricted information.
- **Adaptive threshold**: Probe rate > 50/min escalates κ threshold to 0.7.

## Installation

```bash
pip install -e '.[dev]'
```

## Usage

### Run Demo

```bash
vectorrbac-demo
```

Generates 85 synthetic chunks across 4 clearance levels, builds the topological complex, runs queries at different clearance levels, and simulates a boundary probing attack.

### Run Tests

```bash
pytest tests/ -q
```

## Results

From the demo with 85 synthetic chunks:

| Metric | Result | Target |
|--------|--------|--------|
| Block Rate (attack) | 100% | ≥ 95% |
| Leak Detected | False | False |
| Latency | < 100ms | < 500ms |
| Components (b₀) | 2 | ≥ 2 |

## API Endpoints

- `POST /index` — Add document chunks to the complex
- `POST /query` — Secure retrieval with topological access control
- `GET /status` — System coherence invariants
- `POST /simulate_attack` — Boundary probing attack simulation

## Project Structure

```
vectorrbac/
├── __init__.py     # Package init
├── models.py       # Domain models (DocumentChunk, Clearance, etc.)
├── engine.py       # Core engine (complex, retrieval, κ, cup product)
└── cli.py          # CLI demo with synthetic data + attack simulation
tests/
└── test_engine.py  # Unit and integration tests
```

## How It Works

1. **Indexing**: Chunks are embedded and connected by similarity edges. Triangles form from 3-cliques. Connected components are computed per clearance subcomplex.

2. **Query**: The query embedding finds its nearest accessible vertex. Retrieval is bounded to that vertex's connected component — never crossing into restricted territory.

3. **Impedance Check**: κ_effective measures how close the query is to a clearance boundary. Low κ = boundary probe = blocked.

4. **Cup Product Filter**: Even within an accessible component, if two retrieved edges form a triangle whose vertices include restricted content, those vertices are removed.

5. **Adaptive Throttling**: Repeated boundary probes escalate the κ threshold, making it progressively harder to probe.

## References

- OWASP Agentic AI Threats — RAG poisoning and privilege escalation
- Hatcher, A. *Algebraic Topology* (2002)
- Verlinde, E. "On the Origin of Gravity and the Laws of Newton" (2010) — impedance matching derivation
