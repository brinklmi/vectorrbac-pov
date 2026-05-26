"""VectorRBAC Engine - Topological Access Control for Vector Retrieval.

Core engine that:
1. Builds a simplicial complex from document chunk embeddings
2. Constructs clearance subcomplexes (L0, L1, L2, L3)
3. Computes connected components per clearance level
4. Enforces component-bounded retrieval
5. Applies impedance-based κ_effective throttling
6. Detects emergent super-permissions via cup product
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Optional

import numpy as np

from .models import (
    AttackResult,
    Clearance,
    DocumentChunk,
    QueryContext,
    RetrievalResult,
    SystemStatus,
)


class VectorRBACEngine:
    """Topological access control engine for vector databases.

    Parameters
    ----------
    similarity_threshold : float
        Cosine similarity threshold for edge creation (default 0.65).
    kappa_threshold : float
        Base κ_effective threshold for blocking (default 0.3).
    max_probe_rate : float
        Maximum probes per minute before threshold escalation.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.65,
        kappa_threshold: float = 0.3,
        max_probe_rate: float = 50.0,
    ):
        self.similarity_threshold = similarity_threshold
        self.kappa_threshold = kappa_threshold
        self.max_probe_rate = max_probe_rate

        # Storage
        self._chunks: dict[int, DocumentChunk] = {}
        self._embeddings: dict[int, np.ndarray] = {}
        self._next_id: int = 0

        # Simplicial complex (edges and triangles)
        self._edges: set[tuple[int, int]] = set()
        self._triangles: set[tuple[int, int, int]] = set()

        # Connected components per clearance level
        self._components: dict[int, dict[int, int]] = {}  # clearance -> {chunk_id: component_id}

        # Cup product table: (edge1, edge2) -> triangle
        self._cup_table: dict[tuple[tuple[int, int], tuple[int, int]], tuple[int, int, int]] = {}

        # Probe tracking (sliding window)
        self._probe_timestamps: list[float] = []

        # Betti numbers cache
        self._betti_dirty: bool = True
        self._betti: dict[str, int] = {"b0": 0, "b1": 0, "b2": 0}

    # ─── Indexing ─────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[DocumentChunk]) -> list[int]:
        """Add document chunks to the complex.

        Computes pairwise similarity, creates edges and triangles,
        and rebuilds connected components.

        Returns list of assigned chunk IDs.
        """
        new_ids: list[int] = []

        for chunk in chunks:
            chunk_id = self._next_id
            self._next_id += 1
            chunk.id = chunk_id
            self._chunks[chunk_id] = chunk
            self._embeddings[chunk_id] = np.array(chunk.embedding, dtype=np.float32)
            new_ids.append(chunk_id)

        # Build edges for new chunks against all existing
        self._build_edges(new_ids)

        # Complete triangles (cliques of size 3)
        self._complete_triangles(new_ids)

        # Rebuild components for all clearance levels
        self._rebuild_components()

        # Rebuild cup product table
        self._rebuild_cup_table()

        self._betti_dirty = True
        return new_ids

    def _build_edges(self, new_ids: list[int]) -> None:
        """Build edges between new chunks and all existing chunks."""
        all_ids = list(self._chunks.keys())

        for new_id in new_ids:
            emb_new = self._embeddings[new_id]
            norm_new = np.linalg.norm(emb_new)
            if norm_new == 0:
                continue

            for other_id in all_ids:
                if other_id == new_id:
                    continue

                emb_other = self._embeddings[other_id]
                norm_other = np.linalg.norm(emb_other)
                if norm_other == 0:
                    continue

                # Cosine similarity
                sim = float(np.dot(emb_new, emb_other) / (norm_new * norm_other))

                if sim >= self.similarity_threshold:
                    edge = (min(new_id, other_id), max(new_id, other_id))
                    self._edges.add(edge)

    def _complete_triangles(self, new_ids: list[int]) -> None:
        """Find triangles (3-cliques) involving new vertices."""
        # Build adjacency for fast lookup
        adj: dict[int, set[int]] = defaultdict(set)
        for u, v in self._edges:
            adj[u].add(v)
            adj[v].add(u)

        new_set = set(new_ids)

        for new_id in new_ids:
            neighbors = adj[new_id]
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 >= n2:
                        continue
                    # Check if n1-n2 edge exists
                    edge = (min(n1, n2), max(n1, n2))
                    if edge in self._edges:
                        tri = tuple(sorted((new_id, n1, n2)))
                        self._triangles.add(tri)

    def _rebuild_components(self) -> None:
        """Rebuild connected components for each clearance subcomplex."""
        self._components = {}

        for clearance_level in range(4):
            # Get vertices at or below this clearance
            vertices = {
                cid for cid, chunk in self._chunks.items()
                if chunk.clearance <= clearance_level
            }

            # Get edges within this subcomplex
            edges_in_sub = [
                (u, v) for u, v in self._edges
                if u in vertices and v in vertices
            ]

            # BFS to find components
            component_map: dict[int, int] = {}
            visited: set[int] = set()
            comp_id = 0

            # Build adjacency for subcomplex
            adj: dict[int, set[int]] = defaultdict(set)
            for u, v in edges_in_sub:
                adj[u].add(v)
                adj[v].add(u)

            for start in vertices:
                if start in visited:
                    continue
                # BFS
                queue = deque([start])
                visited.add(start)
                while queue:
                    node = queue.popleft()
                    component_map[node] = comp_id
                    for neighbor in adj[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                comp_id += 1

            self._components[clearance_level] = component_map

    def _rebuild_cup_table(self) -> None:
        """Build cup product table from triangles."""
        self._cup_table = {}

        for tri in self._triangles:
            v0, v1, v2 = tri
            edges = [(v0, v1), (v1, v2), (v0, v2)]

            for i, e1 in enumerate(edges):
                for j, e2 in enumerate(edges):
                    if i >= j:
                        continue
                    self._cup_table[(e1, e2)] = tri
                    self._cup_table[(e2, e1)] = tri

    # ─── Query Retrieval ──────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: list[float],
        user_clearance: Clearance,
        top_k: int = 10,
    ) -> RetrievalResult:
        """Execute a secure retrieval query.

        Enforces component-bounded retrieval with impedance throttling
        and cup product filtering.
        """
        q_emb = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_emb)

        if q_norm == 0 or not self._chunks:
            return RetrievalResult(blocked=True, blocked_reason="Empty query or index")

        # Step 1: Find nearest vertex in clearance-appropriate subcomplex
        component_map = self._components.get(user_clearance, {})
        if not component_map:
            return RetrievalResult(blocked=True, blocked_reason="No accessible chunks")

        accessible_ids = list(component_map.keys())
        nearest_id, nearest_sim = self._find_nearest(q_emb, accessible_ids)

        if nearest_id is None:
            return RetrievalResult(blocked=True, blocked_reason="No nearest vertex found")

        # Step 2: Get component ID
        comp_id = component_map[nearest_id]

        # Step 3: Compute κ_effective (impedance matching)
        kappa = self._compute_kappa(q_emb, nearest_id, user_clearance)

        # Step 4: Check threshold (with adaptive escalation)
        current_threshold = self._get_adaptive_threshold()

        if kappa < current_threshold:
            self._record_probe()
            return RetrievalResult(
                kappa_effective=kappa,
                component_id=comp_id,
                blocked=True,
                blocked_reason=f"Boundary impedance mismatch: κ={kappa:.3f} < threshold={current_threshold:.3f}",
            )

        # Step 5: Retrieve all chunks in same component
        component_chunks = [
            cid for cid, ccomp in component_map.items() if ccomp == comp_id
        ]

        # Step 6: Rank by similarity
        ranked = self._rank_by_similarity(q_emb, component_chunks)

        # Step 7: Cup product filter
        filtered, cup_removed = self._cup_product_filter(ranked, user_clearance)

        # Step 8: Return top_k
        results = []
        for chunk_id, sim in filtered[:top_k]:
            chunk = self._chunks[chunk_id]
            results.append({
                "chunk_id": chunk_id,
                "text": chunk.text,
                "similarity": round(sim, 4),
                "clearance": chunk.clearance.value,
            })

        return RetrievalResult(
            results=results,
            kappa_effective=round(kappa, 4),
            component_id=comp_id,
            blocked=False,
            cup_product_filtered=cup_removed,
        )

    def _find_nearest(
        self, q_emb: np.ndarray, candidate_ids: list[int]
    ) -> tuple[Optional[int], float]:
        """Find nearest chunk by cosine similarity."""
        best_id: Optional[int] = None
        best_sim: float = -1.0
        q_norm = np.linalg.norm(q_emb)

        for cid in candidate_ids:
            emb = self._embeddings[cid]
            norm = np.linalg.norm(emb)
            if norm == 0:
                continue
            sim = float(np.dot(q_emb, emb) / (q_norm * norm))
            if sim > best_sim:
                best_sim = sim
                best_id = cid

        return best_id, best_sim

    def _compute_kappa(
        self, q_emb: np.ndarray, nearest_id: int, user_clearance: Clearance
    ) -> float:
        """Compute κ_effective = 2*Z_query / (Z_query + Z_boundary).

        Z_query = norm of query embedding.
        Z_boundary = average norm of embeddings adjacent to restricted components.
        """
        z_query = float(np.linalg.norm(q_emb))

        # Find boundary vertices: accessible vertices adjacent to restricted ones
        accessible = set(self._components.get(user_clearance, {}).keys())
        restricted = set(self._chunks.keys()) - accessible

        if not restricted:
            return 1.0  # No restricted content, fully matched

        # Find accessible vertices that have edges to restricted vertices
        boundary_ids: list[int] = []
        for u, v in self._edges:
            if u in accessible and v in restricted:
                boundary_ids.append(u)
            elif v in accessible and u in restricted:
                boundary_ids.append(v)

        if not boundary_ids:
            return 1.0  # No boundary, fully matched

        # Z_boundary = average norm of boundary vertex embeddings
        boundary_norms = [float(np.linalg.norm(self._embeddings[bid])) for bid in boundary_ids]
        z_boundary = sum(boundary_norms) / len(boundary_norms) if boundary_norms else z_query

        if z_query + z_boundary == 0:
            return 0.0

        kappa = 2.0 * z_query / (z_query + z_boundary)
        return min(1.0, kappa)

    def _rank_by_similarity(
        self, q_emb: np.ndarray, chunk_ids: list[int]
    ) -> list[tuple[int, float]]:
        """Rank chunks by cosine similarity to query."""
        q_norm = np.linalg.norm(q_emb)
        scored: list[tuple[int, float]] = []

        for cid in chunk_ids:
            emb = self._embeddings[cid]
            norm = np.linalg.norm(emb)
            if norm == 0:
                continue
            sim = float(np.dot(q_emb, emb) / (q_norm * norm))
            scored.append((cid, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _cup_product_filter(
        self, ranked: list[tuple[int, float]], user_clearance: Clearance
    ) -> tuple[list[tuple[int, float]], int]:
        """Filter results using cup product table.

        If two retrieved edges form a triangle whose max clearance
        exceeds user_clearance, remove those vertices.
        """
        result_ids = {cid for cid, _ in ranked}
        removed: set[int] = set()

        for (e1, e2), tri in self._cup_table.items():
            # Check if both edges have vertices in result set
            if e1[0] in result_ids and e1[1] in result_ids and e2[0] in result_ids and e2[1] in result_ids:
                # Check triangle clearance
                tri_clearances = [
                    self._chunks[v].clearance for v in tri if v in self._chunks
                ]
                if tri_clearances and max(tri_clearances) > user_clearance:
                    # Remove triangle vertices from results
                    for v in tri:
                        if v in result_ids:
                            removed.add(v)

        filtered = [(cid, sim) for cid, sim in ranked if cid not in removed]
        return filtered, len(removed)

    # ─── Metabolic Fuse ───────────────────────────────────────────────────────

    def _record_probe(self) -> None:
        """Record a boundary probe event."""
        self._probe_timestamps.append(time.time())

    @property
    def probe_rate(self) -> float:
        """Probes per minute in the last 60 seconds."""
        now = time.time()
        cutoff = now - 60.0
        self._probe_timestamps = [t for t in self._probe_timestamps if t > cutoff]
        return float(len(self._probe_timestamps))

    def _get_adaptive_threshold(self) -> float:
        """Get current κ threshold with adaptive escalation."""
        rate = self.probe_rate
        if rate > self.max_probe_rate:
            return 0.7
        elif rate > 10:
            return 0.5
        return self.kappa_threshold

    # ─── Status ───────────────────────────────────────────────────────────────

    def get_status(self) -> SystemStatus:
        """Get current system coherence invariants."""
        if self._betti_dirty:
            self._compute_betti()

        return SystemStatus(
            betti_numbers=self._betti.copy(),
            kappa_effective_threshold=self._get_adaptive_threshold(),
            probe_rate=self.probe_rate,
            total_chunks=len(self._chunks),
            total_edges=len(self._edges),
            total_triangles=len(self._triangles),
        )

    def _compute_betti(self) -> None:
        """Compute Betti numbers for the full complex."""
        n0 = len(self._chunks)
        n1 = len(self._edges)
        n2 = len(self._triangles)

        # b0 from full complex components
        full_components = self._components.get(3, {})
        if full_components:
            b0 = len(set(full_components.values()))
        else:
            b0 = n0

        # b1 = n1 - n0 + b0 - n2 (Euler characteristic relation)
        # χ = n0 - n1 + n2 = b0 - b1 + b2
        # For POC: b1 = n1 - (n0 - b0) - n2 (rank of boundary)
        b1 = max(0, n1 - (n0 - b0) - n2)
        b2 = 0  # Simplified for POC

        self._betti = {"b0": b0, "b1": b1, "b2": b2}
        self._betti_dirty = False

    # ─── Attack Simulation ────────────────────────────────────────────────────

    def simulate_attack(
        self,
        target_clearance: Clearance,
        num_queries: int = 100,
        attacker_clearance: Clearance = Clearance.PUBLIC,
        seed: int = 42,
    ) -> AttackResult:
        """Simulate a boundary probing attack.

        Generates queries near the boundary of restricted components
        and checks if any leak through.
        """
        rng = np.random.default_rng(seed)

        # Find boundary embeddings (accessible vertices near restricted)
        accessible = set(self._components.get(attacker_clearance, {}).keys())
        restricted = {
            cid for cid, chunk in self._chunks.items()
            if chunk.clearance >= target_clearance
        }

        # Find accessible vertices adjacent to restricted
        boundary_ids: list[int] = []
        for u, v in self._edges:
            if u in accessible and v in restricted:
                boundary_ids.append(u)
            elif v in accessible and u in restricted:
                boundary_ids.append(v)

        if not boundary_ids:
            return AttackResult(
                total_queries=num_queries,
                blocked_count=num_queries,
                avg_kappa=1.0,
                leak_detected=False,
                block_rate=1.0,
            )

        # Generate attack queries near boundary embeddings
        blocked = 0
        kappas: list[float] = []
        leak = False

        for i in range(num_queries):
            # Pick a random boundary vertex and perturb its embedding
            bid = boundary_ids[rng.integers(0, len(boundary_ids))]
            base_emb = self._embeddings[bid]
            noise = rng.normal(0, 0.05, size=base_emb.shape).astype(np.float32)
            attack_emb = base_emb + noise

            result = self.query(
                query_embedding=attack_emb.tolist(),
                user_clearance=attacker_clearance,
                top_k=10,
            )

            kappas.append(result.kappa_effective)

            if result.blocked:
                blocked += 1
            else:
                # Check if any returned chunk has clearance >= target
                for r in result.results:
                    if r["clearance"] >= target_clearance:
                        leak = True

        avg_kappa = sum(kappas) / len(kappas) if kappas else 0.0

        return AttackResult(
            total_queries=num_queries,
            blocked_count=blocked,
            avg_kappa=round(avg_kappa, 4),
            leak_detected=leak,
            block_rate=round(blocked / num_queries, 4) if num_queries > 0 else 0.0,
        )
