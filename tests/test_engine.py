"""Tests for the VectorRBAC engine."""

import numpy as np
import pytest

from vectorrbac.engine import VectorRBACEngine
from vectorrbac.models import Clearance, DocumentChunk


def _make_chunk(clearance: Clearance, center: np.ndarray, rng, dim: int = 16) -> DocumentChunk:
    """Create a chunk with embedding near a center."""
    emb = center + rng.normal(0, 0.1, size=dim).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    return DocumentChunk(id=0, text=f"chunk_{clearance.name}", embedding=emb.tolist(), clearance=clearance)


def _build_test_engine(seed: int = 42) -> VectorRBACEngine:
    """Build a test engine with two clearance clusters."""
    rng = np.random.default_rng(seed)
    dim = 16

    # Public cluster (center at [1, 0, 0, ...])
    center_pub = np.zeros(dim, dtype=np.float32)
    center_pub[0] = 1.0

    # Board cluster (center at [0, 1, 0, ...])
    center_board = np.zeros(dim, dtype=np.float32)
    center_board[1] = 1.0

    chunks = []
    for _ in range(10):
        chunks.append(_make_chunk(Clearance.PUBLIC, center_pub, rng, dim))
    for _ in range(10):
        chunks.append(_make_chunk(Clearance.BOARD, center_board, rng, dim))

    engine = VectorRBACEngine(similarity_threshold=0.7)
    engine.add_chunks(chunks)
    return engine


class TestIndexing:
    def test_add_chunks(self):
        engine = VectorRBACEngine()
        chunks = [
            DocumentChunk(id=0, embedding=[1.0, 0.0, 0.0], clearance=Clearance.PUBLIC),
            DocumentChunk(id=0, embedding=[0.9, 0.1, 0.0], clearance=Clearance.PUBLIC),
        ]
        ids = engine.add_chunks(chunks)
        assert len(ids) == 2
        assert engine.get_status().total_chunks == 2

    def test_edges_created_for_similar_chunks(self):
        engine = VectorRBACEngine(similarity_threshold=0.9)
        chunks = [
            DocumentChunk(id=0, embedding=[1.0, 0.0, 0.0], clearance=Clearance.PUBLIC),
            DocumentChunk(id=0, embedding=[0.99, 0.01, 0.0], clearance=Clearance.PUBLIC),
            DocumentChunk(id=0, embedding=[0.0, 1.0, 0.0], clearance=Clearance.PUBLIC),
        ]
        engine.add_chunks(chunks)
        # First two are very similar, third is orthogonal
        assert engine.get_status().total_edges >= 1

    def test_components_separate_by_clearance(self):
        engine = _build_test_engine()
        status = engine.get_status()
        # Should have at least 2 components (public cluster vs board cluster)
        assert status.betti_numbers["b0"] >= 2


class TestRetrieval:
    def test_public_user_gets_public_chunks(self):
        engine = _build_test_engine()
        # Query near public cluster
        query = np.zeros(16, dtype=np.float32)
        query[0] = 1.0
        result = engine.query(query.tolist(), Clearance.PUBLIC, top_k=5)
        assert not result.blocked
        # All results should be public
        for r in result.results:
            assert r["clearance"] <= Clearance.PUBLIC

    def test_public_user_cannot_get_board_chunks(self):
        engine = _build_test_engine()
        # Query near board cluster but as public user
        query = np.zeros(16, dtype=np.float32)
        query[1] = 1.0
        result = engine.query(query.tolist(), Clearance.PUBLIC, top_k=5)
        # Should either be blocked or return only public chunks
        for r in result.results:
            assert r["clearance"] <= Clearance.PUBLIC

    def test_board_user_gets_all_chunks(self):
        engine = _build_test_engine()
        query = np.zeros(16, dtype=np.float32)
        query[1] = 1.0
        result = engine.query(query.tolist(), Clearance.BOARD, top_k=5)
        assert not result.blocked
        assert len(result.results) > 0

    def test_empty_query_blocked(self):
        engine = _build_test_engine()
        result = engine.query([0.0] * 16, Clearance.PUBLIC, top_k=5)
        assert result.blocked


class TestKappaEffective:
    def test_kappa_is_one_when_no_boundary(self):
        """If no restricted content exists, κ should be 1.0."""
        engine = VectorRBACEngine()
        chunks = [
            DocumentChunk(id=0, embedding=[1.0, 0.0], clearance=Clearance.PUBLIC),
            DocumentChunk(id=0, embedding=[0.9, 0.1], clearance=Clearance.PUBLIC),
        ]
        engine.add_chunks(chunks)
        result = engine.query([1.0, 0.0], Clearance.PUBLIC)
        assert result.kappa_effective == 1.0


class TestMetabolicFuse:
    def test_threshold_escalation(self):
        engine = _build_test_engine()
        # Simulate many probes
        for _ in range(60):
            engine._record_probe()
        # Threshold should escalate
        assert engine._get_adaptive_threshold() > 0.3


class TestAttackSimulation:
    def test_attack_no_leaks(self):
        engine = _build_test_engine()
        result = engine.simulate_attack(
            target_clearance=Clearance.BOARD,
            num_queries=50,
            attacker_clearance=Clearance.PUBLIC,
        )
        assert result.leak_detected is False

    def test_attack_blocks_probes(self):
        engine = _build_test_engine()
        result = engine.simulate_attack(
            target_clearance=Clearance.BOARD,
            num_queries=50,
            attacker_clearance=Clearance.PUBLIC,
        )
        assert result.block_rate >= 0.9
