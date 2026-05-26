"""Tests for the embedding module."""

import numpy as np
from vectorrbac.embedder import embed_texts, embed_query, get_embedding_dimension


class TestEmbedder:
    def test_embed_texts_returns_correct_count(self):
        results = embed_texts(["hello", "world", "test"])
        assert len(results) == 3

    def test_embed_texts_correct_dimension(self):
        dim = get_embedding_dimension()
        results = embed_texts(["hello world"])
        assert len(results[0]) == dim

    def test_embeddings_are_normalized(self):
        results = embed_texts(["some text here"])
        norm = np.linalg.norm(results[0])
        assert abs(norm - 1.0) < 0.01

    def test_same_text_same_embedding(self):
        r1 = embed_texts(["identical text"])
        r2 = embed_texts(["identical text"])
        assert np.allclose(r1[0], r2[0], atol=1e-5)

    def test_similar_texts_higher_similarity(self):
        results = embed_texts(["hello world", "hello earth", "quantum physics"])
        sim_similar = np.dot(results[0], results[1])
        sim_different = np.dot(results[0], results[2])
        assert sim_similar > sim_different

    def test_embed_query_returns_vector(self):
        result = embed_query("test query")
        dim = get_embedding_dimension()
        assert len(result) == dim
        assert abs(np.linalg.norm(result) - 1.0) < 0.01
