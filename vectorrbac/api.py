"""FastAPI application for VectorRBAC.

Exposes the topological access control engine as REST endpoints.
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .embedder import embed_query, embed_texts, get_embedding_dimension, is_real_model_available
from .engine import VectorRBACEngine
from .models import AttackResult, Clearance, DocumentChunk, RetrievalResult, SystemStatus

app = FastAPI(
    title="VectorRBAC - Topological Access Control",
    description="Preventing latent-space privilege escalation in RAG systems via simplicial homology",
    version="6.7.0",
)

# Engine instance
_engine = VectorRBACEngine(similarity_threshold=0.65)


class IndexRequest(BaseModel):
    chunks: list[dict] = Field(..., description="List of {text, clearance, metadata} objects")


class IndexResponse(BaseModel):
    status: str = "success"
    chunk_ids: list[int] = Field(default_factory=list)
    betti_numbers: dict[str, int] = Field(default_factory=dict)
    embedding_mode: str = ""


class QueryRequest(BaseModel):
    query_text: str
    user_clearance: int = Field(ge=0, le=3)
    top_k: int = 10


class AttackRequest(BaseModel):
    target_clearance: int = Field(ge=0, le=3)
    num_queries: int = 100
    attacker_clearance: int = Field(ge=0, le=3, default=0)


@app.post("/index", response_model=IndexResponse)
async def index_chunks(request: IndexRequest) -> IndexResponse:
    """Add document chunks to the topological complex.

    Embeds text using sentence-transformers (or fallback), builds
    similarity edges and clique triangles, computes components.
    """
    texts = [c.get("text", "") for c in request.chunks]
    clearances = [Clearance(c.get("clearance", 0)) for c in request.chunks]
    metadatas = [c.get("metadata", {}) for c in request.chunks]

    # Embed all texts
    embeddings = embed_texts(texts)

    # Create DocumentChunk objects
    chunks = []
    for i, (text, emb, clr, meta) in enumerate(zip(texts, embeddings, clearances, metadatas)):
        chunks.append(DocumentChunk(
            id=0,
            text=text,
            embedding=emb,
            clearance=clr,
            metadata=meta,
        ))

    # Add to engine
    ids = _engine.add_chunks(chunks)
    status = _engine.get_status()

    return IndexResponse(
        status="success",
        chunk_ids=ids,
        betti_numbers=status.betti_numbers,
        embedding_mode="sentence-transformers" if is_real_model_available() else "synthetic",
    )


@app.post("/query", response_model=RetrievalResult)
async def query(request: QueryRequest) -> RetrievalResult:
    """Execute a secure retrieval query with topological access control."""
    # Embed the query
    query_emb = embed_query(request.query_text)

    # Execute retrieval
    result = _engine.query(
        query_embedding=query_emb,
        user_clearance=Clearance(request.user_clearance),
        top_k=request.top_k,
    )

    return result


@app.get("/status", response_model=SystemStatus)
async def get_status() -> SystemStatus:
    """Get current system coherence invariants."""
    return _engine.get_status()


@app.post("/simulate_attack", response_model=AttackResult)
async def simulate_attack(request: AttackRequest) -> AttackResult:
    """Simulate a boundary probing attack."""
    return _engine.simulate_attack(
        target_clearance=Clearance(request.target_clearance),
        num_queries=request.num_queries,
        attacker_clearance=Clearance(request.attacker_clearance),
    )


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "version": "6.7.0",
        "embedding_model": "all-MiniLM-L6-v2" if is_real_model_available() else "synthetic-fallback",
        "embedding_dim": get_embedding_dimension(),
    }
