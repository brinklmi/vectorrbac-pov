"""Domain models for VectorRBAC.

Maps vector database concepts to simplicial topology:
- DocumentChunk = 0-simplex with embedding + clearance
- Semantic similarity edge = 1-simplex (cosine sim > θ)
- Clique triangle = 2-simplex (potential emergent super-permission)
"""

from __future__ import annotations

import time
from enum import IntEnum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field


class Clearance(IntEnum):
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    BOARD = 3


class DocumentChunk(BaseModel):
    """A document chunk with embedding and clearance metadata."""

    id: int
    text: str = ""
    embedding: list[float] = Field(default_factory=list)
    clearance: Clearance = Clearance.PUBLIC
    source_doc: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class QueryContext(BaseModel):
    """Context for a retrieval query with access control."""

    query_id: str = ""
    embedding: list[float] = Field(default_factory=list)
    user_clearance: Clearance = Clearance.PUBLIC
    timestamp: float = Field(default_factory=time.time)
    nearest_vertex_id: Optional[int] = None
    component_id: int = -1
    kappa_effective: float = 1.0
    allowed: bool = True
    blocked_reason: str = ""


class RetrievalResult(BaseModel):
    """Result of a secure retrieval query."""

    results: list[dict[str, Any]] = Field(default_factory=list)
    kappa_effective: float = 1.0
    component_id: int = -1
    blocked: bool = False
    blocked_reason: str = ""
    cup_product_filtered: int = 0


class SystemStatus(BaseModel):
    """System coherence invariants."""

    betti_numbers: dict[str, int] = Field(default_factory=lambda: {"b0": 0, "b1": 0, "b2": 0})
    kappa_effective_threshold: float = 0.3
    probe_rate: float = 0.0
    total_chunks: int = 0
    total_edges: int = 0
    total_triangles: int = 0


class AttackResult(BaseModel):
    """Result of a simulated boundary probing attack."""

    total_queries: int = 0
    blocked_count: int = 0
    avg_kappa: float = 0.0
    leak_detected: bool = False
    block_rate: float = 0.0
