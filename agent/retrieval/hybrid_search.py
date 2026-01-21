"""
Hybrid search: stage‑1 recall using dense vectors (Qdrant) and
keyword search (Elasticsearch).

This component focuses on broad recall. It gathers potentially
relevant snippets from both engines and returns a merged,
deduplicated list. Final ranking is handled separately.
"""

from typing import List, Dict, Any, Optional
import time
import logging
import requests

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from agent.memory.elasticsearch_service import ElasticsearchService
from agent.config import CONFIG

logger = logging.getLogger(__name__)


class HybridSearch:
    def __init__(self) -> None:
        self.qdrant = QdrantClient(url=CONFIG["qdrant"]["url"])
        self.collection = CONFIG["qdrant"]["collection"]
        self.es = ElasticsearchService()
        self.embedding_url = CONFIG["embedding_service"]["url"]
        self.limit = CONFIG["retrieval"]["hybrid_limit"]

    def run(self, query: str, user_id: str, project_id: str) -> List[Dict[str, Any]]:
        """
        Perform hybrid recall:
        - Dense semantic search (Qdrant)
        - Keyword BM25 search (Elasticsearch)

        Returns a merged, deduplicated list of candidate snippets.
        """
        query = (query or "").strip()
        if len(query) < 2:
            return []

        # Dense embedding
        query_vector = self._safe_embed_query(query)

        # Dense recall (if embedding succeeded)
        if query_vector is not None:
            security_filter = Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="project_id", match=MatchValue(value=project_id)),
                ]
            )
            vector_hits = self._safe_qdrant_search(query_vector, security_filter)
        else:
            vector_hits = []

        # Keyword recall
        keyword_hits = self._safe_keyword_search(query, user_id, project_id)

        merged = self._merge(vector_hits, keyword_hits)

        # Sort by combined RRF score descending for downstream ranking
        merged.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)

        # Enforce a final cap to avoid returning too many candidates
        return merged[: self.limit]

    def _safe_embed_query(self, query: str) -> Optional[List[float]]:
        """
        Call the embedding service with basic retry logic.
        Returns None if all attempts fail.
        """
        payload = {"texts": [f"query: {query}"]}

        for attempt in range(2):
            try:
                response = requests.post(self.embedding_url, json=payload, timeout=5)
                response.raise_for_status()
                data = response.json()
                embeddings = data.get("embeddings")
                if not embeddings or not isinstance(embeddings, list):
                    logger.warning("Embedding service returned unexpected payload: %s", data)
                    return None
                vector = embeddings[0]
                # Ensure vector is a list of floats
                if hasattr(vector, "tolist"):
                    vector = vector.tolist()
                return list(vector)
            except Exception as exc:
                logger.warning("Embedding attempt %d failed: %s", attempt + 1, exc)
                time.sleep(0.2)

        logger.error("Embedding service unavailable after retries for query: %s", query)
        return None

    def _safe_qdrant_search(self, query_vector: List[float], security_filter: Filter):
        """
        Query Qdrant with a vector and a security filter.
        Returns a list of hits in a normalized dict format.
        """
        try:
            raw_hits = self.qdrant.search(
                collection_name=self.collection,
                query_vector=query_vector,
                query_filter=security_filter,
                limit=self.limit,
            )
        except Exception as exc:
            logger.exception("Qdrant search failed: %s", exc)
            return []

        normalized = []
        for hit in raw_hits:
            # qdrant_client may return objects or dicts depending on version
            payload = getattr(hit, "payload", None) or (hit.get("payload") if isinstance(hit, dict) else {})
            if not isinstance(payload, dict):
                payload = {}

            text = payload.get("text") or payload.get("content") or ""
            if not text:
                continue

            normalized.append({
                "text": text,
                "doc_id": payload.get("doc_id"),
                # prefer explicit chunk_id; fall back to batch_start or 0
                "chunk_id": payload.get("chunk_id", payload.get("batch_start", 0)),
                "source": "qdrant",
            })

        return normalized

    def _safe_keyword_search(self, query: str, user_id: str, project_id: str):
        """
        Query Elasticsearch for BM25 keyword matches and return normalized hits.
        """
        try:
            raw_hits = self.es.search(
                query=query,
                user_id=user_id,
                project_id=project_id,
                limit=self.limit,
            )
        except Exception as exc:
            logger.exception("Elasticsearch search failed: %s", exc)
            return []

        normalized = []
        for hit in raw_hits:
            text = hit.get("text")
            if not text:
                continue
            normalized.append({
                "text": text,
                "doc_id": hit.get("doc_id"),
                "chunk_id": hit.get("chunk_id", 0),
                "source": "elasticsearch",
            })

        return normalized

    def _merge(self, vector_hits, keyword_hits) -> List[Dict[str, Any]]:
        """
        Merge and deduplicate results from Qdrant and Elasticsearch.
        Compute an RRF score for each snippet based on its rank in
        each engine. Deduplication is based on the normalized text content.
        """
        unique: Dict[str, Dict[str, Any]] = {}
        k = 60  # RRF constant

        # Assign ranks for Qdrant hits
        for rank, hit in enumerate(vector_hits):
            text = (hit.get("text") or "").strip()
            if not text:
                continue

            rrf_score = 1.0 / (k + rank + 1)

            if text in unique:
                unique[text]["rrf_score"] += rrf_score
                unique[text]["sources"].add("qdrant")
            else:
                unique[text] = {
                    "text": text,
                    "doc_id": hit.get("doc_id"),
                    "chunk_id": hit.get("chunk_id", 0),
                    "rrf_score": rrf_score,
                    "sources": {"qdrant"},
                }

        # Assign ranks for Elasticsearch hits
        for rank, hit in enumerate(keyword_hits):
            text = (hit.get("text") or "").strip()
            if not text:
                continue

            rrf_score = 1.0 / (k + rank + 1)

            if text in unique:
                unique[text]["rrf_score"] += rrf_score
                unique[text]["sources"].add("elasticsearch")
                # Prefer a non-null doc_id/chunk_id if missing
                if not unique[text].get("doc_id") and hit.get("doc_id"):
                    unique[text]["doc_id"] = hit.get("doc_id")
                if unique[text].get("chunk_id", 0) == 0 and hit.get("chunk_id"):
                    unique[text]["chunk_id"] = hit.get("chunk_id")
            else:
                unique[text] = {
                    "text": text,
                    "doc_id": hit.get("doc_id"),
                    "chunk_id": hit.get("chunk_id", 0),
                    "rrf_score": rrf_score,
                    "sources": {"elasticsearch"},
                }

        # Convert sources set to list for JSON-serializable output
        results = []
        for v in unique.values():
            results.append({
                "text": v["text"],
                "doc_id": v.get("doc_id"),
                "chunk_id": v.get("chunk_id"),
                "rrf_score": v.get("rrf_score", 0.0),
                "sources": list(v.get("sources", [])),
            })

        return results
