"""
Document ingestion agent.
Handles:
- Chunk ingestion
- Dense (E5) embedding
- Elasticsearch keyword indexing
- Batch upsert into Qdrant
"""

import hashlib
import uuid
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from agent.config import CONFIG
from agent.memory.elasticsearch_service import ElasticsearchService


class DocumentMemory:
    def __init__(self, dense_model):
        """
        dense_model:
            Your fine‑tuned E5 model. Must expose .encode(list[str]).
        """
        self.dense_model = dense_model

        self.collection_name = CONFIG["qdrant"]["collection"]
        self.client = QdrantClient(url=CONFIG["qdrant"]["url"])
        self.es = ElasticsearchService()

        embedding_dim = CONFIG["embedding"]["dimension"]

        # Create collection ONLY if missing
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )

    # Batch Upsert for Document Ingestion
    def add_document(self, text_chunks: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode text chunks using dense model and store them in Qdrant + Elasticsearch.
        """
        try:
            if not text_chunks:
                return {"status": "error", "message": "No text chunks provided."}

            required_keys = ["user_id", "project_id", "doc_id"]
            missing = [k for k in required_keys if k not in metadata]

            if missing:
                return {
                    "status": "error",
                    "message": f"Missing required metadata keys: {missing}"
                }

            # Validate metadata values
            for k in required_keys:
                v = metadata.get(k)
                if not isinstance(v, str) or not v.strip():
                    return {
                        "status": "error",
                        "message": f"Invalid metadata value for {k}: must be a non-empty string."
                    }

            # Dense embeddings
            prefixed = [f"passage: {chunk}" for chunk in text_chunks]
            dense_vectors = self.dense_model.encode(prefixed)

            # Ensure embeddings length matches chunks
            if len(dense_vectors) != len(text_chunks):
                return {
                    "status": "error",
                    "message": "Embedding model returned a different number of vectors than input chunks."
                }

            embedding_dim = CONFIG["embedding"]["dimension"]

            points = []
            es_docs = []

            for i, dense_vec in enumerate(dense_vectors):
                # Safely convert embedding to plain Python list
                if hasattr(dense_vec, "tolist"):
                    vec_list = dense_vec.tolist()
                else:
                    vec_list = list(dense_vec)

                # Validate embedding dimension
                if len(vec_list) != embedding_dim:
                    return {
                        "status": "error",
                        "message": f"Embedding dimension mismatch for chunk {i}: expected {embedding_dim}, got {len(vec_list)}"
                    }

                chunk_text = text_chunks[i]
                payload = {
                    "text": chunk_text,
                    **metadata,
                    "chunk_id": i,
                }

                # Prepare ES doc for bulk indexing
                es_docs.append({
                    "user_id": metadata["user_id"],
                    "project_id": metadata["project_id"],
                    "doc_id": metadata["doc_id"],
                    "chunk_id": i,
                    "text": chunk_text,
                })

                # Use deterministic id for idempotency: hash(doc_id + chunk_id)
                deterministic_id = hashlib.sha256(f"{metadata['doc_id']}::{i}".encode("utf-8")).hexdigest()

                # Prepare Qdrant point
                point = PointStruct(
                    id=deterministic_id,
                    vector=vec_list,
                    payload=payload,
                )

                points.append(point)

            # Bulk index into Elasticsearch (use bulk_index for performance)
            try:
                if hasattr(self.es, "bulk_index"):
                    self.es.bulk_index(es_docs)
                else:
                    for d in es_docs:
                        self.es.index(
                            user_id=d["user_id"],
                            project_id=d["project_id"],
                            doc_id=d["doc_id"],
                            chunk_id=d["chunk_id"],
                            text=d["text"],
                        )
            except Exception as es_exc:
                return {"status": "error", "message": f"elasticsearch_index_error: {str(es_exc)}"}

            # Batch upsert into Qdrant
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )
            except Exception as q_exc:
                return {"status": "error", "message": f"qdrant_upsert_error: {str(q_exc)}"}

            return {"status": "success"}

        except Exception as e:
            return {"status": "error", "message": str(e)}
