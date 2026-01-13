"""
Document ingestion agent.
Handles:
- Chunk ingestion
- Dense (E5) embedding
- Sparse (BM25) embedding
- Batch upsert into Qdrant
"""

import uuid
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SparseVector,
)
from agent.config import CONFIG


class DocumentMemory:
    def __init__(self, dense_model, sparse_encoder):
        """
        dense_model:
            Your fine‑tuned E5 model. Must expose .encode(list[str]).
        sparse_encoder:
            BM25 encoder with:
                - encode_documents(list[str])
        """
        self.dense_model = dense_model
        self.sparse_encoder = sparse_encoder

        self.collection_name = CONFIG["qdrant"]["collection"]
        self.client = QdrantClient(url=CONFIG["qdrant"]["url"])

        embedding_dim = CONFIG["embedding"]["dimension"]

        # Create collection if missing
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": {}
            }
        )

    # Batch Upsert for Document Ingestion
    def add_document(self, text_chunks: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode text chunks using dense + sparse models and store them in Qdrant.
        Returns a structured status dict for safe handling by the upload agent.
        """
        try:
            if not text_chunks:
                return {"status": "error", "message": "No text chunks provided."}

            # Dense embeddings
            prefixed = [f"passage: {chunk}" for chunk in text_chunks]
            dense_vectors = self.dense_model.encode(prefixed)

            # Sparse embeddings
            sparse_vectors = self.sparse_encoder.encode_documents(text_chunks)

            points = []
            for i, dense_vec in enumerate(dense_vectors):
                payload = {
                    "text": text_chunks[i],
                    **metadata,
                }

                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=dense_vec.tolist(),
                    payload=payload,
                    sparse_vector=SparseVector(
                        indices=sparse_vectors[i]["indices"],
                        values=sparse_vectors[i]["values"],
                    ),
                )

                points.append(point)

            # Batch upsert
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

            return {"status": "success"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

