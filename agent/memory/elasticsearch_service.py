"""
ElasticsearchService

Provides a thin wrapper around Elasticsearch for:
- Indexing document chunks for BM25 keyword search
- Querying chunks with user/project isolation
- Ensuring the index exists with correct mappings

This service is used by:
- DocumentMemory (during ingestion)
- HybridSearch (during retrieval)
"""

from elasticsearch import Elasticsearch
from agent.config import CONFIG


class ElasticsearchService:
    def __init__(self):
        es_cfg = CONFIG["elasticsearch"]

        self.index_name = es_cfg["index"]
        self.client = Elasticsearch(es_cfg["url"])

        # Create index if it does not exist
        if not self.client.indices.exists(index=self.index_name):
            self._create_index()

    def _create_index(self):
        """
        Create the Elasticsearch index with appropriate mappings.
        """
        mappings = {
            "mappings": {
                "properties": {
                    "text": {"type": "text"},          # BM25 keyword search
                    "user_id": {"type": "keyword"},    # tenant isolation
                    "project_id": {"type": "keyword"}, # project isolation
                    "doc_id": {"type": "keyword"},     # document grouping
                    "chunk_id": {"type": "integer"},   # chunk ordering
                }
            }
        }

        self.client.indices.create(index=self.index_name, body=mappings)

    def index(self, user_id: str, project_id: str, doc_id: str, chunk_id: int, text: str):
        """
        Index a single text chunk into Elasticsearch.
        """
        body = {
            "text": text,
            "user_id": user_id,
            "project_id": project_id,
            "doc_id": doc_id,
            "chunk_id": chunk_id,
        }

        self.client.index(index=self.index_name, document=body)

    def search(self, query: str, user_id: str, project_id: str, limit: int = 10):
        """
        Perform BM25 keyword search over indexed chunks.
        """
        response = self.client.search(
            index=self.index_name,
            size=limit,
            query={
                "bool": {
                    "must": [
                        {"match": {"text": query}}
                    ],
                    "filter": [
                        {"term": {"user_id": user_id}},
                        {"term": {"project_id": project_id}},
                    ]
                }
            }
        )

        hits = response.get("hits", {}).get("hits", [])

        return [
            {
                "text": hit["_source"]["text"],
                "doc_id": hit["_source"]["doc_id"],
                "chunk_id": hit["_source"]["chunk_id"],
            }
            for hit in hits
        ]