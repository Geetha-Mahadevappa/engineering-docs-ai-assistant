"""
Upload agent for ingesting documents into the vector store.
Handles PDF, text, and markdown files.
"""

import os
import datetime
from langchain_core.tools import tool

from agent.utils.file_loader import load_text_from_file
from agent.utils.chunking import chunk_text
from agent.memory.document_memory import DocumentMemory
from agent.config import CONFIG


@tool("upload_document")
def upload_document_tool(file_path: str, current_user_id: str = None, current_project_id: str = None) -> str:
    """
    Ingest a document into the vector store.
    file_path: path to the document on disk.
    """
    try:
        # Basic validation
        if not file_path or not isinstance(file_path, str):
            return "USER_ERROR: Invalid file path. Please provide a valid file path string."

        if not os.path.exists(file_path):
            return "USER_ERROR: File not found. Please upload a valid document."

        # Load raw text
        try:
            text = load_text_from_file(file_path)
        except Exception as e:
            return (
                "USER_ERROR: The document appears to be corrupted or unreadable. "
                f"Details: {str(e)}"
            )

        if not text or not text.strip():
            return "USER_ERROR: Unsupported file format or empty document."

        # Chunk the document
        chunks = chunk_text(text)
        if not chunks:
            return "USER_ERROR: Document contains no readable text."

        # Configurable batch size
        ingestion_config = CONFIG.get("ingestion", {})
        batch_size = ingestion_config.get("batch_size", 500)

        doc_id = os.path.basename(file_path)
        file_ext = os.path.splitext(doc_id)[1].lower()
        memory = DocumentMemory()

        total_chunks = len(chunks)

        # Batch ingestion loop
        for start in range(0, total_chunks, batch_size):
            end = min(start + batch_size, total_chunks)
            batch_chunks = chunks[start:end]

            metadata = {
                "doc_id": doc_id,
                "batch_start": start,
                "batch_end": end - 1,
                "total_chunks": total_chunks,
                "file_type": file_ext,
                "ingested_at": datetime.now(datetime.timezone.utc).isoformat(),
                "user_id": current_user_id,  # passed from conversation agent
                "project_id": current_project_id,  # optional
            }

            # DocumentMemory handles embedding + upsert and returns a status
            result = memory.add_document(text_chunks=batch_chunks, metadata=metadata)

            # Expecting a structured result for robustness
            if isinstance(result, dict) and result.get("status") == "error":
                message = result.get("message", "Unknown ingestion error.")
                return f"INGESTION_ERROR: Failed to ingest document chunks. Details: {message}"

        return f"Document '{doc_id}' uploaded successfully with {total_chunks} chunks."

    except FileNotFoundError:
        # Extra safety, though we already check os.path.exists
        return "USER_ERROR: File not found. Please upload a valid document."

    except ValueError as e:
        # For explicit value-related issues in helpers, if any
        return f"USER_ERROR: {str(e)}"

    except ConnectionError as e:
        # For Qdrant / vector DB / network issues raised from DocumentMemory
        return (
            "INGESTION_ERROR: Failed to connect to the vector database during upload. "
            f"Details: {str(e)}"
        )

    except Exception as e:
        # Explicitly categorize unexpected failures for the LLM
        return (
            "CRITICAL_SYSTEM_ERROR: The ingestion service encountered an unexpected failure. "
            f"Error details: {str(e)}"
        )
