"""
Search tool for retrieving relevant document chunks.
This tool delegates to the Retrieval Agent, which performs:
- Hybrid recall (BM25 + Dense)
- Cross-Encoder reranking
- Top‑K selection
"""

from langchain_core.tools import tool
from agent.retrieval.retrieval_agent import RetrievalAgent


# Initialize retrieval agent once (it loads models, encoders, etc.)
retrieval_agent = RetrievalAgent()


@tool("search_documents")
def search_documents_tool(query: str, current_user_id: str = None, current_project_id: str = None, top_k: int = 5) -> str:
    """
    Search for relevant documentation.
    query: user query string
    current_user_id: multi-tenant isolation
    current_project_id: project-level isolation
    top_k: number of results to return
    """
    try:
        if not query or not isinstance(query, str):
            return "USER_ERROR: Please provide a valid search query."

        if not current_user_id:
            return "USER_ERROR: Missing user_id for retrieval."

        if not current_project_id:
            return "USER_ERROR: Missing project_id for retrieval."

        # Perform hybrid search + reranking
        results = retrieval_agent.search(
            query=query,
            user_id=current_user_id,
            project_id=current_project_id,
            top_k=top_k,
        )

        if not results:
            return "No relevant documents found."

        # Format results for the Conversation Agent
        formatted = []
        for r in results:
            formatted.append(
                f"[Score: {r['score']:.4f}] {r['text']} "
                f"(doc_id={r['doc_id']}, chunk={r['chunk_id']})"
            )

        return "\n\n".join(formatted)

    except Exception as e:
        return f"RETRIEVAL_ERROR: Failed to search documents. Details: {str(e)}"
