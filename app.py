import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import strawberry
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter

from agent.agents.conversation_agent import run_agent
from agent.tools.upload_agent import upload_document_tool
from agent.config import CONFIG

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Thread pool for running blocking agent code without blocking the event loop
_executor: Optional[ThreadPoolExecutor] = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        max_workers = CONFIG.get("server", {}).get("threadpool_workers", 4)
        _executor = ThreadPoolExecutor(max_workers=max_workers)
    return _executor


# GraphQL Schema
@strawberry.type
class ChatResponse:
    reply: str


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def chat(self, sessionId: str, message: str) -> ChatResponse:
        """
        Synchronous run_agent is executed in a threadpool to avoid blocking the event loop.
        """
        if not sessionId or not isinstance(sessionId, str):
            return ChatResponse(reply="USER_ERROR: Missing or invalid sessionId.")
        if not message or not isinstance(message, str):
            return ChatResponse(reply="USER_ERROR: Missing or invalid message.")

        loop = asyncio.get_event_loop()
        try:
            reply = await loop.run_in_executor(_get_executor(), run_agent, sessionId, message)
            return ChatResponse(reply=reply)
        except Exception as exc:
            logger.exception("Chat mutation failed: %s", exc)
            return ChatResponse(reply="AGENT_ERROR: Failed to process the message.")


schema = strawberry.Schema(mutation=Mutation)
graphql_app = GraphQLRouter(schema)


# FastAPI App
app = FastAPI(title="Agent API")

# Optional CORS configuration (restrict in production)
allowed_origins = CONFIG.get("server", {}).get("cors_allowed_origins", ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    # Ensure executor is created at startup
    _get_executor()
    logger.info("Application startup complete. ThreadPoolExecutor initialized.")


@app.on_event("shutdown")
async def shutdown_event():
    global _executor
    if _executor:
        _executor.shutdown(wait=True)
        _executor = None
        logger.info("ThreadPoolExecutor shut down.")


# GraphQL endpoint
app.include_router(graphql_app, prefix="/graphql")


# Health Check
@app.get("/health")
def health():
    return {"status": "ok"}


# Direct Upload Endpoint
@app.post("/upload")
async def upload(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Accept a file upload and process ingestion in the background.

    - Validates filename and content type
    - Enforces a max size limit (configurable)
    - Uses BackgroundTasks to avoid blocking the request
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Missing file or filename.")

    # Basic allowed content types; extend as needed
    allowed_types = CONFIG.get("upload", {}).get(
        "allowed_content_types",
        [
            "text/plain",
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ],
    )
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    # Read content (be mindful of memory for very large files)
    content = await file.read()

    # Enforce max upload size (bytes)
    max_size = CONFIG.get("upload", {}).get("max_size_bytes", 20 * 1024 * 1024)  # default 20 MB
    if len(content) > max_size:
        raise HTTPException(status_code=413, detail="File too large.")

    # Sanitize filename (basic)
    filename = file.filename.split("/")[-1].split("\\")[-1]

    # Run ingestion in background to keep endpoint responsive
    try:
        if background_tasks is not None:
            background_tasks.add_task(upload_document_tool, filename, content)
            return {"status": "accepted"}
        else:
            # Fallback: run in threadpool if BackgroundTasks not provided
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(_get_executor(), upload_document_tool, filename, content)
            return {"status": "uploaded"}
    except Exception as exc:
        logger.exception("Upload failed for file %s: %s", filename, exc)
        raise HTTPException(status_code=500, detail="Failed to process upload.")
