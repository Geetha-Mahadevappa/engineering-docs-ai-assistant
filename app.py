import strawberry
from fastapi import FastAPI, UploadFile, File
from strawberry.fastapi import GraphQLRouter

from agent.agents.conversation_agent import run_agent
from agent.tools.upload_agent import upload_document_tool


# GraphQL Schema
@strawberry.type
class ChatResponse:
    reply: str


@strawberry.type
class Mutation:
    @strawberry.mutation
    def chat(self, sessionId: str, message: str) -> ChatResponse:
        reply = run_agent(sessionId, message)
        return ChatResponse(reply=reply)


schema = strawberry.Schema(mutation=Mutation)
graphql_app = GraphQLRouter(schema)


# FastAPI App
app = FastAPI()

# GraphQL endpoint
app.include_router(graphql_app, prefix="/graphql")


# Health Check
@app.get("/health")
def health():
    return {"status": "ok"}


# Direct Upload Endpoint
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    upload_document_tool(file.filename, content)
    return {"status": "uploaded"}
