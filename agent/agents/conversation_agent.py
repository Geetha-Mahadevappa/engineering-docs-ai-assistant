from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

from agent.memory.conversation_memory import get_conversation_memory
from agent.tools.upload_agent import upload_document_tool
from agent.tools.search_documents import search_documents_tool
from agent.config import CONFIG

_agent_cache = {}


def build_agent(chat_id: str):
    if chat_id in _agent_cache:
        return _agent_cache[chat_id]

    llm_config = CONFIG["llm"]
    provider = llm_config["provider"]
    model = llm_config["model"]
    base_url = llm_config["base_url"]
    temperature = llm_config.get("temperature", 0.1)

    llm = None

    # Choose backend based on provider
    if provider == "ollama":
        try:
            llm = ChatOllama(
                model=model,
                temperature=temperature,
                base_url=base_url
            )
            llm.invoke("ping")  # health check
        except Exception:
            return None

    elif provider == "vllm":
        try:
            llm = ChatOpenAI(
                model=model,
                openai_api_key="empty",
                openai_api_base=base_url,
                temperature=temperature,
            )
            llm.invoke("ping")
        except Exception:
            return None

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

    tools = [
        upload_document_tool,
        search_documents_tool,
    ]

    memory = get_conversation_memory(chat_id)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
    )

    _agent_cache[chat_id] = agent
    return agent


def run_agent(chat_id: str, query: str) -> str:
    agent = build_agent(chat_id)

    if agent is None:
        return "The AI model is temporarily unavailable. Please try again later."

    return agent.run(query)
