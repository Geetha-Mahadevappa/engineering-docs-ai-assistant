"""
Conversation memory backed by Redis with proactive health check
and safe fallback to in‑process memory.
"""

import redis
from langchain.memory import ConversationBufferWindowMemory, ChatMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from agent.config import CONFIG


def get_conversation_memory(chat_id: str) -> ConversationBufferWindowMemory:
    """
    Returns a conversation memory object backed by Redis.
    If Redis is unavailable, falls back to in‑process memory.
    """
    redis_url = CONFIG["redis"]["url"]

    try:
        # Proactively verify Redis connectivity
        client = redis.from_url(redis_url, socket_connect_timeout=2)
        client.ping()  # If this fails, fallback is triggered immediately

        # Redis is healthy → use persistent memory
        message_history = RedisChatMessageHistory(
            url=redis_url,
            session_id=chat_id,
        )

    except Exception:
        # Redis unavailable → fallback to RAM-only memory
        # This prevents the agent from crashing and keeps the LLM responsive.
        message_history = ChatMessageHistory()

    # Wrap in a windowed memory buffer
    return ConversationBufferWindowMemory(
        memory_key="chat_history",
        chat_memory=message_history,
        return_messages=True,
        k=20,
    )
