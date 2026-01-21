"""
Conversation memory backed by Redis with proactive health check
and safe fallback to in‑process memory.
"""

import logging
import redis
from langchain.memory import ConversationBufferWindowMemory, ChatMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from agent.config import CONFIG

logger = logging.getLogger(__name__)

# Module-level Redis client cache to avoid reconnecting on every call
_redis_client = None


def get_conversation_memory(chat_id: str) -> ConversationBufferWindowMemory:
    """
    Returns a conversation memory object backed by Redis.
    If Redis is unavailable, falls back to in‑process memory.
    """
    global _redis_client

    redis_url = CONFIG.get("redis", {}).get("url")
    message_history = None

    try:
        if not redis_url:
            # Keep the original comment and behavior: fallback when Redis not configured
            raise ValueError("Redis URL not configured")

        # Reuse client if already created to reduce connection overhead
        if _redis_client is None:
            # short connect timeout to fail fast if Redis is unreachable
            _redis_client = redis.from_url(redis_url, socket_connect_timeout=2)

        # Proactively verify Redis connectivity; raises redis.exceptions.RedisError on failure
        _redis_client.ping()

        # Redis is healthy → use persistent memory
        message_history = RedisChatMessageHistory(
            url=redis_url,
            session_id=chat_id,
        )

    except (redis.exceptions.RedisError, ValueError) as redis_err:
        # Redis unavailable → fallback to RAM-only memory
        # This prevents the agent from crashing and keeps the LLM responsive.
        logger.warning("Redis unavailable or misconfigured; falling back to in-process memory. %s", redis_err)
        message_history = ChatMessageHistory()

    except Exception as exc:
        # Catch unexpected errors but preserve fallback behavior
        logger.exception("Unexpected error while initializing Redis-backed memory: %s", exc)
        message_history = ChatMessageHistory()

    # Wrap in a windowed memory buffer
    return ConversationBufferWindowMemory(
        memory_key="chat_history",
        chat_memory=message_history,
        return_messages=True,
        k=20,
    )
