"""对话模型：未配置 API Key 时返回 None，各节点自动降级为规则/模板回答"""
from __future__ import annotations

from functools import lru_cache

from . import config


@lru_cache
def get_llm():
    """返回 ChatOpenAI 实例（兼容任何 OpenAI 协议网关）；无 Key 时返回 None。"""
    if not config.OPENAI_API_KEY:
        return None
    try:
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": config.CHAT_MODEL,
            "temperature": 0.2,
            "timeout": 45,
            "max_retries": 1,
            "api_key": config.OPENAI_API_KEY,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return ChatOpenAI(**kwargs)
    except Exception:
        return None
