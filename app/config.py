"""全局配置：从项目根目录的 .env 文件读取"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# OpenAI 官方或任意 OpenAI 兼容服务
OPENAI_API_KEY: str = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_BASE_URL: str = (os.getenv("OPENAI_BASE_URL") or "").strip()
CHAT_MODEL: str = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
EMBEDDING_MODEL: str = (os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small").strip()
PORT: int = int(os.getenv("PORT") or "8000")
