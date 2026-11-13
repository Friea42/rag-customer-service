"""RAG 知识库检索。

优先使用 OpenAI Embedding 做语义检索；
当未配置 Key、网关不支持 embedding 或网络不可用时，自动降级为离线 TF-IDF（中文按字符 n-gram）。
全程内存计算，不依赖额外的向量数据库。
"""
from __future__ import annotations

import re
import threading
from functools import lru_cache

import numpy as np

from . import config

KNOWLEDGE_PATH = config.ROOT / "data" / "knowledge.md"


def load_chunks() -> list[dict]:
    """把 knowledge.md 按 ## 二级标题切分为知识块（忽略文件顶部一级标题）。"""
    text = KNOWLEDGE_PATH.read_text(encoding="utf-8")
    chunks: list[dict] = []
    for part in re.split(r"(?m)^##\s+", text):
        part = part.strip()
        if not part:
            continue
        lines = part.splitlines()
        title = lines[0].strip().strip("# ").strip()
        body = "\n".join(lines[1:]).strip()
        if not body or body == title:  # 跳过没有正文的标题块（如文档一级标题）
            continue
        chunks.append({"title": title, "content": body})
    return chunks


class KnowledgeBase:
    def __init__(self) -> None:
        self.chunks = load_chunks()
        self._docs = [f"{c['title']}。{c['content']}" for c in self.chunks]
        self.mode = "not-initialized"
        self._lock = threading.Lock()
        self._ready = False

    # ---------------- 构建索引 ----------------
    def _build_openai(self) -> None:
        from langchain_openai import OpenAIEmbeddings

        kwargs = {"model": config.EMBEDDING_MODEL}
        if config.OPENAI_API_KEY:
            kwargs["api_key"] = config.OPENAI_API_KEY
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL

        embed = OpenAIEmbeddings(**kwargs)
        vectors = embed.embed_documents(self._docs)  # 这里才真正发起网络请求
        dense = np.asarray(vectors, dtype=np.float32)
        self._embeddings = embed
        self._dense = dense / np.linalg.norm(dense, axis=1, keepdims=True)
        self.mode = "openai-embedding"

    def _build_tfidf(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        # 中文没有空格分词，按字符 2~4 元组构建 TF-IDF，无需额外分词器
        self._tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self._sparse = self._tfidf.fit_transform(self._docs)
        self.mode = "tfidf-offline"

    def _ensure(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            # 同时配置了 Key 和嵌入模型名时才尝试在线 Embedding；
            # 未配置嵌入模型（如 DeepSeek 无 Embedding 服务）或请求失败时使用离线 TF-IDF
            if config.OPENAI_API_KEY and config.EMBEDDING_MODEL:
                try:
                    self._build_openai()
                except Exception:
                    # Embedding 服务不可用（额度/模型/网络）→ 静默降级
                    self._build_tfidf()
            else:
                self._build_tfidf()
            self._ready = True

    # ---------------- 检索 ----------------
    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        self._ensure()
        if self.mode == "openai-embedding":
            q = np.asarray(self._embeddings.embed_query(query), dtype=np.float32)
            q = q / np.linalg.norm(q)
            scores = self._dense @ q
        else:
            from sklearn.metrics.pairwise import cosine_similarity

            qv = self._tfidf.transform([query])
            scores = cosine_similarity(qv, self._sparse)[0]

        order = np.argsort(scores)[::-1][:k]
        hits = [self.chunks[i] for i in order if float(scores[i]) > 0.02]
        return hits or [self.chunks[0]]

    def stats(self) -> dict:
        self._ensure()
        return {"chunks": len(self.chunks), "retriever": self.mode}


@lru_cache
def get_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase()
