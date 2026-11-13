"""RAG 客服 Agent 入口。

使用方式：
  1) 启动 Web 服务（浏览器对话 + API）：
       python main.py
     然后浏览器打开 http://127.0.0.1:8000

  2) 终端交互式对话：
       python main.py --cli

启动前请先在项目根目录 .env 文件中填入 OPENAI_API_KEY。
未填 Key 时系统也能运行：意图识别走关键词规则、知识库走离线 TF-IDF 检索。
"""
from __future__ import annotations

import argparse
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import config
from app.agent import chat
from app.knowledge_base import get_knowledge_base

app = FastAPI(title="RAG 客服 Agent", version="1.0.0")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户消息")
    thread_id: Optional[str] = Field(None, description="会话 ID；不传则新建会话")


@app.get("/healthz")
def healthz():
    kb = get_knowledge_base()
    return {
        "status": "ok",
        "llm_configured": bool(config.OPENAI_API_KEY),
        "model": config.CHAT_MODEL if config.OPENAI_API_KEY else None,
        "knowledge_base": kb.stats(),
    }


@app.post("/chat")
def do_chat(req: ChatRequest):
    reply, thread_id = chat(req.message, req.thread_id)
    return {"reply": reply, "thread_id": thread_id}


@app.get("/")
def index():
    return FileResponse(config.ROOT / "static" / "index.html")


def run_cli() -> None:
    print("=" * 60)
    print("  RAG 客服 Agent（终端模式）")
    print(f"  对话模型：{config.CHAT_MODEL if config.OPENAI_API_KEY else '未配置 API Key，使用规则/模板模式'}")
    print("  输入 exit 或 quit 结束对话")
    print("=" * 60)

    thread_id = None
    while True:
        try:
            message = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not message:
            continue
        if message.lower() in ("exit", "quit"):
            print("再见！")
            break
        reply, thread_id = chat(message, thread_id)
        print(f"\n客服：{reply}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 客服 Agent")
    parser.add_argument("--cli", action="store_true", help="使用终端交互式对话（默认启动 Web 服务）")
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        print(f"启动 Web 服务：http://127.0.0.1:{config.PORT}")
        uvicorn.run("main:app", host="127.0.0.1", port=config.PORT, reload=False)
