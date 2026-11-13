# RAG Customer Service Agent

A production-ready customer service chatbot built with **Python, LangGraph, LangChain, and FastAPI**.
It combines **RAG (Retrieval-Augmented Generation)** for policy Q&A, **intent routing** for multi-turn
business flows (refunds, logistics), and **tool calling** for real order operations — all without
requiring any external vector database.

![langgraph](https://img.shields.io/badge/langgraph-%E2%9A%A1%EF%B8%8F%20Workflow-blue)
![fastapi](https://img.shields.io/badge/fastapi-0.141-blue)
![python](https://img.shields.io/badge/python-3.12-green)

---

## Features

- **Intent Routing with LangGraph** — The workflow graph first classifies every user message into one
  of three intents (`refund`, `logistics`, `faq`) and dispatches it to the corresponding node.
- **Multi-turn Business Flows** — Both refund and logistics nodes collect required slots (order ID,
  refund reason) across multiple conversation turns before calling tools, with the conversation
  state preserved per `thread_id`.
- **RAG Knowledge Base** — A lightweight RAG system backs the FAQ path. It first attempts semantic
  retrieval via OpenAI Embeddings; if unavailable (no API key, gateway doesn't support embeddings,
  or network issues), it **automatically falls back to an offline TF-IDF retriever** built with
  `scikit-learn` — no extra infrastructure required.
- **Mock Tool Functions** — `query_order(order_id)` and `request_refund(order_id, reason)` simulate
  a real backend. Swap them for HTTP/RPC calls to your order service without touching the workflow.
- **LLM with Graceful Degradation** — If no API key is configured the whole system still runs:
  intent classification falls back to keyword rules, replies use templates, and RAG uses TF-IDF.
- **FastAPI + Web Chat UI** — Exposes a `/chat` endpoint plus a single-page browser chat interface
  served at `GET /`.

---

## Project Structure

```
rag-customer-service/
├── main.py                 # FastAPI entry point (Web + --cli mode)
├── requirements.txt
├── .env                    # 👈 contains your API key — DO NOT commit
├── .env.example            # template
├── .gitignore
├── app/
│   ├── agent.py            # LangGraph workflow definition
│   ├── config.py            # dotenv loader
│   ├── llm.py               # ChatOpenAI wrapper (any OpenAI-compatible gateway)
│   ├── knowledge_base.py    # RAG retriever: Embedding → TF-IDF fallback
│   └── tools.py             # query_order() / request_refund()
├── data/
│   └── knowledge.md         # Knowledge base (edit this file to add policies)
└── static/
    └── index.html           # Browser chat UI
```

---

## How the Workflow Works

```
              ┌──────────────┐
  user msg ──▶│   intent      │  classify → refund / logistics / faq
              └──────┬───────┘
                     │  conditional edges
           ┌─────────┼──────────┐
           ▼         ▼          ▼
      ┌────────┐ ┌────────┐ ┌──────┐
      │ refund │ │logistics│ │ faq  │
      │ node   │ │  node  │ │ node │
      └────┬───┘ └───┬────┘ └──┬───┘
           │         │         │
           ▼         ▼         ▼
      call tool   call tool  RAG + LLM
      refund()    order()    (or TF-IDF)
           │         │         │
           └─────────┴─────────┘
                     ▼
                    END
```

Missing slots trigger an **end-of-turn wait**: the node replies with a clarifying question and sets
`pending` in the graph state, then the next turn resumes from `intent` to continue collecting.

---

## Local Installation & Run

### 1. Clone & create virtual environment

```powershell
git clone https://github.com/<your-username>/rag-customer-service.git
cd rag-customer-service

python -m venv venv
.\venv\Scripts\Activate.ps1     # Windows PowerShell
# source venv/bin/activate      # macOS / Linux
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure API key

Copy `.env.example` to `.env` and fill in your own values:

```ini
# Required — any OpenAI-compatible provider works (OpenAI, DeepSeek, Qwen, etc.)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://api.deepseek.com          # leave blank for OpenAI official
OPENAI_MODEL=deepseek-chat
EMBEDDING_MODEL=                                  # optional; leave blank → offline TF-IDF
PORT=8000
```

> 💡 If you skip the API key entirely, the system runs in rule/template mode — useful for demos.

### 4. Start the web server

```powershell
python main.py
# → http://127.0.0.1:8000
```

Or run the terminal-only CLI mode:

```powershell
python main.py --cli
```

### 5. API endpoints

| Method | Path         | Purpose                                             |
| ------ | ------------ | --------------------------------------------------- |
| GET    | `/healthz`   | Service + LLM + KB status (great for smoke checks)  |
| GET    | `/`          | Browser chat UI                                    |
| POST   | `/chat`      | `{"message": "...", "thread_id": "optional"}`       |

---

## Customization Notes

- **Add your own knowledge** → Edit `data/knowledge.md`. The retriever splits on `## Section` headings
  automatically; no rebuild step required.
- **Swap mock tools for real ones** → Replace `app/tools.py` with HTTP/RPC calls to your backend.
  The LangGraph workflow above doesn't change.
- **Use a different LLM provider** → Set `OPENAI_BASE_URL` and `OPENAI_MODEL` to your provider's
  values (OpenAI-compatible gateways all work — DeepSeek, Qwen, Zhipu, Moonshot, etc.).
- **Production RAG** → For a large knowledge base, plug `langchain-chroma` or
  `langchain-pinecone` into `app/knowledge_base.py`. The current in-memory TF-IDF fallback
  is great for small demos but scales to millions of chunks with a vector DB.

---

## Requirements

Python 3.10+ tested on 3.12.7
- `langgraph >= 0.6`
- `langchain >= 0.3`
- `langchain-openai >= 0.2`
- `fastapi >= 0.115`
- `uvicorn[standard] >= 0.30`
- `python-dotenv >= 1.0`
- `numpy >= 1.26`, `scikit-learn >= 1.5` (for TF-IDF fallback)

---

## License

MIT — feel free to use as a portfolio project or starting point.
