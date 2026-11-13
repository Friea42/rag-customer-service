# RAG 客服 Agent 项目搭建

> **Workspace:** c:\Users\yangd\Desktop\agenttrae

---

**User**

这是一个带 RAG 知识库问答能力的客服 Agent，使用 Python 和 LangGraph 框架。

用 LangGraph 定义流程：意图识别节点（判断用户是想“退货”还是“查询物流”），然后路由到对应的处理节点。

支持调用模拟工具函数：query_order(id) 和 request_refund(id, reason)。

环境配置（重点）：请先帮我在项目根目录创建虚拟环境（Windows 用 python -m venv venv），并生成 requirements.txt（包含 langgraph, langchain, langchain-openai, fastapi, uvicorn 等依赖）。

执行流程：请按顺序执行：

创建虚拟环境 venv

激活虚拟环境

在虚拟环境内执行 pip install -r requirements.txt

主程序：写一个 main.py，启动一个简单的 FastAPI 服务，带一个对话接口，我只需要填入 API Key 就能在浏览器或终端里跟客服 Agent 对话。

**Assistant**

我来帮你搭建这个带 RAG 的客服 Agent 项目。首先让我查看当前工作区状态，并调取相关历史经验。

工作区是空的，我将从零搭建整个项目。历史经验提示：Windows 环境下要使用预编译 wheel 的依赖、LangGraph 版本 API 兼容处理、结构化输出走普通 JSON 解析以兼容各种 OpenAI 网关。我先制定任务计划：


现在创建所有项目文件，同时检查 Python 版本：


Python 3.12.7，依赖兼容性很好。继续写 LangGraph 工作流、FastAPI 入口和浏览器对话页面：


代码文件全部就绪。现在创建虚拟环境：


虚拟环境创建成功。现在激活虚拟环境（先尝试 PowerShell 激活脚本）：


激活成功。注意 Shell 环境不跨命令保留，所以我在同一条命令里激活并安装依赖（先升级 pip，再安装 requirements.txt）：


全部依赖安装成功（langgraph 1.2.12、langchain 1.4.2、fastapi 0.141.1 等）。现在进行冒烟测试，验证 Graph 编译、意图路由、多轮退货流程、物流工具调用和 TF-IDF 降级检索：


冒烟测试基本通过，但发现一个路由问题：**"七天无理由退货规则是什么"被规则分类器误判为退货意图**（因为含"退货"二字），实际上这是政策咨询应走知识库。我来修复规则分类器——政策/咨询类问法优先判定为 FAQ：


重新运行冒烟测试，并补充几个意图边界用例：


还剩一个边界："你们发什么快递"是问承运商政策，应走 FAQ。补充信号词后快速验证：


全部路由用例通过。现在后台启动 FastAPI 服务并验证 HTTP 接口：


服务已在后台启动，等待就绪后测试接口：


接口已通，但暴露两个问题：① PowerShell 测试客户端发送中文 JSON 时编码被破坏（导致服务端收到乱码、意图落入 FAQ）；② 知识库把文件顶部的一级标题"客服知识库"也切成了一个空知识块。先修复知识库解析，再用 Python 客户端（正确 UTF-8）重测：


重启服务以加载修复后的知识库：


用 Python 客户端（正确的 UTF-8 编码）完整测试 HTTP API，包括多轮会话：


全部测试通过。服务当前正在后台运行，打开浏览器预览页面：


项目已全部搭建完成并验证通过，服务正在 **http://127.0.0.1:8000** 运行，浏览器已打开对话页面。

## 完成情况

**环境（已按顺序执行完毕）**
1. ✅ `python -m venv venv` 创建虚拟环境（Python 3.12.7）
2. ✅ `.\venv\Scripts\Activate.ps1` 激活成功
3. ✅ 虚拟环境内安装全部依赖：langgraph 1.2.12、langchain 1.4.2、langchain-openai 1.6.5、fastapi 0.141.1、uvicorn 0.53、scikit-learn 1.9.1 等

**项目结构**
```
agenttrae/
├── main.py                  # FastAPI 入口（Web 服务 + --cli 终端模式）
├── requirements.txt
├── .env                     # 👉 只需在这里填入 OPENAI_API_KEY
├── data/knowledge.md        # 客服知识库（11 条政策，可自行增删）
├── static/index.html        # 浏览器对话页面（白色简约风）
└── app/
    ├── config.py            # 读取 .env 配置
    ├── llm.py               # ChatOpenAI（兼容任意 OpenAI 网关）
    ├── tools.py             # 模拟工具 query_order() / request_refund()
    ├── knowledge_base.py    # RAG：OpenAI Embedding → 失败自动降级离线 TF-IDF
    └── agent.py             # LangGraph 工作流
```

**LangGraph 流程**：`START → intent 意图识别 → 条件路由 → refund / logistics / faq → END`
- **退货**：多轮收集订单号 → 退货原因 → 调用 `request_refund(id, reason)`
- **物流**：收集订单号 → 调用 `query_order(id)`
- **知识库**：RAG 检索政策文档后生成回答
- 通过内存 Checkpointer 按 `thread_id` 保持多轮上下文

**已验证的测试**：Graph 编译、意图路由边界用例（"退货规则是什么"正确走 FAQ 而非退货）、物流查询、退货三轮对话、重复申请拦截、TF-IDF 降级检索、HTTP 接口与前端页面均正常。

## 使用方式

服务当前以**无 Key 模式**运行（意图走关键词规则、知识库走离线 TF-IDF，可直接试用：查 `A1001` 物流、说"我要退货"）。填入 API Key 获得完整 LLM 能力：

1. 编辑 `.env`，填入 `OPENAI_API_KEY`（如用 DeepSeek 等兼容网关，同时填 `OPENAI_BASE_URL` 并改 `OPENAI_MODEL`，文件内有示例）
2. 重启服务：
   ```powershell
   .\venv\Scripts\Activate.ps1
   python main.py            # Web 模式：http://127.0.0.1:8000
   python main.py --cli      # 或终端对话模式
   ```
3. 接口：`POST /chat`，JSON 入参 `{"message": "...", "thread_id": "可选"}`；`GET /healthz` 可查看 LLM 与检索器状态

后续接入真实业务时，只需替换 `app/tools.py` 中两个函数为真实订单系统 API，工作流无需改动；知识库内容直接在 `data/knowledge.md` 中按 `## 标题 + 正文` 格式维护即可。

---

**User**

请帮我配置 .env 文件填入 API Key

**Assistant**

我先调取相关历史经验，并查看当前 `.env` 文件状态：


要配置 `.env`，我需要你提供两项信息。**API Key 是敏感凭证，请务必使用你自己的 Key**——我不会凭空生成或复用任何历史 Key。先选择你使用的服务商：


你选择了 **DeepSeek**，收到。现在只差最后一步——请把你的 **DeepSeek API Key 粘贴发送给我**（格式形如 `sk-xxxxxxxx...`）。

拿到 Key 后我会帮你完成以下配置：

```env
OPENAI_API_KEY=sk-你的真实Key
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
EMBEDDING_MODEL=text-embedding-3-small
```

两点安全提示：
1. Key 只会写入你本地项目根目录的 `.env` 文件，我不会在终端回显完整 Key（输出中只显示 `sk-****` 脱敏形式）。
2. 如果项目后续要用 Git 管理，我会顺带把 `.env` 加进 `.gitignore`，防止密钥进入版本库。

请粘贴 Key 即可 👇

---

**User**

sk-74c37ae1643e458899ad0c0b4193f16a

**Assistant**

收到 Key。我先调取本轮相关经验，然后把配置写入 `.env`：


现在写入配置。注意一个关键点：**DeepSeek 不提供 Embedding 接口**，所以知识库检索直接使用离线 TF-IDF（项目已内置），我会把 `EMBEDDING_MODEL` 留空并让代码据此跳过必定失败的 embedding 请求：


配置已写入。现在用最小调用验证 Key 是否有效（仅输出脱敏信息）：


The key is valid (authenticated successfully — got past auth), but the DeepSeek account has insufficient balance (402 Insufficient Balance). The key itself works, but the account needs充值.

Let me report this clearly. Options:
1. User tops up DeepSeek balance (platform.deepseek.com → 充值), minimum 10 yuan typically, then it works without any code change.
2. Meanwhile, the service still runs in offline mode (rule-based intent + TF-IDF), fully functional.

Should I restart the server? The currently running server (job-945...) was started with old config (no key). Even with 402, the code gracefully catches LLM exceptions and falls back to rules — so behavior would be: tries LLM, fails on each LLM call (intent classify → rule fallback; reply → template fallback). That adds latency (~timeout? No, 402 returns immediately, and max_retries=1). It works, but each message does one failing call. Until balance topped up, better to keep server running without key or restart anyway? If I restart with the key, each chat attempt incurs a quick 402 and falls back — function still works, slight delay. Since user asked just to configure, and verification revealed balance issue, I should report and let them top up. I should restart the server with new config anyway so once they top up, it just works (config is read at process start; get_llm is lru_cached). Actually llm is instantiated at first call and cached; with the key configured, after top-up it would work even without restart since the client object doesn't validate at construction. The running OLD server process has config loaded with empty key (lru_cached get_llm returned None forever). So a restart IS needed for the running web service to pick up the key. I'll restart it now — that completes "配置生效". Then do a quick healthz to show llm_configured: true.

Let me restart the background server and verify healthz.

Key 认证有效（返回 402 业务错误而非 401 认证失败），只是账户余额不足。重启正在运行的服务让配置生效，并确认状态：


配置完成 ✅ 服务已重启并加载新配置（`llm_configured: true`，模型 `deepseek-chat`）。

## 已完成的配置

| 配置项 | 值 |
|---|---|
| `OPENAI_API_KEY` | `sk-****f16a`（已脱敏显示） |
| `OPENAI_BASE_URL` | `https://api.deepseek.com` |
| `OPENAI_MODEL` | `deepseek-chat` |
| `EMBEDDING_MODEL` | 留空（DeepSeek 无 Embedding 服务，知识库自动使用内置离线 TF-IDF 检索，已同步优化代码跳过无效请求） |

另外创建了 `.gitignore` 并把 `.env` 加入忽略，防止密钥进入 Git 版本库。

## ⚠️ 验证发现：账户余额不足

Key 本身**认证通过**（不是无效 Key），但 DeepSeek 返回了业务错误：

```
402 Insufficient Balance（账户余额不足）
```

你需要先充值才能使用大模型对话：
1. 登录 [platform.deepseek.com](https://platform.deepseek.com) → 左侧「充值」
2. 充值后**无需改代码、无需重启**，直接在浏览器 http://127.0.0.1:8000 对话即可自动生效

充值前系统也能正常用——每次调用会快速失败并自动降级：意图识别走关键词规则、回复走模板、知识库走 TF-IDF（就是你刚才测试过的离线模式）。

## 安全提示

- 你的 Key 刚才以明文出现在了对话中，如果这个会话可能被他人看到，建议在 DeepSeek 控制台**重新生成（轮换）一个 Key**，然后告诉我新 Key 或自行替换 `.env` 第 3 行。
- `.env` 已被 `.gitignore` 忽略，提交代码时不会泄露；请勿截图或分享该文件。

---

**User**

好的，充值后我再回来测试

**Assistant**

好的，充值后直接回来就行，无需改任何代码或配置——服务正在后台运行，直接在 http://127.0.0.1:8000 对话即可自动生效。

如果服务已关闭，重新启动只需两步：

```powershell
.\venv\Scripts\Activate.ps1
python main.py
```

回来后如果调用仍报错，把错误信息发给我即可。

---

