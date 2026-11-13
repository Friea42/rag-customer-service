"""LangGraph 客服工作流。

流程：
    START
      → intent（意图识别：退货 refund / 物流 logistics / 知识库 faq）
        ├─ refund   → 收集订单号、退货原因（多轮）→ 调用 request_refund
        ├─ logistics→ 收集订单号（多轮）        → 调用 query_order
        └─ faq      → RAG 知识库检索 + 生成回答

使用内存 Checkpointer 按 thread_id 保存多轮对话状态（订单号、待补充槽位等）。
"""
from __future__ import annotations

import json
import re
import uuid
from functools import lru_cache
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# langgraph 新版本名为 InMemorySaver，旧版本为 MemorySaver，两者都兼容
try:
    from langgraph.checkpoint.memory import InMemorySaver as _MemorySaver
except ImportError:  # pragma: no cover
    from langgraph.checkpoint.memory import MemorySaver as _MemorySaver

from .knowledge_base import get_knowledge_base
from .llm import get_llm
from .tools import query_order, request_refund

# ----------------------------------------------------------------------
# 状态定义
# ----------------------------------------------------------------------
class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    intent: str  # refund / logistics / faq
    order_id: str
    reason: str
    pending: str  # logistics:order / refund:order / refund:reason；为空表示没有待补充信息


# ----------------------------------------------------------------------
# 意图识别（LLM JSON 解析 + 关键词规则兜底，兼容各种 OpenAI 网关）
# ----------------------------------------------------------------------
ORDER_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{1,3}\d{3,8})(?![A-Za-z0-9])")

INTENT_SYSTEM = """你是电商客服系统的「意图识别」模块。只输出一个 JSON 对象，不要输出任何解释或多余文字。
- intent 只能取以下三个值：
  - "refund"：退货、退款、退换货相关
  - "logistics"：物流、快递、运单、发货状态、包裹到哪了
  - "faq"：退换货规则/运费/时效等政策咨询、常见问题、闲聊或其他任何情况
- order_id：消息中出现的订单号（形如 A1001），没有则填 ""
- reason：用户描述的退货原因，没有则填 ""
输出示例：{"intent": "refund", "order_id": "A1001", "reason": "商品有划痕"}"""


def extract_order_id(text: str) -> str:
    match = ORDER_ID_PATTERN.search(text or "")
    return match.group(1).upper() if match else ""


def _parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


# 政策/规则咨询类信号：即使句子中出现“退货/物流”等字样，也应走知识库而非业务办理
_FAQ_SIGNALS = (
    "规则", "政策", "流程", "规定", "怎么规定", "运费", "时效", "几天", "多久",
    "到账", "发票", "保修", "质保", "换货", "可以退吗", "能退吗", "能退么",
    "支持退", "怎么退", "如何退", "怎么操作", "什么是", "介绍", "咨询", "吗？",
    "什么快递", "哪个快递", "哪家快递",
    "吗?", "啥时候", "什么时候",
)


def _rule_classify(text: str) -> dict:
    is_question = text.rstrip().endswith(("吗", "？", "?"))
    if any(k in text for k in _FAQ_SIGNALS) or (is_question and not any(
        k in text for k in ("我要", "申请", "帮我", "办理", "提交")
    )):
        intent = "faq"
    elif any(k in text for k in ("退货", "退款", "退换", "退钱", "退回去", "退单")):
        intent = "refund"
    elif any(k in text for k in ("物流", "快递", "运单", "发货", "到哪", "派送", "签收", "包裹")):
        intent = "logistics"
    else:
        intent = "faq"
    return {"intent": intent, "order_id": extract_order_id(text), "reason": ""}


def classify(text: str) -> dict:
    llm = get_llm()
    if llm is not None:
        try:
            resp = llm.invoke([
                SystemMessage(content=INTENT_SYSTEM),
                HumanMessage(content=text),
            ])
            data = _parse_json(resp.content) or {}
            if data.get("intent") in ("refund", "logistics", "faq"):
                return {
                    "intent": data["intent"],
                    "order_id": (str(data.get("order_id") or "")).strip().upper()
                    or extract_order_id(text),
                    "reason": str(data.get("reason") or "").strip(),
                }
        except Exception:
            pass  # 模型不可用 / 返回异常 → 规则兜底
    return _rule_classify(text)


def _llm_reply(system: str, user: str, fallback: str) -> str:
    """有可用 LLM 则生成自然语言回答，否则使用模板兜底。"""
    llm = get_llm()
    if llm is None:
        return fallback
    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        content = (getattr(resp, "content", "") or "").strip()
        return content or fallback
    except Exception:
        return fallback


# ----------------------------------------------------------------------
# 节点 1：意图识别 / 多轮槽位接续
# ----------------------------------------------------------------------
def intent_node(state: State) -> dict:
    text = str(state["messages"][-1].content)
    pending = state.get("pending") or ""

    # 上一轮客服在追问订单号
    if pending in ("logistics:order", "refund:order"):
        oid = extract_order_id(text)
        if not oid:
            return {
                "pending": pending,
                "messages": [AIMessage(content="没有识别到订单号哦，请直接发送订单号（例如 A1001）。")],
            }
        intent = "logistics" if pending == "logistics:order" else "refund"
        return {"order_id": oid, "pending": "", "intent": intent}

    # 上一轮客服在追问退货原因：把整句话当作原因
    if pending == "refund:reason":
        return {"reason": text.strip(), "pending": "", "intent": "refund"}

    # 新一轮对话：识别意图
    info = classify(text)
    update: dict = {"intent": info["intent"], "pending": ""}
    if info["order_id"]:
        update["order_id"] = info["order_id"]
    if info["intent"] == "refund" and info["reason"]:
        update["reason"] = info["reason"]
    return update


# ----------------------------------------------------------------------
# 节点 2：物流查询
# ----------------------------------------------------------------------
def logistics_node(state: State) -> dict:
    text = str(state["messages"][-1].content)
    oid = state.get("order_id") or extract_order_id(text)

    if not oid:
        return {
            "pending": "logistics:order",
            "messages": [AIMessage(content="好的，请问您的订单号是多少？订单号格式如 A1001。")],
        }

    info = query_order(oid)
    if not info["ok"]:
        return {
            "pending": "logistics:order",
            "messages": [AIMessage(content=f"{info['error']}，请重新提供订单号。")],
        }

    order = info["order"]
    facts = (
        f"订单号：{order['id']}\n"
        f"商品：{order['item']}\n"
        f"订单状态：{order['status']}\n"
        f"物流公司：{order['carrier']}\n"
        f"运单号：{order['tracking_no']}\n"
        f"送达情况：{order['eta']}"
    )
    reply = _llm_reply(
        system="你是电商客服。请根据查询到的订单物流信息，用简洁友好的中文回答用户，不要编造未提供的信息。",
        user=f"物流信息如下：\n{facts}",
        fallback=f"已为您查询到订单 {oid} 的物流信息：\n{facts}",
    )
    return {"order_id": oid, "pending": "", "messages": [AIMessage(content=reply)]}


# ----------------------------------------------------------------------
# 节点 3：退货（多轮收集订单号 + 原因，然后调用工具）
# ----------------------------------------------------------------------
def refund_node(state: State) -> dict:
    text = str(state["messages"][-1].content)
    oid = state.get("order_id") or extract_order_id(text)

    if not oid:
        return {
            "pending": "refund:order",
            "messages": [AIMessage(content="您好，办理退货需要先确认订单，请问您的订单号是多少？")],
        }

    info = query_order(oid)
    if not info["ok"]:
        return {
            "pending": "refund:order",
            "messages": [AIMessage(content=f"{info['error']}，请重新提供订单号。")],
        }

    reason = (state.get("reason") or "").strip()
    if not reason:
        return {
            "order_id": oid,
            "pending": "refund:reason",
            "messages": [
                AIMessage(
                    content=(
                        f"好的，订单 {oid}（{info['order']['item']}）已确认。"
                        "请简单说明退货原因，例如：质量问题、尺码不合适、不想要了。"
                    )
                )
            ],
        }

    result = request_refund(oid, reason)
    if not result["ok"]:
        return {"messages": [AIMessage(content=f"抱歉，退货申请未能提交：{result['error']}")]}

    reply = _llm_reply(
        system="你是电商客服。退货申请已成功提交，请用简洁友好的中文告知用户结果与退款时效，不要编造信息。",
        user=(
            f"退款单号：{result['refund_id']}\n订单号：{oid}\n退货原因：{reason}\n"
            f"系统提示：{result['message']}"
        ),
        fallback=(
            "您的退货申请已提交成功！\n"
            f"退款单号：{result['refund_id']}\n订单号：{oid}\n退货原因：{reason}\n"
            f"{result['message']}"
        ),
    )
    return {"order_id": oid, "reason": "", "pending": "", "messages": [AIMessage(content=reply)]}


# ----------------------------------------------------------------------
# 节点 4：知识库问答（RAG）
# ----------------------------------------------------------------------
def faq_node(state: State) -> dict:
    text = str(state["messages"][-1].content)
    kb = get_knowledge_base()
    docs = kb.retrieve(text, k=3)

    context = "\n\n".join(f"【{d['title']}】\n{d['content']}" for d in docs)
    fallback = "根据知识库，为您找到以下信息：\n\n" + "\n\n".join(
        f"{i}. {d['title']}\n{d['content']}" for i, d in enumerate(docs, start=1)
    )
    reply = _llm_reply(
        system=(
            "你是电商店铺的客服助手。请只依据下面的知识库内容回答用户问题，使用简洁有条理的中文；"
            "知识库中没有相关内容时，礼貌说明并建议联系人工客服（在线时间 9:00-22:00）。"
        ),
        user=f"知识库检索结果：\n{context}\n\n用户问题：{text}",
        fallback=fallback,
    )
    return {"messages": [AIMessage(content=reply)]}


# ----------------------------------------------------------------------
# 路由：若有待补充槽位（节点刚发出追问），直接结束本轮等待用户回复
# ----------------------------------------------------------------------
def route(state: State) -> str:
    if state.get("pending"):
        return END
    return {"refund": "refund", "logistics": "logistics"}.get(state.get("intent"), "faq")


@lru_cache
def build_graph():
    graph = StateGraph(State)
    graph.add_node("intent", intent_node)
    graph.add_node("logistics", logistics_node)
    graph.add_node("refund", refund_node)
    graph.add_node("faq", faq_node)

    graph.add_edge(START, "intent")
    graph.add_conditional_edges(
        "intent",
        route,
        {"refund": "refund", "logistics": "logistics", "faq": "faq", END: END},
    )
    graph.add_edge("logistics", END)
    graph.add_edge("refund", END)
    graph.add_edge("faq", END)

    return graph.compile(checkpointer=_MemorySaver())


def chat(message: str, thread_id: str | None = None) -> tuple[str, str]:
    """对话入口。同一 thread_id 下自动保留多轮上下文。"""
    thread_id = thread_id or str(uuid.uuid4())
    result = build_graph().invoke(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return str(result["messages"][-1].content), thread_id
