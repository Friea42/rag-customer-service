"""模拟业务工具函数。

真实项目中把下面两个函数替换为订单系统 / 售后系统的 HTTP 或 RPC 调用即可，
上层 LangGraph 工作流无需任何改动。
"""
from __future__ import annotations

# ------------------------------------------------------------------
# 模拟订单数据库
# ------------------------------------------------------------------
_ORDERS: dict[str, dict] = {
    "A1001": {
        "id": "A1001",
        "item": "无线蓝牙耳机",
        "status": "运输中",
        "carrier": "顺丰速运",
        "tracking_no": "SF102488976541",
        "eta": "预计明天 18:00 前送达",
        "refund_status": None,
    },
    "A1002": {
        "id": "A1002",
        "item": "保温杯 500ml（雾霾蓝）",
        "status": "已签收",
        "carrier": "中通快递",
        "tracking_no": "ZT770122334455",
        "eta": "已于 2 天前签收",
        "refund_status": None,
    },
    "A1003": {
        "id": "A1003",
        "item": "机械键盘（青轴）",
        "status": "待发货",
        "carrier": "待揽收",
        "tracking_no": "-",
        "eta": "预计 24 小时内发出",
        "refund_status": None,
    },
}


def query_order(order_id: str) -> dict:
    """根据订单号查询订单与物流信息。

    返回: {"ok": True, "order": {...}} 或 {"ok": False, "error": "..."}
    """
    oid = (order_id or "").strip().upper()
    order = _ORDERS.get(oid)
    if not order:
        return {
            "ok": False,
            "error": f"没有查询到订单号 {oid} 的记录，请核对后重新提供（示例订单号：A1001）",
        }
    return {"ok": True, "order": dict(order)}


def request_refund(order_id: str, reason: str) -> dict:
    """提交退货/退款申请。

    返回: {"ok": True, "refund_id": ..., "message": ...} 或 {"ok": False, "error": ...}
    """
    oid = (order_id or "").strip().upper()
    reason = (reason or "").strip()
    order = _ORDERS.get(oid)
    if not order:
        return {"ok": False, "error": f"没有查询到订单号 {oid} 的记录，无法提交退货申请"}
    if not reason:
        return {"ok": False, "error": "退货原因不能为空"}
    if order.get("refund_status") == "退款中":
        return {"ok": False, "error": "该订单已有一笔退款中的申请，请勿重复提交"}

    order["refund_status"] = "退款中"
    refund_id = f"RF{oid[1:]}-2026"
    return {
        "ok": True,
        "refund_id": refund_id,
        "message": "退货申请已进入审核流程，审核通过后 1-3 个工作日内原路退款",
    }
