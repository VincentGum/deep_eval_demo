"""
客服工具模块 - 定义客服 Agent 可用的工具

【工具说明】
本模块定义了客服 Agent 可以调用的工具：
1. lookup_order - 根据订单号查询订单状态
2. create_refund_case - 创建退款工单（敏感操作）
3. get_order_status - 获取通用订单政策信息

【设计原则】
- 每个工具都是 LangChain Tool 的子类
- 使用装饰器 @tool 定义工具函数
- Mock 实现用于演示，生产环境替换为真实 API
"""

from __future__ import annotations

from langchain_core.tools import tool, Tool
from typing import Optional, Type
from pydantic import BaseModel, Field


# ============================================================================
# 工具输入模型定义
# ============================================================================

class LookupOrderInput(BaseModel):
    """
    lookup_order 工具的输入参数模型
    
    【字段】
    - query: 订单查询条件（订单号或关键词）
    """
    query: str = Field(description="订单号或查询关键词，如 'A100'")


class CreateRefundCaseInput(BaseModel):
    """
    create_refund_case 工具的输入参数模型
    
    【字段】
    - order_id: 需要退款的订单号
    - reason: 退款原因
    """
    order_id: str = Field(description="需要退款的订单号")
    reason: str = Field(description="退款原因")


class GetOrderStatusInput(BaseModel):
    """
    get_order_status 工具的输入参数模型
    
    无需参数，返回通用政策信息
    """
    pass  # 空输入模型


# ============================================================================
# Mock 数据存储
# ============================================================================

# 模拟订单数据库
MOCK_ORDERS = {
    "A100": {
        "order_id": "A100",
        "status": "delivered",
        "delivery_date": "2024-01-15",
        "items": ["Widget Pro", "Basic Case"],
        "total": 299.99,
    },
    "B200": {
        "order_id": "B200", 
        "status": "in_transit",
        "expected_delivery": "2024-01-20",
        "items": ["Widget Basic"],
        "total": 99.99,
    },
    "C300": {
        "order_id": "C300",
        "status": "pending",
        "items": ["Widget Deluxe"],
        "total": 199.99,
    },
    "D400": {
        "order_id": "D400",
        "status": "cancelled",
        "cancellation_date": "2024-01-10",
        "items": ["Widget Pro"],
        "total": 299.99,
    },
}

# 模拟退款工单计数器
_refund_case_counter = 1000


# ============================================================================
# 辅助函数
# ============================================================================

def _format_order_response(order: dict) -> str:
    """格式化订单响应"""
    if order["status"] == "delivered":
        return (
            f"Order {order['order_id']} is delivered on {order['delivery_date']}. "
            f"Items: {', '.join(order['items'])}. Total: ${order['total']:.2f}."
        )
    elif order["status"] == "in_transit":
        return (
            f"Order {order['order_id']} is in transit and expected on {order['expected_delivery']}. "
            f"Items: {', '.join(order['items'])}. Total: ${order['total']:.2f}."
        )
    elif order["status"] == "pending":
        return (
            f"Order {order['order_id']} is pending payment. "
            f"Items: {', '.join(order['items'])}. Total: ${order['total']:.2f}."
        )
    elif order["status"] == "cancelled":
        return (
            f"Order {order['order_id']} was cancelled on {order['cancellation_date']}. "
            f"Items: {', '.join(order['items'])}. Total: ${order['total']:.2f}."
        )
    return f"Order {order['order_id']} status: {order['status']}."


# ============================================================================
# 工具函数定义
# ============================================================================

@tool(args_schema=LookupOrderInput)
def lookup_order(query: str) -> str:
    """
    根据订单号查询订单状态
    
    【参数】
    - query: 订单号或查询关键词
    
    【返回】
    格式化的订单信息字符串
    
    【示例】
    >>> lookup_order.invoke({"query": "A100"})
    'Order A100 is delivered on 2024-01-15.'
    """
    query = query.strip().upper()
    
    # 尝试精确匹配订单号
    if query in MOCK_ORDERS:
        order = MOCK_ORDERS[query]
        return _format_order_response(order)
    
    # 尝试模糊匹配
    for order_id, order in MOCK_ORDERS.items():
        if query in order_id or any(query.lower() in item.lower() for item in order["items"]):
            return _format_order_response(order)
    
    # 未找到订单
    return f"Order not found for query: {query}"


@tool(args_schema=CreateRefundCaseInput)
def create_refund_case(order_id: str, reason: str) -> str:
    """
    创建退款工单
    
    【参数】
    - order_id: 订单号
    - reason: 退款原因
    
    【返回】
    退款工单创建结果，包括工单号
    
    【安全说明】
    此函数执行敏感操作，需要后续人工审核
    """
    global _refund_case_counter
    
    order_id = order_id.strip().upper()
    
    # 验证订单是否存在
    if order_id not in MOCK_ORDERS:
        return f"Cannot create refund case: Order {order_id} not found."
    
    order = MOCK_ORDERS[order_id]
    
    # 检查订单状态 - 已取消的订单不能退款
    if order["status"] == "cancelled":
        return f"Cannot create refund case: Order {order_id} is already cancelled."
    
    # 生成退款工单号
    _refund_case_counter += 1
    case_number = f"DRAFT-{_refund_case_counter}"
    
    # 记录退款请求（实际会写入数据库）
    refund_info = {
        "case_number": case_number,
        "order_id": order_id,
        "reason": reason,
        "amount": order["total"],
        "status": "pending_review",  # 待审核状态
    }
    
    # 返回工单信息
    return (
        f"Refund case created successfully.\n"
        f"Case Number: {case_number}\n"
        f"Order: {order_id}\n"
        f"Amount: ${refund_info['amount']:.2f}\n"
        f"Reason: {reason}\n"
        f"Status: Pending Human Review\n\n"
        f"Note: This refund request requires human approval before processing."
    )


@tool(args_schema=GetOrderStatusInput)
def get_order_status() -> str:
    """
    获取订单政策信息
    
    【返回】
    格式化的政策说明文本
    """
    return """
## Order Status Information

### How to Check Your Order
You can check your order status using your order number (e.g., A100, B200, C300).

### Delivery Timeframes
- Standard shipping: 5-7 business days
- Express shipping: 2-3 business days
- Same-day delivery: Available in select areas

### Common Order Statuses
- **Pending**: Order received, awaiting payment confirmation
- **Processing**: Order confirmed, preparing for shipment
- **In Transit**: Package shipped, on the way to you
- **Delivered**: Package has been delivered
- **Cancelled**: Order has been cancelled

### Need More Help?
If you have questions about a specific order, please provide your order number.
"""


# ============================================================================
# 工具列表导出
# ============================================================================

# 导出所有工具，供 Agent 绑定使用
CUSTOMER_SUPPORT_TOOLS = [
    lookup_order,
    create_refund_case,
    get_order_status,
]

__all__ = [
    "lookup_order",
    "create_refund_case", 
    "get_order_status",
    "CUSTOMER_SUPPORT_TOOLS",
]
