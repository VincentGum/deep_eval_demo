"""Customer Agent 模块 - 智能客服系统。

导出模块的主要接口，供外部调用。
"""

from .agent import (
    build_customer_support_graph,
    invoke_customer_agent,
    AgentState,
)

__all__ = [
    "build_customer_support_graph",  # 构建状态图
    "invoke_customer_agent",          # 主入口函数
    "AgentState",                    # 状态类型
]
