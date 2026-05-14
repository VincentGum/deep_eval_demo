"""
客户支持 Agent - 基于 PEV (Plan-Execute-Verify) 架构的智能客服系统

【模块概述】
本模块实现了客服 Agent 的核心逻辑：
1. PLAN 节点 - 分析用户意图，决定调用哪些工具
2. EXECUTE 节点 - 调用 tools.py 中定义的工具
3. VERIFY 节点 - 调用 policies.verify_policy 检查敏感词和置信度
4. HUMAN_REVIEW 节点 - 敏感操作需人工审批

【核心函数】
- invoke_customer_agent(user_message, human_approval_func) - 主入口函数
- build_customer_support_graph() - 构建 LangGraph 状态图

【设计原则】
- 使用结构化推理 (Structured Reasoning) 而非关键词匹配
- 支持 Human-in-the-Loop 机制处理敏感操作
- 策略验证确保回复符合合规要求
"""

from __future__ import annotations

from typing import TypedDict, Annotated, Sequence
from datetime import datetime

# LangChain 核心组件
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.utils.function_calling import convert_to_openai_function

# 工具和策略
from .tools import (
    lookup_order,           # 订单查询工具
    create_refund_case,    # 创建退款工单工具
    get_order_status,      # 获取订单状态工具
)
from .policies import (
    verify_policy,         # 策略验证函数
    is_sensitive_operation,  # 判断是否为敏感操作
    SensitiveOperation,   # 敏感操作枚举
)
from .mock_llm import (
    StructuredReasoningLLM,  # 结构化推理 LLM
    StructuredResult as IntentClassification,  # 意图分类结果（别名）
    Intent,  # 意图枚举类型
)


# ============================================================================
# LangGraph 状态定义
# ============================================================================

class AgentState(TypedDict):
    """
    Agent 状态定义 - LangGraph 使用 TypedDict 定义状态结构
    
    【状态字段】
    - messages: 消息历史列表
    - intent: 识别的用户意图
    - tools_called: 已调用的工具列表
    - human_review: 是否需要人工审核
    - human_review_reason: 人工审核原因
    - is_sensitive: 是否检测到敏感内容
    - confidence: 响应置信度
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    intent: str                    # 识别的用户意图
    tools_called: list[str]       # 已调用的工具名称列表
    human_review: bool            # 是否需要人工审核
    human_review_reason: str      # 人工审核的原因
    is_sensitive: bool            # 是否包含敏感内容
    confidence: float             # 响应置信度 (0.0-1.0)


def add_messages(left: list, right: list) -> list:
    """
    消息合并函数 - 将新消息追加到消息列表
    
    Args:
        left: 现有消息列表
        right: 新消息列表
    
    Returns:
        合并后的消息列表
    """
    # 简单追加，实际使用中可能需要去重等逻辑
    return left + right


# ============================================================================
# LLM 和工具配置
# ============================================================================

# 创建结构化推理 LLM 实例
# 使用 Mock 实现，生产环境可替换为真实的 LLM API
llm = StructuredReasoningLLM()

# 将工具函数转换为 OpenAI 函数格式
# 供 LLM 调用时使用
tools = [lookup_order, create_refund_case, get_order_status]
tool_functions = [convert_to_openai_function(t) for t in tools]

# 将工具绑定到 LLM
# 这样 LLM 可以自动决定调用哪个工具
llm_with_tools = llm.bind(functions=tool_functions)


# ============================================================================
# Agent 节点定义
# ============================================================================

def analyze_intent_node(state: AgentState) -> AgentState:
    """
    【PLAN 节点】分析用户意图
    
    使用结构化推理 LLM 分析用户消息，确定：
    1. 用户的真实意图 (intent)
    2. 需要调用的工具 (task_type)
    3. 任务参数 (parameters)
    
    Args:
        state: 当前 Agent 状态
    
    Returns:
        更新后的 Agent 状态
    """
    messages = state["messages"]
    user_message = messages[-1].content
    
    # 调用结构化推理 LLM
    reasoning_result = llm.analyze_intent(user_message)
    
    # 更新状态
    return {
        "intent": reasoning_result.intent,
        "tools_called": [],          # 清空已调用工具
        "human_review": False,        # 默认不需要人工审核
        "human_review_reason": "",
        "confidence": reasoning_result.confidence,
    }


def execute_tools_node(state: AgentState) -> AgentState:
    """
    【EXECUTE 节点】执行工具调用
    
    根据 PLAN 节点的决策，调用相应的工具：
    - order_status: 调用 lookup_order
    - refund: 调用 create_refund_case
    - policy: 调用 get_order_status
    
    Args:
        state: 当前 Agent 状态
    
    Returns:
        更新后的 Agent 状态
    """
    messages = state["messages"]
    user_message = messages[-1].content
    intent = state["intent"]
    reasoning = llm.analyze_intent(user_message)
    
    tools_called = []
    tool_results = []
    
    # 根据意图类型决定调用哪些工具
    # intent 可能是 Intent 枚举或字符串
    intent_value = intent.value if isinstance(intent, Intent) else intent
    
    if intent_value == "order_status":
        # 查询订单状态
        order_id = reasoning.parameters.get("order_id", "")
        result = lookup_order.invoke({"query": order_id})
        tools_called.append("lookup_order")
        tool_results.append(str(result) if hasattr(result, 'content') else result)
    
    elif intent_value == "refund":
        # 创建退款工单 - 敏感操作！
        refund_reason = reasoning.parameters.get("reason", "用户请求退款")
        result = create_refund_case.invoke({
            "order_id": reasoning.parameters.get("order_id", ""),
            "reason": refund_reason,
        })
        tools_called.append("create_refund_case")
        tool_results.append(str(result) if hasattr(result, 'content') else result)
    
    elif intent_value == "policy":
        # 查询政策
        result = get_order_status.invoke({})
        tools_called.append("get_order_status")
        tool_results.append(str(result) if hasattr(result, 'content') else result)
    
    # 添加工具执行结果到消息
    if tool_results:
        messages = messages + [AIMessage(content="\n".join(tool_results))]
    
    return {
        "messages": messages,
        "tools_called": state["tools_called"] + tools_called,
    }


def verify_response_node(state: AgentState) -> AgentState:
    """
    【VERIFY 节点】验证响应策略
    
    检查回复是否：
    1. 包含敏感词/违规内容
    2. 符合置信度要求
    3. 需要人工审核
    
    Args:
        state: 当前 Agent 状态
    
    Returns:
        更新后的 Agent 状态
    """
    messages = state["messages"]
    last_response = messages[-1].content if messages else ""
    
    # 调用策略验证函数
    # 返回 (是否合规, 敏感词列表, 置信度)
    is_compliant, sensitive_words, confidence = verify_policy(
        last_response, 
        state.get("intent", "")
    )
    
    # 检查是否包含敏感内容
    is_sensitive = len(sensitive_words) > 0
    
    # 检查是否为敏感操作（如退款）
    intent = state.get("intent", "")
    intent_value = intent.value if isinstance(intent, Intent) else intent
    requires_human_review = (
        is_sensitive or 
        intent_value == "refund" or
        "refund" in state.get("tools_called", [])
    )
    
    # 构建审核原因
    reason = ""
    if is_sensitive:
        reason = f"检测到敏感词: {', '.join(sensitive_words)}"
    elif requires_human_review:
        reason = "退款操作需要人工审核"
    
    return {
        "is_sensitive": is_sensitive,
        "human_review": requires_human_review,
        "human_review_reason": reason,
        "confidence": confidence,
    }


def human_review_node(state: AgentState) -> AgentState:
    """
    【HUMAN_REVIEW 节点】处理人工审核流程
    
    如果操作需要人工审核：
    1. 调用 human_approval_func 获取审核结果
    2. 根据审核结果决定是否继续执行敏感操作
    
    Args:
        state: 当前 Agent 状态
    
    Returns:
        更新后的 Agent 状态
    """
    if not state.get("human_review"):
        return state
    
    # 获取审批函数
    approval_func = state.get("human_approval_func")
    
    # 如果没有审批函数，使用默认值（拒绝）
    if not approval_func:
        # 默认拒绝敏感操作
        state["human_review"] = False
        state["human_review_reason"] = "无审批函数，默认拒绝敏感操作"
        return state
    
    # 调用审批函数
    user_message = state["messages"][0].content if state["messages"] else ""
    draft_response = state.get("draft_response", "")
    
    try:
        approved, reason = approval_func(user_message, draft_response)
    except Exception:
        approved = False
        reason = "审批函数执行失败"
    
    # 根据审批结果更新状态
    if approved:
        state["human_review"] = False  # 审核通过，不需要等待
        state["human_review_reason"] = f"已批准: {reason}" if reason else "已批准"
    else:
        # 审核拒绝，保持 human_review 状态，generate_response 会输出提示
        state["human_review"] = True
        state["human_review_reason"] = f"已拒绝: {reason}" if reason else "已拒绝"
    
    return state


def generate_response_node(state: AgentState) -> AgentState:
    """
    【RESPONSE 节点】生成最终响应
    
    综合所有信息，生成最终的回复消息：
    1. 如果需要人工审核，输出待审核提示
    2. 否则输出工具执行结果
    
    Args:
        state: 当前 Agent 状态
    
    Returns:
        更新后的 Agent 状态
    """
    messages = state["messages"]
    
    # 如果需要人工审核
    if state.get("human_review"):
        review_msg = (
            f"\n\n[待人工审核]\n"
            f"原因: {state.get('human_review_reason', '请确认操作')}\n"
            f"请确认是否继续执行。"
        )
        messages = messages + [AIMessage(content=review_msg)]
    else:
        # 正常响应 - 添加回复前缀
        if state["messages"]:
            response_intro = "I'm a customer service assistant. "
            current_msg = messages[-1].content
            
            # 如果之前有审核（已批准或已拒绝），在响应中说明
            reason = state.get("human_review_reason", "")
            if reason and ("已批准" in reason or "人工" in reason or "审核" in reason):
                current_msg = f"[人工审核已批准]\n{current_msg}"
            
            if not current_msg.startswith(response_intro):
                messages = messages[:-1] + [AIMessage(content=response_intro + current_msg)]
    
    return {"messages": messages}


# ============================================================================
# 条件路由函数
# ============================================================================

def should_execute_tools(state: AgentState) -> str:
    """
    判断是否需要执行工具
    
    根据意图类型决定路由：
    - greeting/thanks/off_topic: 直接生成响应
    - order_status/refund/policy: 执行工具
    
    Returns:
        "execute_tools" 或 "generate_response"
    """
    intent = state.get("intent", "")
    # intent 可能是 Intent 枚举或字符串
    intent_value = intent.value if isinstance(intent, Intent) else intent
    
    # 这些意图需要执行工具
    tool_intents = {"order_status", "refund", "policy"}
    
    if intent_value in tool_intents:
        return "execute_tools"
    return "generate_response"


def should_verify(state: AgentState) -> str:
    """
    判断是否需要验证
    
    Returns:
        "verify_response" 或 "generate_response"
    """
    # 如果执行了工具调用，需要验证
    if state.get("tools_called"):
        return "verify_response"
    return "generate_response"


def should_human_review(state: AgentState) -> str:
    """
    判断是否需要人工审核
    
    Returns:
        "human_review" 或 "generate_response"
    """
    if state.get("human_review"):
        return "human_review"
    return "generate_response"


# ============================================================================
# 构建 LangGraph
# ============================================================================

def build_customer_support_graph():
    """
    构建客服 Agent 的 LangGraph 状态图
    
    【图结构】
    START -> analyze_intent (PLAN)
                  ↓
            ┌─────┴─────┐
            ↓           ↓
      [不需要工具]   [需要工具]
            ↓           ↓
            └─────┬─────┘
                  ↓
            execute_tools (EXECUTE)
                  ↓
            verify_response (VERIFY)
                  ↓
            ┌─────┴─────┐
            ↓           ↓
        [需要审核]  [不需要审核]
            ↓           ↓
            └─────┬─────┘
                  ↓
            generate_response
                  ↓
                END
    
    Returns:
        编译后的 StateGraph
    """
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 注册节点
    workflow.add_node("analyze_intent", analyze_intent_node)     # PLAN 节点
    workflow.add_node("execute_tools", execute_tools_node)       # EXECUTE 节点
    workflow.add_node("verify_response", verify_response_node)   # VERIFY 节点
    workflow.add_node("human_review", human_review_node)         # HUMAN_REVIEW 节点
    workflow.add_node("generate_response", generate_response_node)  # RESPONSE 节点
    
    # 设置入口点
    workflow.set_entry_point("analyze_intent")
    
    # 添加边 (edges)
    workflow.add_conditional_edges(
        "analyze_intent",
        should_execute_tools,
        {
            "execute_tools": "execute_tools",
            "generate_response": "generate_response",
        }
    )
    
    workflow.add_edge("execute_tools", "verify_response")
    
    workflow.add_conditional_edges(
        "verify_response",
        should_human_review,
        {
            "human_review": "human_review",
            "generate_response": "generate_response",
        }
    )
    
    workflow.add_edge("human_review", "generate_response")
    workflow.add_edge("generate_response", END)
    
    # 编译图
    return workflow.compile()


# ============================================================================
# 主入口函数
# ============================================================================

def invoke_customer_agent(
    user_message: str,
    human_approval_func: callable = None,
) -> dict:
    """
    调用客服 Agent 处理用户消息
    
    【参数】
    - user_message: 用户输入的消息
    - human_approval_func: 人工审批回调函数（可选）
    
    【返回】
    包含以下键的字典：
    - response: Agent 的回复文本
    - intent: 识别的用户意图
    - tools_called: 调用的工具列表
    - human_review: 是否需要人工审核
    - state: 完整的 Agent 状态
    
    【示例】
    >>> result = invoke_customer_agent("Where is my order #A100?")
    >>> print(result["response"])
    >>> print(result["intent"])  # "order_status"
    """
    # 构建图（每次调用都重新构建以确保状态干净）
    graph = build_customer_support_graph()
    
    # 初始化状态
    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "intent": "",
        "tools_called": [],
        "human_review": False,
        "human_review_reason": "",
        "is_sensitive": False,
        "confidence": 1.0,
        "human_approval_func": human_approval_func,  # 传递审批函数
    }
    
    # 执行图
    final_state = graph.invoke(initial_state)
    
    # 提取响应
    messages = final_state.get("messages", [])
    response = messages[-1].content if messages else ""
    
    # 返回结果
    return {
        "response": response,
        "intent": final_state.get("intent", ""),
        "tools_called": final_state.get("tools_called", []),
        "human_review": final_state.get("human_review", False),
        "human_review_reason": final_state.get("human_review_reason", ""),
        "is_sensitive": final_state.get("is_sensitive", False),
        "confidence": final_state.get("confidence", 1.0),
        "state": final_state,
    }
