"""PEV (Plan-Execute-Verify) Customer Support Agent with Human-in-the-Loop.

Architecture:
    - PLAN: Analyze user intent, decide tools to use
    - EXECUTE: Call tools and get results
    - VERIFY: Check policy, decide if human review needed
    - HUMAN_REVIEW: (Optional) Manual approval for sensitive actions
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from .mock_llm import MockChatModel, get_mock_model
from .policies import PolicyDecision, verify_policy
from .tools import create_refund_case, extract_order_id, lookup_order


# --- State Definition ---

class AgentStage(str, Enum):
    """Agent execution stages."""
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"
    HUMAN_REVIEW = "human_review"
    FINAL = "final"


@dataclass
class PlanResult:
    """Result from the PLAN stage."""
    intent: str
    tools_to_use: list[str]
    confidence: float
    draft_response: str


@dataclass
class ExecuteResult:
    """Result from the EXECUTE stage."""
    tool_calls: list[dict[str, Any]]
    tool_results: list[str]


@dataclass
class VerifyResult:
    """Result from the VERIFY stage."""
    decision: PolicyDecision


@dataclass
class HumanReviewResult:
    """Result from HUMAN_REVIEW stage."""
    approved: bool
    modified_response: str | None = None


class AgentState(TypedDict):
    """State passed between nodes in the agent graph."""
    # Input
    user_message: str

    # Stage outputs
    current_stage: AgentStage
    plan: PlanResult | None
    execute: ExecuteResult | None
    verify: VerifyResult | None
    human_review: HumanReviewResult | None

    # Final output
    final_response: str
    messages: list[dict[str, Any]]

    # Metadata
    error: str | None


# --- Node Implementations ---

def create_plan_node(llm: MockChatModel | None = None) -> Callable[[AgentState], AgentState]:
    """Create the PLAN node that analyzes user intent and decides actions."""
    model = llm or get_mock_model()

    def plan_node(state: AgentState) -> AgentState:
        user_message = state["user_message"]

        # Extract order ID if present
        order_id = extract_order_id(user_message)

        # Get LLM response
        messages = [
            SystemMessage(content="""You are a customer support agent. Analyze the user's message and:
1. Identify their intent (order_status, refund, general_inquiry, etc.)
2. Decide if you need to use tools
3. Draft a response

Always extract the order ID from the message if present."""),
            HumanMessage(content=user_message),
        ]

        response = model.invoke(messages)
        response_content = response.content if hasattr(response, 'content') else str(response)

        # Extract metadata
        confidence = 0.5
        intent = "unknown"
        if hasattr(response, 'additional_kwargs'):
            confidence = response.additional_kwargs.get('confidence', 0.5)
            intent = response.additional_kwargs.get('intent', 'unknown')

        # Determine tools to use based on user message keywords
        # (More reliable than LLM-returned intent)
        tools_to_use = []
        draft_response = response_content
        user_message_lower = user_message.lower()

        # Sensitive action keywords that require human review
        sensitive_keywords = ("refund", "退款", "cancel", "取消", "complaint", "投诉",
                            "chargeback", "赔偿", "damaged", "损坏", "late", "late",
                            "open", "broken", "问题", "issue")

        # Check if order ID is present
        has_order_id = order_id != "UNKNOWN"
        has_sensitive_keyword = any(keyword in user_message_lower for keyword in sensitive_keywords)

        if has_order_id:
            # Always lookup order if ID is present
            tools_to_use = ["lookup_order"]

            # Add create_refund_case if sensitive action detected
            if has_sensitive_keyword:
                tools_to_use.append("create_refund_case")
                draft_response = f"Based on your inquiry about order {order_id}, I found:"
            else:
                draft_response = f"Based on your inquiry about order {order_id}, I found:"
        elif has_sensitive_keyword:
            # No order ID but has sensitive keywords - still need to create refund case
            tools_to_use = ["create_refund_case"]
            draft_response = response_content

        # Extract confidence from LLM metadata
        if hasattr(response, 'additional_kwargs'):
            confidence = response.additional_kwargs.get('confidence', confidence)

        state["plan"] = PlanResult(
            intent=intent,
            tools_to_use=tools_to_use,
            confidence=confidence,
            draft_response=draft_response,
        )
        state["current_stage"] = AgentStage.EXECUTE

        return state

    return plan_node


def create_execute_node() -> Callable[[AgentState], AgentState]:
    """Create the EXECUTE node that calls tools."""
    tools_map = {
        "lookup_order": lookup_order,
        "create_refund_case": create_refund_case,
    }

    def execute_node(state: AgentState) -> AgentState:
        plan = state.get("plan")
        if not plan:
            state["error"] = "No plan found, execute called before plan"
            state["current_stage"] = AgentStage.FINAL
            return state

        tool_calls = []
        tool_results = []
        user_message = state["user_message"]
        order_id = extract_order_id(user_message)

        # Execute each tool
        for tool_name in plan.tools_to_use:
            tool_func = tools_map.get(tool_name)
            if not tool_func:
                tool_results.append(f"Tool {tool_name} not found")
                continue

            try:
                # Prepare tool arguments
                if tool_name == "lookup_order":
                    result = tool_func.invoke({"order_id": order_id})
                elif tool_name == "create_refund_case":
                    reason = user_message
                    result = tool_func.invoke({"order_id": order_id, "reason": reason})
                else:
                    result = tool_func.invoke({})

                tool_calls.append({
                    "tool": tool_name,
                    "args": {"order_id": order_id} if order_id != "UNKNOWN" else {},
                })
                tool_results.append(str(result))
            except Exception as e:
                tool_results.append(f"Error calling {tool_name}: {str(e)}")

        state["execute"] = ExecuteResult(
            tool_calls=tool_calls,
            tool_results=tool_results,
        )
        state["current_stage"] = AgentStage.VERIFY

        return state

    return execute_node


def create_verify_node() -> Callable[[AgentState], AgentState]:
    """Create the VERIFY node that checks policy compliance."""
    def verify_node(state: AgentState) -> AgentState:
        plan = state.get("plan")
        execute = state.get("execute")

        if not plan:
            state["error"] = "No plan found"
            state["current_stage"] = AgentStage.FINAL
            return state

        # Build draft response from plan and tool results
        draft = plan.draft_response
        if execute and execute.tool_results:
            draft += "\n" + "\n".join(execute.tool_results)

        # Verify policy
        decision = verify_policy(
            user_message=state["user_message"],
            draft=draft,
            confidence=plan.confidence,
        )

        state["verify"] = VerifyResult(decision=decision)

        # Decide next stage
        if decision.requires_human:
            state["current_stage"] = AgentStage.HUMAN_REVIEW
        else:
            state["current_stage"] = AgentStage.FINAL
            state["final_response"] = draft

        return state

    return verify_node


def create_human_review_node(
    human_approval_func: Callable[[str, str], tuple[bool, str | None]] | None = None
) -> Callable[[AgentState], AgentState]:
    """Create the HUMAN_REVIEW node that requires manual approval.

    Args:
        human_approval_func: Optional function that takes (user_message, draft_response)
            and returns (approved, modified_response). If None, auto-approves after 3 seconds.
    """
    def human_review_node(state: AgentState) -> AgentState:
        plan = state.get("plan")
        execute = state.get("execute")

        if not plan:
            state["error"] = "No plan found"
            state["current_stage"] = AgentStage.FINAL
            return state

        # Build draft response
        draft = plan.draft_response
        if execute and execute.tool_results:
            draft += "\n" + "\n".join(execute.tool_results)

        # Call human approval function
        if human_approval_func:
            approved, modified = human_approval_func(state["user_message"], draft)
        else:
            # Default: simulate human review
            # In real scenario, this would pause and wait for human input
            approved = True
            modified = None
            print(f"[HUMAN REVIEW REQUIRED] Reason: {state['verify'].decision.reason}")
            print(f"[HUMAN REVIEW] Draft response:\n{draft}")
            print("[HUMAN REVIEW] Auto-approved for demo (set human_approval_func for real approval)")

        state["human_review"] = HumanReviewResult(
            approved=approved,
            modified_response=modified,
        )

        if approved:
            state["final_response"] = modified if modified else draft
        else:
            state["final_response"] = "Your request has been escalated to our support team. We will contact you within 24 hours."

        state["current_stage"] = AgentStage.FINAL

        return state

    return human_review_node


# --- Graph Builder ---

def build_customer_support_graph(
    human_approval_func: Callable[[str, str], tuple[bool, str | None]] | None = None,
    llm: MockChatModel | None = None,
) -> StateGraph:
    """Build the PEV customer support agent graph.

    Args:
        human_approval_func: Function for human approval (user_message, draft) -> (approved, modified_response)
        llm: Mock LLM instance (uses default if None)

    Returns:
        Compiled StateGraph
    """
    # Create nodes
    plan_node = create_plan_node(llm)
    execute_node = create_execute_node()
    verify_node = create_verify_node()
    human_review_node = create_human_review_node(human_approval_func)

    # Build graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("plan", plan_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("final", lambda s: s)

    # Define edges
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "execute")
    workflow.add_edge("execute", "verify")

    # Conditional edge from verify
    def should_human_review(state: AgentState) -> str:
        verify = state.get("verify")
        if verify and verify.decision.requires_human:
            return "human_review"
        return "final"

    workflow.add_conditional_edges("verify", should_human_review)
    workflow.add_edge("human_review", "final")
    workflow.add_edge("final", END)

    return workflow.compile()


# --- Main Interface ---

def invoke_customer_agent(
    user_message: str,
    human_approval_func: Callable[[str, str], tuple[bool, str | None]] | None = None,
    llm: MockChatModel | None = None,
) -> dict[str, Any]:
    """Invoke the customer support agent.

    Args:
        user_message: User's input message
        human_approval_func: Optional function for human approval
        llm: Mock LLM instance (uses default if None)

    Returns:
        Dictionary with:
            - response: Final response to user
            - state: Full agent state for debugging
            - needs_human_review: Whether human review was required
    """
    # Build or get cached graph
    graph = build_customer_support_graph(human_approval_func, llm)

    # Initialize state
    initial_state: AgentState = {
        "user_message": user_message,
        "current_stage": AgentStage.PLAN,
        "plan": None,
        "execute": None,
        "verify": None,
        "human_review": None,
        "final_response": "",
        "messages": [],
        "error": None,
    }

    # Run graph
    final_state = graph.invoke(initial_state)

    # Build response
    needs_human_review = (
        final_state.get("verify") is not None
        and final_state.get("verify").decision.requires_human
    )

    return {
        "response": final_state.get("final_response", ""),
        "state": final_state,
        "needs_human_review": needs_human_review,
        "stage": final_state.get("current_stage", AgentStage.FINAL).value,
        "error": final_state.get("error"),
    }


# --- Exports ---

__all__ = [
    "build_customer_support_graph",
    "invoke_customer_agent",
    "AgentState",
    "AgentStage",
    "PlanResult",
    "ExecuteResult",
    "VerifyResult",
    "HumanReviewResult",
]
