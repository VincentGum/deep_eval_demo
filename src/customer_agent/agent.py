from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from .policies import verify_policy
from .tools import create_refund_case, extract_order_id, lookup_order


class CustomerState(TypedDict, total=False):
    user_message: str
    intent: str
    order_id: str
    plan: list[str]
    tool_trace: list[dict[str, str]]
    draft_response: str
    final_response: str
    confidence: float
    requires_human: bool
    verification_reason: str
    human_decision: dict[str, Any]


def _mock_llm_response(messages: list[Any]) -> AIMessage:
    user_text = " ".join(
        message.content for message in messages if isinstance(message, HumanMessage)
    ).lower()
    if "refund" in user_text or "退款" in user_text:
        content = (
            "I found the order and prepared a refund case. I will ask a support "
            "specialist to approve it before confirming anything to the customer."
        )
    elif "where" in user_text or "status" in user_text or "物流" in user_text:
        content = (
            "I checked the order status and can share the latest delivery update."
        )
    else:
        content = (
            "I can help with that. I will summarize the next step clearly and keep "
            "the customer informed."
        )
    return AIMessage(content=content)


MOCK_LLM = RunnableLambda(_mock_llm_response)


def _plan(state: CustomerState) -> CustomerState:
    user_message = state["user_message"]
    normalized = user_message.lower()
    order_id = extract_order_id(user_message)

    if "refund" in normalized or "退款" in normalized:
        intent = "refund_request"
        plan = ["lookup_order", "create_refund_case", "draft_customer_reply"]
        confidence = 0.86
    elif "where" in normalized or "status" in normalized or "物流" in normalized:
        intent = "order_status"
        plan = ["lookup_order", "draft_customer_reply"]
        confidence = 0.92
    else:
        intent = "general_support"
        plan = ["draft_customer_reply"]
        confidence = 0.62

    return {
        "intent": intent,
        "order_id": order_id,
        "plan": plan,
        "confidence": confidence,
    }


def _execute(state: CustomerState) -> CustomerState:
    order_id = state.get("order_id", "UNKNOWN")
    tool_trace: list[dict[str, str]] = []

    if "lookup_order" in state.get("plan", []):
        result = lookup_order.invoke({"order_id": order_id})
        tool_trace.append({"tool": "lookup_order", "result": result})

    if "create_refund_case" in state.get("plan", []):
        result = create_refund_case.invoke(
            {"order_id": order_id, "reason": state["user_message"]}
        )
        tool_trace.append({"tool": "create_refund_case", "result": result})

    tool_context = "\n".join(item["result"] for item in tool_trace) or "No tool needed."
    llm_message = MOCK_LLM.invoke(
        [
            SystemMessage(content="You are a concise customer support assistant."),
            HumanMessage(content=f"{state['user_message']}\n\nTool context:\n{tool_context}"),
        ]
    )

    return {"tool_trace": tool_trace, "draft_response": llm_message.content}


def _verify(state: CustomerState) -> CustomerState:
    policy = verify_policy(
        state["user_message"],
        state["draft_response"],
        state.get("confidence", 0.0),
    )

    if policy.requires_human and not state.get("human_decision"):
        decision = interrupt(
            {
                "reason": policy.reason,
                "draft_response": state["draft_response"],
                "tool_trace": state.get("tool_trace", []),
            }
        )
        return {
            "requires_human": True,
            "verification_reason": policy.reason,
            "human_decision": decision,
        }

    return {
        "requires_human": policy.requires_human,
        "verification_reason": policy.reason,
    }


def _respond(state: CustomerState) -> CustomerState:
    if state.get("requires_human"):
        decision = state.get("human_decision") or {}
        approved = bool(decision.get("approved"))
        note = decision.get("note", "A human reviewer checked this case.")
        if approved:
            final = (
                f"{state['draft_response']} Human approval recorded: {note} "
                "A support specialist reviewed this before sending."
            )
        else:
            final = (
                "I escalated this case to a support specialist and will avoid "
                "confirming any sensitive action until review is complete."
            )
    else:
        final = state["draft_response"]

    return {"final_response": final}


def build_customer_support_graph():
    graph = StateGraph(CustomerState)
    graph.add_node("plan_step", _plan)
    graph.add_node("execute_step", _execute)
    graph.add_node("verify_step", _verify)
    graph.add_node("respond_step", _respond)

    graph.set_entry_point("plan_step")
    graph.add_edge("plan_step", "execute_step")
    graph.add_edge("execute_step", "verify_step")
    graph.add_edge("verify_step", "respond_step")
    graph.add_edge("respond_step", END)

    return graph.compile(checkpointer=MemorySaver())


def invoke_customer_agent(
    user_message: str,
    human_decision: dict[str, Any] | None = None,
    thread_id: str = "offline-demo",
) -> CustomerState:
    app = build_customer_support_graph()
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"user_message": user_message}, config=config)

    paused_for_human = "__interrupt__" in result or "final_response" not in result
    if paused_for_human:
        if human_decision is None:
            return result
        result = app.invoke(Command(resume=human_decision), config=config)

    return result


def next_human_action(state: CustomerState) -> Literal["review", "none"]:
    return "review" if "__interrupt__" in state else "none"
