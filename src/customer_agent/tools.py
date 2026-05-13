from __future__ import annotations

from langchain_core.tools import tool


@tool
def lookup_order(order_id: str) -> str:
    """Return local mock order status."""
    orders = {
        "A100": "Order A100 is in transit and expected tomorrow.",
        "B200": "Order B200 was delivered yesterday.",
        "C300": "Order C300 payment is pending.",
    }
    return orders.get(order_id.upper(), "Order was not found in the local mock database.")


@tool
def create_refund_case(order_id: str, reason: str) -> str:
    """Create a local mock refund case that still needs human approval."""
    return f"Refund case DRAFT-{order_id.upper()} created for review: {reason}"


def extract_order_id(text: str) -> str:
    for token in text.replace("#", " ").replace(",", " ").split():
        cleaned = "".join(char for char in token.upper() if char.isalnum())
        if len(cleaned) == 4 and cleaned[0].isalpha() and cleaned[1:].isdigit():
            return cleaned
    return "UNKNOWN"
