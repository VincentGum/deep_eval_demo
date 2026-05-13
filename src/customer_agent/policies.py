from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyDecision:
    requires_human: bool
    reason: str


SENSITIVE_KEYWORDS = (
    "refund",
    "退款",
    "complaint",
    "投诉",
    "chargeback",
    "赔偿",
    "cancel",
    "取消",
)


def verify_policy(user_message: str, draft: str, confidence: float) -> PolicyDecision:
    text = f"{user_message} {draft}".lower()
    if confidence < 0.7:
        return PolicyDecision(True, "low_confidence")
    if any(keyword in text for keyword in SENSITIVE_KEYWORDS):
        return PolicyDecision(True, "sensitive_customer_action")
    return PolicyDecision(False, "auto_approved")
