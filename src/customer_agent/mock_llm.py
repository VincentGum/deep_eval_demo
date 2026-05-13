"""Mock LLM for offline demo - simulates LLM responses deterministically."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult


@dataclass
class IntentPattern:
    """Pattern to match user intent and generate mock response."""
    keywords: tuple[str, ...]
    response: str
    confidence: float
    suggested_action: str | None = None


# Predefined intent patterns for customer support scenarios
# Intent name mapping for normalized output
INTENT_NAMES = {
    "order": "order_status",
    "delivery": "order_status",
    "shipping": "order_status",
    "物流": "order_status",
    "快递": "order_status",
    "发货": "order_status",
    "track": "order_status",
    "status": "order_status",
    "状态": "order_status",
    "在哪里": "order_status",
    "refund": "refund",
    "退款": "refund",
    "cancel": "cancel",
    "取消": "cancel",
    "complaint": "complaint",
    "投诉": "complaint",
    "chargeback": "refund",
    "hello": "greeting",
    "hi": "greeting",
    "help": "greeting",
    "你好": "greeting",
    "问": "greeting",
    "thank": "thanks",
    "thanks": "thanks",
    "谢谢": "thanks",
    "感谢": "thanks",
}

# Intent patterns with normalized names
INTENT_PATTERNS: list[IntentPattern] = [
    IntentPattern(
        keywords=("order", "delivery", "shipping", "物流", "快递", "发货", "track", "status", "状态", "在哪里"),
        response="I'll look up your order status right away.",
        confidence=0.85,
        suggested_action="lookup_order",
    ),
    IntentPattern(
        keywords=("refund", "退款", "cancel", "取消", "complaint", "投诉", "chargeback"),
        response="I understand you need help with this matter. Let me look into it for you.",
        confidence=0.75,
        suggested_action="lookup_order",
    ),
    IntentPattern(
        keywords=("hello", "hi", "你好", "问"),
        response="Hello! I'm here to help you with your order inquiries. Could you please provide your order ID?",
        confidence=0.95,
        suggested_action=None,
    ),
    IntentPattern(
        keywords=("thank", "thanks", "谢谢", "感谢", "help"),
        response="You're welcome! Is there anything else I can help you with?",
        confidence=0.95,
        suggested_action=None,
    ),
]


@dataclass
class MockChatModelConfig:
    """Configuration for MockChatModel."""
    default_response: str = "I'm not sure how to help with that. Could you provide more details?"
    default_confidence: float = 0.5
    intent_patterns: list[IntentPattern] = field(default_factory=lambda: INTENT_PATTERNS)


class MockChatModel(BaseChatModel):
    """A deterministic mock chat model for offline testing.

    This mock simulates LLM responses by pattern matching on user input,
    returning predefined responses with associated confidence scores.

    Usage:
        model = MockChatModel()
        response = model.invoke([HumanMessage(content="Where is my order?")])
    """

    model_name: str = "mock-customer-support-v1"
    config: MockChatModelConfig = field(default_factory=MockChatModelConfig)

    @property
    def _llm_type(self) -> str:
        return "mock_chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        run_config: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a mock response based on input message content."""
        # Get the last human message
        user_message = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_message = msg.content.lower()
                break

        # Match intent pattern
        intent, response, confidence = self._match_intent(user_message)

        # Build response metadata
        ai_message = AIMessage(
            content=response,
            additional_kwargs={
                "intent": intent,
                "confidence": confidence,
                "user_message_original": user_message,
            }
        )

        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

    def _stream(self, messages: list[BaseMessage], **kwargs: Any) -> Iterator[BaseMessage]:
        """Generate a mock streaming response (word by word)."""
        result = self._generate(messages)
        content = result.generations[0].message.content

        # Stream word by word
        words = content.split()
        for word in words:
            yield AIMessage(content=word + " ")

    def _match_intent(
        self, text: str
    ) -> tuple[str, str, float]:
        """Match user input against intent patterns.

        Returns:
            Tuple of (normalized_intent, response, confidence)
        """
        for pattern in self.config.intent_patterns:
            if any(keyword in text for keyword in pattern.keywords):
                # Normalize intent name
                first_keyword = pattern.keywords[0]
                normalized_intent = INTENT_NAMES.get(first_keyword, first_keyword)
                return (
                    normalized_intent,
                    pattern.response,
                    pattern.confidence,
                )

        # No match - return default
        return (
            "unknown",
            self.config.default_response,
            self.config.default_confidence,
        )

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "MockChatModel":
        """Bind tools to the mock model (no-op for mock)."""
        return self


# Singleton instance for convenience
_default_model: MockChatModel | None = None


def get_mock_model() -> MockChatModel:
    """Get the default mock chat model instance."""
    global _default_model
    if _default_model is None:
        _default_model = MockChatModel()
    return _default_model
