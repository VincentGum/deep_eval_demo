"""Mock LLM with structured reasoning - simulates LLM thought process."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult


# ============================================================================
# Intent Classification System
# ============================================================================

class Intent(Enum):
    """Customer support intent types."""
    ORDER_STATUS = "order_status"
    REFUND = "refund"
    CANCEL = "cancel"
    COMPLAINT = "complaint"
    GREETING = "greeting"
    THANKS = "thanks"
    OFF_TOPIC = "off_topic"
    UNKNOWN = "unknown"


@dataclass
class IntentRule:
    """A rule for intent classification based on semantic features."""
    intent: Intent
    condition: str  # Description of the rule
    # Semantic features required for this intent
    required_features: tuple[str, ...] = ()
    # Semantic features that prevent this intent
    excluded_features: tuple[str, ...] = ()
    base_confidence: float = 0.7


# ============================================================================
# Semantic Feature Extraction
# ============================================================================

@dataclass
class SemanticFeatures:
    """Extracted semantic features from user input."""
    # Action-related
    has_order_reference: bool = False  # Mentions order ID or number
    has_action_verb: bool = False     # Verbs like refund, cancel, track
    action_type: str | None = None    # refund, cancel, track, inquire

    # Emotional indicators
    has_positive_emotion: bool = False   # thank, appreciate
    has_negative_emotion: bool = False  # sorry, angry, disappointed, complain
    is_greeting: bool = False           # greeting words (hello, hi, 你好)

    # Question indicators
    is_question: bool = False
    question_type: str | None = None  # where, when, what, why, how

    # Entity indicators
    has_product_reference: bool = False  # Mentions product/item
    has_time_reference: bool = False    # Mentions time (yesterday, tomorrow)

    # Sentiment
    sentiment: str = "neutral"  # positive, negative, neutral

    # Language detection
    language: str = "en"  # en, zh, mixed

    # Complexity
    is_single_intent: bool = True
    is_multi_turn: bool = False  # Follow-up question


class SemanticAnalyzer:
    """Analyzes semantic features from user input.

    This simulates how an LLM would analyze the semantic content of input
    rather than just pattern matching on keywords.
    """

    # Action verbs and their mappings
    ACTION_VERBS = {
        # Refund-related
        "refund": "refund",
        "reimburse": "refund",
        "退款": "refund",
        "退钱": "refund",
        "退货": "refund",
        # Cancel-related
        "cancel": "cancel",
        "cancellation": "cancel",
        "取消": "cancel",
        "撤销": "cancel",
        # Track-related
        "track": "track",
        "where": "track",
        "status": "track",
        "物流": "track",
        "快递": "track",
        "发货": "track",
        "在哪里": "track",
        "状态": "track",
        "发货": "track",
        # Inquiry-related
        "ask": "inquire",
        "check": "inquire",
        "look": "inquire",
        "show": "inquire",
        "tell": "inquire",
        "get": "inquire",
        "查询": "track",
        "查": "track",
        "问": "track",
        "什么时候": "track",
        # Complaint-related
        "complaint": "complaint",
        "complain": "complaint",
        "投诉": "complaint",
        "issue": "complaint",
        "problem": "complaint",
        "damaged": "complaint",
        "late": "complaint",
        "broken": "complaint",
        "wrong": "complaint",
        "损坏": "complaint",
        "坏了": "complaint",
        "有问题": "complaint",
    }

    # Positive emotion words
    POSITIVE_WORDS = {
        "thank", "thanks", "appreciate", "grateful",
        "谢谢", "感谢", "感激",
    }

    # Chinese greeting words
    CHINESE_GREETINGS = {
        "你好", "您好", "你好呀", "嗨", "hi", "哈喽",
        "早", "早上好", "下午好", "晚上好", "初次见面",
    }

    # Negative emotion words
    NEGATIVE_WORDS = {
        "sorry", "disappointed", "angry", "frustrated", "upset",
        "terrible", "horrible", "worst", "bad",
        "遗憾", "失望", "愤怒", "糟糕",
    }

    # Question words
    QUESTION_WORDS = {
        # English
        "where": "location",
        "when": "time",
        "what": "content",
        "why": "reason",
        "how": "method",
        "can": "ability",
        "could": "ability",
        "is": "confirmation",
        "are": "confirmation",
        # Chinese question indicators
        "天气": "weather",
        "什么时候": "time",
        "在哪里": "location",
        "怎么办": "method",
        "怎么样": "evaluation",
        "怎么": "method",
    }

    # Order ID patterns
    ORDER_PATTERNS = [
        r"order\s*#?\s*[A-Z0-9]{3,}",  # order #A123
        r"订单[号#]?\s*[A-Z0-9]+",      # 订单号A123
        r"#[A-Z0-9]{3,}",               # #A123
    ]

    def analyze(self, text: str) -> SemanticFeatures:
        """Analyze text and extract semantic features.

        This simulates LLM's ability to understand meaning rather than
        just matching keywords.
        """
        text_lower = text.lower()
        text_no_punct = re.sub(r"[^\w\s]", " ", text_lower)
        words = set(text_no_punct.split())

        features = SemanticFeatures()

        # Language detection
        if re.search(r"[\u4e00-\u9fff]", text):
            features.language = "zh" if re.search(r"^[\u4e00-\u9fff]", text) else "mixed"
        else:
            features.language = "en"

        # Order reference detection (semantic understanding)
        for pattern in self.ORDER_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                features.has_order_reference = True
                break

        # Action verb detection with semantic mapping
        for word in words:
            if word in self.ACTION_VERBS:
                features.has_action_verb = True
                if features.action_type is None:
                    features.action_type = self.ACTION_VERBS[word]
            # Also check substrings
            for verb, action in self.ACTION_VERBS.items():
                if verb in word and len(verb) > 3:
                    features.has_action_verb = True
                    if features.action_type is None:
                        features.action_type = action

        # Emotion detection (semantic understanding)
        for word in words:
            if word in self.POSITIVE_WORDS:
                features.has_positive_emotion = True
                features.sentiment = "positive"
            if word in self.NEGATIVE_WORDS:
                features.has_negative_emotion = True
                features.sentiment = "negative"

        # Chinese greeting detection (semantic understanding)
        if features.language == "zh":
            for greeting in self.CHINESE_GREETINGS:
                if greeting in text:
                    features.is_greeting = True
                    break

        # Special handling for complaint indicators
        complaint_indicators = ["damaged", "late", "broken", "wrong", "open", "missing"]
        if any(ind in text_lower for ind in complaint_indicators):
            features.has_negative_emotion = True
            features.sentiment = "negative"
            if features.action_type is None:
                features.action_type = "complaint"

        # Question detection
        if "?" in text or any(q in text_lower for q in ["吗", "呢", "怎么", "是不是"]):
            features.is_question = True

        # Question type detection
        for qword, qtype in self.QUESTION_WORDS.items():
            if qword in text_lower:
                features.question_type = qtype
                break

        # Time reference
        time_words = ["yesterday", "tomorrow", "today", "last week", "next week",
                      "昨天", "今天", "明天", "上周", "下周"]
        if any(t in text_lower for t in time_words):
            features.has_time_reference = True

        # Product reference
        product_words = ["item", "product", "package", "goods", "商品", "物品", "包裹"]
        if any(p in text_lower for p in product_words):
            features.has_product_reference = True

        return features


# ============================================================================
# Intent Reasoning Engine
# ============================================================================

class IntentReasoner:
    """Reasons about user intent based on semantic features.

    This simulates how an LLM would reason about intent based on
    understanding the meaning of the input.
    """

    def __init__(self):
        self.analyzer = SemanticAnalyzer()

        # Define classification rules
        self.rules = [
            # THANKS - explicit positive emotion
            IntentRule(
                intent=Intent.THANKS,
                condition="User expresses gratitude",
                excluded_features=("has_action_verb",),
            ),
            # GREETING - simple greeting without action
            IntentRule(
                intent=Intent.GREETING,
                condition="User greets without specific request",
            ),
            # REFUND - action verb is refund OR negative emotion + action
            IntentRule(
                intent=Intent.REFUND,
                condition="User requests refund or reimbursement",
                required_features=("has_action_verb",),
            ),
            # CANCEL - action verb is cancel
            IntentRule(
                intent=Intent.CANCEL,
                condition="User requests cancellation",
                required_features=("has_action_verb",),
            ),
            # COMPLAINT - negative emotion or problem indicators
            IntentRule(
                intent=Intent.COMPLAINT,
                condition="User reports a problem or expresses dissatisfaction",
                required_features=("has_negative_emotion",),
            ),
            # ORDER_STATUS - has order reference or track/inquire action
            IntentRule(
                intent=Intent.ORDER_STATUS,
                condition="User asks about order status or location",
                required_features=("has_order_reference", "has_action_verb"),
            ),
            # OFF_TOPIC - unrelated to customer service
            IntentRule(
                intent=Intent.OFF_TOPIC,
                condition="Query is unrelated to orders or customer service",
            ),
        ]

    def reason(self, text: str) -> tuple[Intent, float, str]:
        """Reason about user intent from input text.

        Returns:
            Tuple of (intent, confidence, reasoning)
        """
        features = self.analyzer.analyze(text)

        # Build feature set for matching
        feature_set = set()
        if features.has_order_reference:
            feature_set.add("has_order_reference")
        if features.has_action_verb:
            feature_set.add("has_action_verb")
        if features.has_positive_emotion:
            feature_set.add("has_positive_emotion")
        if features.has_negative_emotion:
            feature_set.add("has_negative_emotion")
        if features.is_question:
            feature_set.add("is_question")
        if features.question_type:
            feature_set.add(f"question_{features.question_type}")

        # Reasoning trace
        reasoning_parts = []

        # Step 1: Check for explicit positive emotion (thanks)
        if features.has_positive_emotion and not features.has_action_verb:
            reasoning_parts.append(
                f"Detected positive emotion ('thanks') without action verb. "
                f"Classifying as THANKS intent."
            )
            return Intent.THANKS, 0.95, "\n".join(reasoning_parts)

        # Step 2: Check for simple greeting
        # Skip this for Chinese questions - they might be generic questions
        if features.language == "en" and len(text.split()) <= 5 and not features.has_action_verb:
            # Short English message without action
            if features.sentiment == "positive":
                reasoning_parts.append(
                    f"Short message ({len(text.split())} words) with positive sentiment. "
                    f"Classifying as GREETING intent."
                )
                return Intent.GREETING, 0.90, "\n".join(reasoning_parts)
            elif features.sentiment == "neutral":
                reasoning_parts.append(
                    f"Short message ({len(text.split())} words) with neutral sentiment. "
                    f"Classifying as GREETING intent."
                )
                return Intent.GREETING, 0.85, "\n".join(reasoning_parts)

        # Step 2a: Check for Chinese greeting
        if features.is_greeting and features.language == "zh":
            reasoning_parts.append(
                f"Detected Chinese greeting word from CHINESE_GREETINGS list. "
                f"Classifying as GREETING intent."
            )
            return Intent.GREETING, 0.90, "\n".join(reasoning_parts)

        # Step 2b: Check for Chinese generic questions
        if features.language == "zh" and features.is_question:
            generic_keywords = ["天气", "时间", "怎么样", "怎么办", "好不好", "能不能"]
            if any(kw in text for kw in generic_keywords):
                reasoning_parts.append(
                    f"Detected Chinese generic question with keywords: {generic_keywords}. "
                    f"Classifying as OFF_TOPIC intent (not order-related)."
                )
                return Intent.OFF_TOPIC, 0.85, "\n".join(reasoning_parts)

        # Step 3: Analyze action type
        if features.action_type:
            if features.action_type == "refund":
                reasoning_parts.append(
                    f"Action verb detected: '{features.action_type}'. "
                    f"Checking if refund request requires human review..."
                )
                # Refunds typically need human review
                confidence = 0.85
                if features.has_negative_emotion:
                    confidence = 0.90
                return Intent.REFUND, confidence, "\n".join(reasoning_parts)

            elif features.action_type == "cancel":
                reasoning_parts.append(
                    f"Action verb detected: '{features.action_type}'. "
                    f"Checking cancellation request..."
                )
                return Intent.CANCEL, 0.85, "\n".join(reasoning_parts)

            elif features.action_type == "complaint":
                reasoning_parts.append(
                    f"Problem indicators detected (complaint context). "
                    f"Classifying as COMPLAINT intent."
                )
                return Intent.COMPLAINT, 0.90, "\n".join(reasoning_parts)

            elif features.action_type in ("track", "inquire"):
                reasoning_parts.append(
                    f"Action verb detected: '{features.action_type}'. "
                    f"User is inquiring about order status."
                )
                if features.has_order_reference:
                    reasoning_parts.append(
                        f"Order reference detected. High confidence for ORDER_STATUS."
                    )
                    return Intent.ORDER_STATUS, 0.90, "\n".join(reasoning_parts)
                else:
                    reasoning_parts.append(
                        f"No order reference found. Assuming ORDER_STATUS inquiry."
                    )
                    return Intent.ORDER_STATUS, 0.75, "\n".join(reasoning_parts)

        # Step 4: Check for product inquiry (not order-related)
        product_indicators = ["product id", "do you have", "in stock", "color", "size", "item #", "item id"]
        if any(ind in text.lower() for ind in product_indicators):
            if not features.has_order_reference and not features.has_action_verb:
                reasoning_parts.append(
                    f"Product inquiry indicators detected but no order reference. "
                    f"Classifying as OFF_TOPIC intent (not order-related)."
                )
                return Intent.OFF_TOPIC, 0.85, "\n".join(reasoning_parts)

        # Step 5: Check for problem indicators
        if features.has_negative_emotion:
            reasoning_parts.append(
                f"Negative emotion detected. "
                f"Classifying as COMPLAINT intent."
            )
            return Intent.COMPLAINT, 0.85, "\n".join(reasoning_parts)

        # Step 5: Check for question patterns
        if features.is_question and features.question_type == "location":
            if features.has_order_reference:
                reasoning_parts.append(
                    f"Question about location with order reference. "
                    f"Classifying as ORDER_STATUS intent."
                )
                return Intent.ORDER_STATUS, 0.88, "\n".join(reasoning_parts)

        # Step 6: Check for off-topic
        off_topic_words = ["weather", "news", "time", "天气", "新闻", "时间"]
        if any(word in text.lower() for word in off_topic_words):
            if not features.has_action_verb:
                reasoning_parts.append(
                    f"Off-topic keywords detected without action verb. "
                    f"Classifying as OFF_TOPIC intent."
                )
                return Intent.OFF_TOPIC, 0.90, "\n".join(reasoning_parts)

        # Step 7: Default to ORDER_STATUS if there's any question
        if features.is_question:
            reasoning_parts.append(
                f"Question detected but no clear action. "
                f"Defaulting to ORDER_STATUS inquiry."
            )
            return Intent.ORDER_STATUS, 0.70, "\n".join(reasoning_parts)

        # Default: UNKNOWN
        reasoning_parts.append(
            f"No clear intent pattern matched. "
            f"Classifying as UNKNOWN intent."
        )
        return Intent.UNKNOWN, 0.50, "\n".join(reasoning_parts)


# ============================================================================
# Response Generator
# ============================================================================

class ResponseGenerator:
    """Generates responses based on intent and context."""

    # Response templates
    TEMPLATES = {
        Intent.ORDER_STATUS: {
            "with_order": "I found your order {order_id}. Let me look up the current status for you.",
            "without_order": "I'd be happy to help you check your order status. Could you please provide your order ID?",
            "tracking": "I can see your order {order_id} is currently in transit. It should arrive within 2-3 business days.",
        },
        Intent.REFUND: {
            "standard": "I understand you need a refund for {order_info}. I'm creating a refund request for you. This will require human review before processing.",
            "emotional": "I'm sorry to hear you're having this issue. Let me help you process a refund for {order_info}. This will need human review.",
        },
        Intent.CANCEL: {
            "standard": "I understand you want to cancel {order_info}. Let me create a cancellation request for you. This will require human review before processing.",
            "urgent": "I see this is urgent. Let me create a cancellation request for {order_info} right away. This will need human review.",
        },
        Intent.COMPLAINT: {
            "standard": "I'm sorry to hear about your experience with {order_info}. Let me create a complaint case for you so our team can address this issue. This will require human review.",
            "product_issue": "I'm sorry to hear the product arrived damaged. Let me create a complaint case for {order_info}. Our quality team will review this. This requires human review.",
        },
        Intent.GREETING: {
            "en": "Hello! I'm here to help you with your order inquiries. How can I assist you today?",
            "zh": "您好！我可以帮助您查询订单相关的问题。请问有什么可以帮到您的？",
            "mixed": "Hello! 我可以帮助您处理订单相关的问题。How can I help you today?",
        },
        Intent.THANKS: {
            "en": "You're welcome! Is there anything else I can help you with?",
            "zh": "不客气！还有什么可以帮到您的吗？",
            "mixed": "You're welcome! 还有什么需要帮助的吗？",
        },
        Intent.OFF_TOPIC: {
            "en": "I'm a customer service assistant, I can help you with order-related questions. For other inquiries, please visit our main website.",
            "zh": "我是客服助手，可以帮助您处理订单相关的问题。其他问题请访问我们的官网。",
        },
        Intent.UNKNOWN: {
            "en": "I'm not sure I understood that correctly. Could you please provide more details about your request?",
            "zh": "我没有完全理解您的问题。请您提供更多详细信息，好吗？",
        },
    }

    def generate(
        self,
        intent: Intent,
        features: SemanticFeatures,
        reasoning: str,
        order_id: str | None = None,
    ) -> str:
        """Generate a response based on intent and context."""
        templates = self.TEMPLATES.get(intent, self.TEMPLATES[Intent.UNKNOWN])

        # Get base template
        if isinstance(templates, dict):
            if "emotional" in templates and features.has_negative_emotion:
                template = templates["emotional"]
            elif "urgent" in templates and "immediately" in features.question_type:
                template = templates["urgent"]
            elif "tracking" in templates and features.question_type == "location":
                template = templates["tracking"]
            elif "with_order" in templates and order_id:
                template = templates["with_order"]
            elif "without_order" in templates:
                template = templates["without_order"]
            else:
                template = templates.get("standard", templates.get("en", ""))
        else:
            template = templates

        # Fill in order_id if present
        if order_id:
            if "{order_id}" in template:
                template = template.format(order_id=order_id)
            elif "{order_info}" in template:
                template = template.format(order_info=f"order {order_id}")
            else:
                template = f"{template} (Order: {order_id})"
        else:
            # If template has {order_info} but no order_id, use a different message
            if "{order_info}" in template:
                return "I understand you need help with an order. Could you please provide your order ID so I can assist you better?"
            elif "{order_id}" in template:
                template = template.format(order_id="your order")

        return template


# ============================================================================
# Mock Chat Model
# ============================================================================

@dataclass
class MockLLMConfig:
    """Configuration for MockChatModel."""
    show_reasoning: bool = True  # Include reasoning in response metadata
    language: str = "en"  # Preferred response language


class MockChatModel(BaseChatModel):
    """A structured reasoning mock chat model.

    This mock simulates LLM responses by:
    1. Analyzing semantic features of the input
    2. Reasoning about the user's intent
    3. Generating appropriate responses

    Unlike keyword matching, this approach simulates actual LLM understanding.
    """

    model_name: str = "mock-customer-support-v2"
    config: MockLLMConfig = field(default_factory=MockLLMConfig)

    @property
    def _reasoner(self) -> IntentReasoner:
        """Lazily create and cache IntentReasoner."""
        if not hasattr(self, '_cached_reasoner'):
            self._cached_reasoner = IntentReasoner()
        return self._cached_reasoner

    @property
    def _generator(self) -> ResponseGenerator:
        """Lazily create and cache ResponseGenerator."""
        if not hasattr(self, '_cached_generator'):
            self._cached_generator = ResponseGenerator()
        return self._cached_generator

    def __init__(self, config: MockLLMConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        if config:
            self.config = config

    @property
    def _llm_type(self) -> str:
        return "mock_reasoning"

    def _generate(
        self,
        messages: list[BaseMessage],
        run_config: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a mock response using structured reasoning."""
        # Get the last human message
        user_message = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break

        # Extract order ID if present
        order_id = self._extract_order_id(user_message)

        # Reason about intent
        intent, confidence, reasoning = self._reasoner.reason(user_message)

        # Generate response
        features = self._reasoner.analyzer.analyze(user_message)
        response = self._generator.generate(
            intent=intent,
            features=features,
            reasoning=reasoning,
            order_id=order_id,
        )

        # Build response with metadata
        additional_kwargs = {
            "intent": intent.value,
            "confidence": confidence,
            "reasoning": reasoning if self.config.show_reasoning else None,
            "order_id": order_id,
            "features": {
                "has_order_reference": features.has_order_reference,
                "has_action_verb": features.has_action_verb,
                "action_type": features.action_type,
                "sentiment": features.sentiment,
                "language": features.language,
            },
        }

        ai_message = AIMessage(content=response, additional_kwargs=additional_kwargs)
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

    def _extract_order_id(self, text: str) -> str | None:
        """Extract order ID from text."""
        patterns = [
            r"order\s*#?\s*([A-Z0-9]{3,})",
            r"订单[号#]?\s*([A-Z0-9]+)",
            r"#([A-Z0-9]{3,})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    def _stream(self, messages: list[BaseMessage], **kwargs: Any) -> Iterator[BaseMessage]:
        """Generate a mock streaming response."""
        result = self._generate(messages)
        content = result.generations[0].message.content
        for word in content.split():
            yield AIMessage(content=word + " ")

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "MockChatModel":
        """Bind tools to the mock model (no-op for mock)."""
        return self


# ============================================================================
# Singleton Instance
# ============================================================================

_default_model: MockChatModel | None = None


def get_mock_model() -> MockChatModel:
    """Get the default mock chat model instance."""
    global _default_model
    if _default_model is None:
        _default_model = MockChatModel()
    return _default_model


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    # Demo of structured reasoning
    model = MockChatModel()

    test_cases = [
        "Where is my order #A100?",
        "I want to request a refund",
        "My order #B200 was damaged",
        "Thank you so much!",
        "Hello, I need some help",
        "I need to cancel order #A100 immediately",
        "Can you tell me the status?",
        "我的订单什么时候发货？",
    ]

    print("=" * 80)
    print("Structured Reasoning Demo")
    print("=" * 80)

    for text in test_cases:
        print(f"\nInput: {text}")
        print("-" * 40)
        result = model.invoke([HumanMessage(content=text)])
        meta = result.additional_kwargs
        print(f"Intent: {meta['intent']}")
        print(f"Confidence: {meta['confidence']}")
        print(f"Order ID: {meta['order_id']}")
        if meta.get('reasoning'):
            print(f"Reasoning:\n{meta['reasoning']}")
        print(f"Response: {result.content}")
