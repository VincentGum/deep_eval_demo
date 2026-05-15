"""结构化推理 Mock LLM - 模拟 LLM 的思考过程。

本模块实现了一个结构化的 Mock LLM，用于在没有真实 LLM API 的情况下
模拟 LLM 的理解和推理能力。与简单的关键词匹配不同，这里采用：

1. 语义特征提取 (SemanticAnalyzer)
   - 识别动作动词、情感词汇、问题类型等
   - 支持中英文双语

2. 意图推理 (IntentReasoner)
   - 基于语义特征进行多步推理
   - 输出推理过程（用于调试）

3. 回复生成 (ResponseGenerator)
   - 基于意图和上下文生成模板化回复
   - 支持多语言

4. Mock Chat Model
   - 实现 LangChain BaseChatModel 接口
   - 模拟完整的 LLM 调用流程
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult


# =============================================================================
# 意图分类系统 - 定义客服场景中的用户意图类型
# =============================================================================

class Intent(Enum):
    """客服场景中的用户意图类型枚举。
    
    每个枚举值代表一种典型的用户请求类型。
    """
    ORDER_STATUS = "order_status"  # 订单状态查询
    REFUND = "refund"              # 退款请求
    CANCEL = "cancel"              # 取消订单
    COMPLAINT = "complaint"        # 投诉
    GREETING = "greeting"          # 问候/打招呼
    THANKS = "thanks"              # 感谢
    OFF_TOPIC = "off_topic"        # 超出服务范围
    UNKNOWN = "unknown"            # 未知意图


@dataclass
class IntentRule:
    """意图分类规则。
    
    定义一个意图所需的语义特征条件。
    
    Attributes:
        intent: 意图类型
        condition: 规则描述（用于文档）
        required_features: 必须具备的语义特征
        excluded_features: 不能具备的语义特征
        base_confidence: 基础置信度
    """
    intent: Intent
    condition: str
    required_features: tuple[str, ...] = ()      # 必须具备的特征
    excluded_features: tuple[str, ...] = ()       # 不能具备的特征
    base_confidence: float = 0.7                  # 基础置信度


# =============================================================================
# 语义特征提取 - 从用户输入中提取结构化的语义信息
# =============================================================================

@dataclass
class SemanticFeatures:
    """从用户输入中提取的语义特征。
    
    这些特征用于后续的意图推理，支持中英文双语。
    
    Attributes:
        has_order_reference: 是否提及订单号
        has_action_verb: 是否包含动作动词（退款、取消等）
        action_type: 动作类型（refund, cancel, track, inquire, complaint）
        has_positive_emotion: 是否有积极情绪
        has_negative_emotion: 是否有消极情绪
        is_greeting: 是否是问候语
        is_question: 是否是问句
        question_type: 问题类型（where, when, what, why, how）
        has_product_reference: 是否提及产品
        has_time_reference: 是否提及时间
        sentiment: 情感倾向（positive, negative, neutral）
        language: 语言类型（en, zh, mixed）
        is_single_intent: 是否单一意图
        is_multi_turn: 是否是多轮对话
    """
    # 动作相关
    has_order_reference: bool = False   # 提及订单号
    has_action_verb: bool = False       # 动作动词
    action_type: str | None = None      # 动作类型

    # 情感指标
    has_positive_emotion: bool = False  # 积极情绪
    has_negative_emotion: bool = False  # 消极情绪
    is_greeting: bool = False           # 问候语

    # 问题指标
    is_question: bool = False
    question_type: str | None = None    # where, when, what, why, how

    # 实体指标
    has_product_reference: bool = False  # 提及产品
    has_time_reference: bool = False     # 提及时间

    # 情感
    sentiment: str = "neutral"          # 情感倾向

    # 语言
    language: str = "en"                # en, zh, mixed

    # 复杂性
    is_single_intent: bool = True
    is_multi_turn: bool = False


class SemanticAnalyzer:
    """语义特征分析器。
    
    模拟 LLM 对输入文本的语义理解能力，提取结构化的特征信息。
    与简单的关键词匹配不同，这里尝试理解语义。
    """
    
    # 动作动词及其映射
    # 退款相关动词
    ACTION_VERBS = {
        "refund": "refund",
        "reimburse": "refund",
        "退款": "refund",
        "退钱": "refund",
        "退货": "refund",
        "return": "refund",
        "charged": "refund",
        "charge": "refund",
        "overcharge": "refund",
        "double": "refund",
        
        # 取消相关动词
        "cancel": "cancel",
        "cancellation": "cancel",
        "取消": "cancel",
        "撤销": "cancel",
        
        # 跟踪相关动词
        "track": "track",
        "where": "track",
        "status": "track",
        "物流": "track",
        "快递": "track",
        "发货": "track",
        "在哪里": "track",
        "状态": "track",
        "查询": "track",
        "查": "track",
        "问": "track",
        "什么时候": "track",
        
        # 询问相关动词
        "ask": "inquire",
        "check": "inquire",
        "look": "inquire",
        "show": "inquire",
        "tell": "inquire",
        "get": "inquire",
        
        # 投诉相关动词
        "complaint": "complaint",
        "complain": "complaint",
        "投诉": "complaint",
        "issue": "complaint",
        "problem": "complaint",
        "damaged": "complaint",
        "late": "complaint",
        "broken": "complaint",
        "wrong": "complaint",
        "replacement": "complaint",
        "replace": "complaint",
        "defective": "complaint",
        "crushed": "complaint",
        "missing": "complaint",
        "损坏": "complaint",
        "坏了": "complaint",
        "有问题": "complaint",
    }

    # 积极情绪词汇
    POSITIVE_WORDS = {
        "thank", "thanks", "appreciate", "grateful",
        "谢谢", "感谢", "感激",
    }

    # 中文问候词汇
    CHINESE_GREETINGS = {
        "你好", "您好", "你好呀", "嗨", "hi", "哈喽",
        "早", "早上好", "下午好", "晚上好", "初次见面",
    }

    # 消极情绪词汇
    NEGATIVE_WORDS = {
        "sorry", "disappointed", "angry", "frustrated", "upset",
        "terrible", "horrible", "worst", "bad",
        "遗憾", "失望", "愤怒", "糟糕",
    }

    # 疑问词及其类型映射
    QUESTION_WORDS = {
        # 英文疑问词
        "where": "location",
        "when": "time",
        "what": "content",
        "why": "reason",
        "how": "method",
        "can": "ability",
        "could": "ability",
        "is": "confirmation",
        "are": "confirmation",
        
        # 中文疑问词
        "天气": "weather",
        "什么时候": "time",
        "在哪里": "location",
        "怎么办": "method",
        "怎么样": "evaluation",
        "怎么": "method",
    }

    # 订单号正则模式
    ORDER_PATTERNS = [
        r"order\s*#?\s*[A-Z0-9]{3,}",  # order #A123
        r"订单[号#]?\s*[A-Z0-9]+",      # 订单号A123
        r"#[A-Z0-9]{3,}",               # #A123
    ]

    def analyze(self, text: str) -> SemanticFeatures:
        """分析文本并提取语义特征。
        
        模拟 LLM 的语义理解能力，通过分析词汇和模式提取特征。
        
        Args:
            text: 输入文本
            
        Returns:
            SemanticFeatures 对象，包含提取的特征
        """
        text_lower = text.lower()
        text_no_punct = re.sub(r"[^\w\s]", " ", text_lower)
        words_unique = set(text_no_punct.split())
        words_ordered = text_no_punct.split()  # 保持顺序用于动作动词检测

        features = SemanticFeatures()

        # ---- 语言检测 ----
        if re.search(r"[\u4e00-\u9fff]", text):
            # 检测到中文字符
            features.language = "zh" if re.search(r"^[\u4e00-\u9fff]", text) else "mixed"
        else:
            features.language = "en"

        # ---- 订单号检测 ----
        for pattern in self.ORDER_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                features.has_order_reference = True
                break

        # ---- 动作动词检测（高优先级动作覆盖低优先级） ----
        # 优先级：refund > cancel > complaint > inquire > help > order > check > policy
        ACTION_PRIORITY = {"refund": 5, "cancel": 4, "complaint": 3, "inquire": 1, "help": 1, "order": 2, "check": 2, "policy": 2}
        detected_actions = []  # (priority, action_type)
        
        for word in words_ordered:
            if word in self.ACTION_VERBS:
                features.has_action_verb = True
                action = self.ACTION_VERBS[word]
                priority = ACTION_PRIORITY.get(action, 0)
                detected_actions.append((priority, action))
            # 检查子串匹配（处理复合词）
            for verb, action in self.ACTION_VERBS.items():
                if verb in word and len(verb) > 3:
                    features.has_action_verb = True
                    priority = ACTION_PRIORITY.get(action, 0)
                    detected_actions.append((priority, action))
        
        # 选择优先级最高的动作
        if detected_actions:
            detected_actions.sort(key=lambda x: x[0], reverse=True)
            features.action_type = detected_actions[0][1]
        
        # 中文 2 字动词检测（退款、取消 等）
        if features.language == "zh":
            for verb, action in self.ACTION_VERBS.items():
                if len(verb) == 2 and verb in text:
                    features.has_action_verb = True
                    zh_priority = ACTION_PRIORITY.get(action, 0)
                    curr_priority = ACTION_PRIORITY.get(features.action_type, 0) if features.action_type else 0
                    if zh_priority > curr_priority:
                        features.action_type = action

        # ---- 情感检测 ----
        for word in words_unique:
            if word in self.POSITIVE_WORDS:
                features.has_positive_emotion = True
                features.sentiment = "positive"
            if word in self.NEGATIVE_WORDS:
                features.has_negative_emotion = True
                features.sentiment = "negative"

        # ---- 中文问候语检测 ----
        if features.language == "zh":
            for greeting in self.CHINESE_GREETINGS:
                if greeting in text:
                    features.is_greeting = True
                    break

        # ---- 投诉指标检测 ----
        complaint_indicators = ["damaged", "late", "broken", "wrong", "open", "missing",
                                "crushed", "defective", "unacceptable", "terrible", "awful",
                                "disappointed", "frustrated", "angry", "upset", "poor quality",
                                "bad quality", "not working", "doesn't work", "not satisfied",
                                "dented", "scratched", "torn", "leaked", "lost"]
        if any(ind in text_lower for ind in complaint_indicators):
            features.has_negative_emotion = True
            features.sentiment = "negative"
            if features.action_type is None:
                features.action_type = "complaint"

        # ---- 问题检测 ----
        if "?" in text or any(q in text_lower for q in ["吗", "呢", "怎么", "是不是"]):
            features.is_question = True

        # ---- 问题类型检测 ----
        for qword, qtype in self.QUESTION_WORDS.items():
            if qword in text_lower:
                features.question_type = qtype
                break

        # ---- 时间引用检测 ----
        time_words = ["yesterday", "tomorrow", "today", "last week", "next week",
                      "昨天", "今天", "明天", "上周", "下周"]
        if any(t in text_lower for t in time_words):
            features.has_time_reference = True

        # ---- 产品引用检测 ----
        product_words = ["item", "product", "package", "goods", "商品", "物品", "包裹"]
        if any(p in text_lower for p in product_words):
            features.has_product_reference = True

        return features


# =============================================================================
# 意图推理引擎 - 基于语义特征进行多步推理
# =============================================================================

class IntentReasoner:
    """意图推理器。
    
    基于语义特征进行多步推理，模拟 LLM 的思考过程。
    每一步推理都会生成详细的推理轨迹（reasoning）。
    """
    
    def __init__(self):
        self.analyzer = SemanticAnalyzer()
        
        # 定义意图分类规则
        self.rules = [
            # 感谢意图：表达感谢且无动作动词
            IntentRule(
                intent=Intent.THANKS,
                condition="User expresses gratitude",
                excluded_features=("has_action_verb",),
            ),
            # 问候意图：简单问候，无特定请求
            IntentRule(
                intent=Intent.GREETING,
                condition="User greets without specific request",
            ),
            # 退款意图：有动作动词且是退款相关
            IntentRule(
                intent=Intent.REFUND,
                condition="User requests refund or reimbursement",
                required_features=("has_action_verb",),
            ),
            # 取消意图：动作动词是取消
            IntentRule(
                intent=Intent.CANCEL,
                condition="User requests cancellation",
                required_features=("has_action_verb",),
            ),
            # 投诉意图：有消极情绪或问题指标
            IntentRule(
                intent=Intent.COMPLAINT,
                condition="User reports a problem or expresses dissatisfaction",
                required_features=("has_negative_emotion",),
            ),
            # 订单状态意图：有订单引用或跟踪动作
            IntentRule(
                intent=Intent.ORDER_STATUS,
                condition="User asks about order status or location",
                required_features=("has_order_reference", "has_action_verb"),
            ),
            # 超出范围意图：与客服无关的查询
            IntentRule(
                intent=Intent.OFF_TOPIC,
                condition="Query is unrelated to orders or customer service",
            ),
        ]

    def reason(self, text: str) -> tuple[Intent, float, str]:
        """推理用户意图。
        
        通过多步检查，基于语义特征确定用户意图。
        
        Args:
            text: 用户输入文本
            
        Returns:
            三元组 (意图类型, 置信度, 推理过程)
        """
        features = self.analyzer.analyze(text)

        # 构建特征集合
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

        # 推理过程记录
        reasoning_parts = []

        # ---- 第1步：检测感谢 ----
        if features.has_positive_emotion and not features.has_action_verb:
            reasoning_parts.append(
                f"Detected positive emotion ('thanks') without action verb. "
                f"Classifying as THANKS intent."
            )
            return Intent.THANKS, 0.95, "\n".join(reasoning_parts)

        # ---- 第2步：检测问候（英文短消息）----
        if features.language == "en" and len(text.split()) <= 5 and not features.has_action_verb:
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

        # ---- 第2a步：检测中文问候 ----
        if features.is_greeting and features.language == "zh":
            reasoning_parts.append(
                f"Detected Chinese greeting word from CHINESE_GREETINGS list. "
                f"Classifying as GREETING intent."
            )
            return Intent.GREETING, 0.90, "\n".join(reasoning_parts)

        # ---- 第2b步：检测中文通用问题 ----
        if features.language == "zh" and features.is_question:
            generic_keywords = ["天气", "时间", "怎么样", "怎么办", "好不好", "能不能"]
            if any(kw in text for kw in generic_keywords):
                reasoning_parts.append(
                    f"Detected Chinese generic question with keywords: {generic_keywords}. "
                    f"Classifying as OFF_TOPIC intent (not order-related)."
                )
                return Intent.OFF_TOPIC, 0.85, "\n".join(reasoning_parts)

        # ---- 第3步：分析动作类型 ----
        if features.action_type:
            if features.action_type == "refund":
                reasoning_parts.append(
                    f"Action verb detected: '{features.action_type}'. "
                    f"Checking if refund request requires human review..."
                )
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

        # ---- 第4步：检查产品咨询（超出范围）----
        product_indicators = ["product id", "do you have", "in stock", "color", "size", "item #", "item id"]
        if any(ind in text.lower() for ind in product_indicators):
            if not features.has_order_reference and not features.has_action_verb:
                reasoning_parts.append(
                    f"Product inquiry indicators detected but no order reference. "
                    f"Classifying as OFF_TOPIC intent (not order-related)."
                )
                return Intent.OFF_TOPIC, 0.85, "\n".join(reasoning_parts)

        # ---- 第5步：检查消极情绪 ----
        if features.has_negative_emotion:
            reasoning_parts.append(
                f"Negative emotion detected. "
                f"Classifying as COMPLAINT intent."
            )
            return Intent.COMPLAINT, 0.85, "\n".join(reasoning_parts)

        # ---- 第5b步：检查位置问题 ----
        if features.is_question and features.question_type == "location":
            if features.has_order_reference:
                reasoning_parts.append(
                    f"Question about location with order reference. "
                    f"Classifying as ORDER_STATUS intent."
                )
                return Intent.ORDER_STATUS, 0.88, "\n".join(reasoning_parts)

        # ---- 第6步：检查超出范围关键词 ----
        off_topic_words = ["weather", "news", "time", "天气", "新闻", "时间"]
        if any(word in text.lower() for word in off_topic_words):
            if not features.has_action_verb:
                reasoning_parts.append(
                    f"Off-topic keywords detected without action verb. "
                    f"Classifying as OFF_TOPIC intent."
                )
                return Intent.OFF_TOPIC, 0.90, "\n".join(reasoning_parts)

        # ---- 第7步：问句默认查询订单 ----
        if features.is_question:
            reasoning_parts.append(
                f"Question detected but no clear action. "
                f"Defaulting to ORDER_STATUS inquiry."
            )
            return Intent.ORDER_STATUS, 0.70, "\n".join(reasoning_parts)

        # ---- 默认：未知意图 ----
        reasoning_parts.append(
            f"No clear intent pattern matched. "
            f"Classifying as UNKNOWN intent."
        )
        return Intent.UNKNOWN, 0.50, "\n".join(reasoning_parts)


# =============================================================================
# 回复生成器 - 基于意图和上下文生成回复
# =============================================================================

class ResponseGenerator:
    """回复生成器。
    
    基于意图类型和上下文信息，从模板库中选择合适的回复模板。
    支持根据情感状态、语言类型等选择不同的模板变体。
    """
    
    # 回复模板库
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
            "standard": "I'm sorry to hear about your experience with {order_info}. Let me create a complaint case for you so our team can address this issue.",
            "product_issue": "I'm sorry to hear the product arrived damaged. Let me create a complaint case for {order_info}. Our quality team will review this.",
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
        """生成回复。
        
        基于意图、语义特征和推理过程，从模板库中选择合适的回复。
        
        Args:
            intent: 识别的意图类型
            features: 语义特征
            reasoning: 推理过程（用于日志）
            order_id: 提取的订单号
            
        Returns:
            生成的回复文本
        """
        templates = self.TEMPLATES.get(intent, self.TEMPLATES[Intent.UNKNOWN])

        # 选择合适的模板变体
        question_type = features.question_type or ""
        if isinstance(templates, dict):
            if "emotional" in templates and features.has_negative_emotion:
                template = templates["emotional"]
            elif "urgent" in templates and "immediately" in question_type:
                template = templates["urgent"]
            elif "tracking" in templates and question_type == "location":
                template = templates["tracking"]
            elif "with_order" in templates and order_id:
                template = templates["with_order"]
            elif "without_order" in templates:
                template = templates["without_order"]
            else:
                # 回退到标准模板或英文模板
                template = templates.get("standard", templates.get("en", ""))
        else:
            template = templates

        # 填充订单信息
        if order_id:
            if "{order_id}" in template:
                template = template.format(order_id=order_id)
            elif "{order_info}" in template:
                template = template.format(order_info=f"order {order_id}")
            else:
                template = f"{template} (Order: {order_id})"
        else:
            # 没有订单号时的特殊处理
            if "{order_info}" in template:
                # 根据意图类型生成不同的回退消息
                if intent == Intent.REFUND:
                    return "I understand you need a refund. Could you please provide your order ID so I can process your refund request?"
                elif intent == Intent.CANCEL:
                    return "I understand you want to cancel an order. Could you please provide your order ID so I can process your cancellation?"
                elif intent == Intent.COMPLAINT:
                    return "I'm sorry to hear about your issue. Let me create a complaint case for you. Could you please provide your order ID so our team can address this?"
                else:
                    return "I understand you need help with an order. Could you please provide your order ID so I can assist you better?"
            elif "{order_id}" in template:
                template = template.format(order_id="your order")

        return template


# =============================================================================
# Mock Chat Model - 实现 LangChain BaseChatModel 接口
# =============================================================================

@dataclass
class MockLLMConfig:
    """MockChatModel 的配置类。"""
    show_reasoning: bool = True   # 是否在响应元数据中包含推理过程
    language: str = "en"          # 偏好的回复语言


class MockChatModel(BaseChatModel):
    """结构化推理 Mock Chat Model。
    
    实现 LangChain 的 BaseChatModel 接口，模拟完整的 LLM 调用流程：
    1. 分析输入的语义特征
    2. 推理用户意图
    3. 生成合适的回复
    
    与简单的关键词匹配不同，这种方法模拟了 LLM 的理解能力。
    
    Attributes:
        model_name: 模型名称
        config: 配置对象
    """
    
    model_name: str = "mock-customer-support-v2"
    config: MockLLMConfig = field(default_factory=MockLLMConfig)

    @property
    def _reasoner(self) -> IntentReasoner:
        """延迟创建并缓存 IntentReasoner。"""
        if not hasattr(self, '_cached_reasoner'):
            self._cached_reasoner = IntentReasoner()
        return self._cached_reasoner

    @property
    def _generator(self) -> ResponseGenerator:
        """延迟创建并缓存 ResponseGenerator。"""
        if not hasattr(self, '_cached_generator'):
            self._cached_generator = ResponseGenerator()
        return self._cached_generator

    def __init__(self, config: MockLLMConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        if config:
            self.config = config

    @property
    def _llm_type(self) -> str:
        """返回模型类型标识。"""
        return "mock_reasoning"

    def _generate(
        self,
        messages: list[BaseMessage],
        run_config: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """生成 Mock 响应。
        
        主要流程：
        1. 提取用户消息
        2. 提取订单号
        3. 推理用户意图
        4. 生成回复
        5. 构建带元数据的响应
        """
        # 获取最后一条用户消息
        user_message = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break

        # 提取订单号
        order_id = self._extract_order_id(user_message)

        # 推理意图
        intent, confidence, reasoning = self._reasoner.reason(user_message)

        # 生成回复
        features = self._reasoner.analyzer.analyze(user_message)
        response = self._generator.generate(
            intent=intent,
            features=features,
            reasoning=reasoning,
            order_id=order_id,
        )

        # 构建带元数据的响应
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
        """从文本中提取订单号。"""
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
        """生成 Mock 流式响应。"""
        result = self._generate(messages)
        content = result.generations[0].message.content
        for word in content.split():
            yield AIMessage(content=word + " ")

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "MockChatModel":
        """绑定工具到模型（Mock 实现为空操作）。"""
        return self


# =============================================================================
# 结构化推理 LLM - 整合所有组件
# =============================================================================

class StructuredReasoningLLM:
    """
    结构化推理 LLM - 整合语义分析、意图分类和响应生成的完整推理链
    
    【组件构成】
    - SemanticAnalyzer: 语义特征分析
    - IntentReasoner: 意图推理
    - ResponseGenerator: 响应生成
    
    【功能】
    1. 接收用户消息
    2. 分析语义特征
    3. 确定用户意图
    4. 生成结构化响应
    
    【使用方式】
    ```python
    llm = StructuredReasoningLLM()
    result = llm.invoke("我的订单到哪了？")
    print(result.intent)        # 意图分类
    print(result.reasoning)     # 推理过程
    print(result.task_type)     # 任务类型
    ```
    """
    
    def __init__(self):
        """初始化所有组件"""
        self.analyzer = SemanticAnalyzer()
        self.reasoner = IntentReasoner()
        self.generator = ResponseGenerator()
    
    def invoke(self, user_message: str) -> "StructuredResult":
        """
        处理用户消息，执行完整的结构化推理
        
        Args:
            user_message: 用户输入消息
        
        Returns:
            StructuredResult: 包含意图、推理过程、任务类型和参数的结构化结果
        
        【推理流程】
        1. SemanticAnalyzer.analyze() → 提取语义特征
        2. IntentReasoner.classify() → 确定意图分类
        3. ResponseGenerator.generate() → 生成响应
        """
        # 步骤 1: 语义分析 - 提取关键特征
        features = self.analyzer.analyze(user_message)
        
        # 步骤 2: 意图分类 - 基于特征确定意图
        # reason() 返回 tuple: (Intent, confidence, reasoning)
        intent, confidence, reasoning = self.reasoner.reason(user_message)
        
        # 步骤 3: 从意图映射到任务类型
        task_type = self._intent_to_task_type(intent)
        
        # 步骤 4: 从语义特征提取参数
        parameters = self._extract_parameters(features, user_message)
        
        # 步骤 5: 生成响应
        response = self.generator.generate(
            intent=intent,
            features=features,
            reasoning=reasoning,
            order_id=parameters.get("order_id"),
        )
        
        result = StructuredResult(
            intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            task_type=task_type,
            parameters=parameters,
            response=response,
        )
        
        # 保存到实例以便后续访问
        self._last_result = result
        return result
    
    def bind(self, **kwargs):
        """绑定工具到模型（Mock 实现为空操作）。"""
        return self
    
    def _last_result(self) -> StructuredResult | None:
        """获取上一次推理结果"""
        return getattr(self, '_last_result', None)
    
    def analyze_intent(self, user_message: str) -> StructuredResult:
        """
        分析用户意图（与 invoke 相同，但方法名不同）
        
        Args:
            user_message: 用户输入消息
        
        Returns:
            StructuredResult: 包含意图、推理过程等结构化结果
        """
        return self.invoke(user_message)
    
    def _intent_to_task_type(self, intent: Intent) -> str:
        """将意图映射为任务类型"""
        mapping = {
            Intent.ORDER_STATUS: "order_status",
            Intent.REFUND: "refund",
            Intent.COMPLAINT: "complaint",
            Intent.CANCEL: "cancel",
            Intent.GREETING: "greeting",
            Intent.THANKS: "thanks",
            Intent.OFF_TOPIC: "off_topic",
        }
        return mapping.get(intent, "other")
    
    def _extract_parameters(self, features: SemanticFeatures, text: str) -> dict:
        """从语义特征中提取参数"""
        params = {}
        
        # 提取订单引用（有订单号存在）
        if features.has_order_reference:
            # 从文本中提取订单号
            import re
            order_match = re.search(r'[A-Z]\d{3}', text.upper())
            if order_match:
                params["order_id"] = order_match.group(0)
        
        # 提取动作类型
        if features.action_type:
            params["action_type"] = features.action_type
        
        return params


@dataclass
class StructuredResult:
    """
    结构化推理结果 - 包含 LLM 推理的完整输出
    
    【字段说明】
    - intent: 意图分类结果
    - confidence: 置信度 (0-1)
    - reasoning: 推理过程说明
    - task_type: 任务类型（order_status/refund/policy/other）
    - parameters: 提取的参数（订单号、原因等）
    - response: 生成的响应文本
    """
    intent: Intent
    confidence: float
    reasoning: str
    task_type: str
    parameters: dict
    response: str


# =============================================================================
# 单例模式 - 全局 Mock 模型实例
# =============================================================================

_default_model: MockChatModel | None = None


def get_mock_model() -> MockChatModel:
    """获取全局默认的 Mock 模型实例。
    
    使用单例模式，避免重复创建实例。
    
    Returns:
        MockChatModel 实例
    """
    global _default_model
    if _default_model is None:
        _default_model = MockChatModel()
    return _default_model
