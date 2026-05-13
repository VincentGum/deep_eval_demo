"""Customer Agent 评测 Rubrics 和指标定义。

本文件定义了 Customer Agent 的评测指标和评分标准（Rubrics）。
使用 DeepEval 原生的 G-Eval 和自定义指标。
"""

# =============================================================================
# G-Eval Rubrics: 客服回复质量评估
# =============================================================================

RESPONSE_QUALITY_RUBRIC = """
评估客服 Agent 回复的质量，评分标准：

1. 准确性 (40%): 回复内容是否准确回答了用户的问题
   - 正确理解用户意图
   - 提供正确的信息或操作指引

2. 完整性 (25%): 回复是否提供完整的信息
   - 包含必要的细节
   - 提供后续步骤或建议

3. 专业性 (20%): 回复是否符合客服专业标准
   - 使用礼貌用语
   - 语气友好、耐心
   - 不包含不当表达

4. 合规性 (15%): 回复是否符合公司政策
   - 不泄露敏感信息
   - 不做出未授权的承诺

评分范围: 1-5 分 (1=差, 3=一般, 5=优秀)
"""

# =============================================================================
# G-Eval Rubrics: 意图识别准确性评估
# =============================================================================

INTENT_RECOGNITION_RUBRIC = """
评估 Agent 对用户意图的识别准确性，评分标准：

1. 意图识别 (40%): 是否正确识别了用户的主要意图
   - 订单状态查询
   - 退款/取消请求
   - 投诉处理
   - 问候/闲聊

2. 实体提取 (30%): 是否正确提取了关键实体
   - 订单号
   - 产品信息
   - 问题描述

3. 上下文理解 (30%): 是否理解用户的完整请求
   - 多意图情况
   - 隐含需求
   - 紧急程度

评分范围: 1-5 分
"""

# =============================================================================
# G-Eval Rubrics: 工具使用合理性评估
# =============================================================================

TOOL_USAGE_RUBRIC = """
评估 Agent 对工具使用的合理性，评分标准：

1. 必要性 (35%): 是否在必要时使用了工具
   - 无需工具的简单查询不应调用工具
   - 复杂操作应调用相应工具

2. 正确性 (35%): 是否选择了正确的工具
   - lookup_order 用于订单查询
   - create_refund_case 用于退款/投诉

3. 效率 (30%): 是否以最少的工具调用完成请求
   - 避免不必要的重复调用
   - 合理利用已有信息

评分范围: 1-5 分
"""

# =============================================================================
# G-Eval Rubrics: Human-in-the-Loop 判断准确性
# =============================================================================

HUMAN_REVIEW_RUBRIC = """
评估 Agent 对人工审核需求的判断准确性，评分标准：

1. 必要性判断 (50%): 是否正确判断需要人工审核
   - 退款/取消: 需要审核
   - 投诉: 需要审核
   - 简单查询: 不需要审核

2. 时机把握 (30%): 是否在合适的时机请求审核
   - 应该在确认用户意图后立即请求
   - 不应过早或过晚

3. 透明度 (20%): 是否清楚地向用户说明审核状态
   - 明确告知用户正在等待审核
   - 说明预计处理时间

评分范围: 1-5 分
"""

# =============================================================================
# G-Eval Rubrics: 敏感词检测准确性
# =============================================================================

SENSITIVE_CONTENT_RUBRIC = """
评估 Agent 对敏感内容的处理，评分标准：

1. 检测准确性 (40%): 是否正确识别敏感内容
   - 投诉中的负面情绪
   - 退款请求中的紧急程度
   - 隐含的法律/投诉风险

2. 处理适当性 (35%): 对敏感内容的处理是否恰当
   - 是否正确触发人工审核
   - 是否使用安抚性语言

3. 信息安全 (25%): 是否保护了敏感信息
   - 不泄露用户隐私
   - 不在日志中暴露敏感数据

评分范围: 1-5 分
"""

# =============================================================================
# DeepEval 内置指标配置
# =============================================================================

CUSTOMER_AGENT_METRICS = {
    # 内置指标 (需要 LLM API)
    "answer_relevancy": {
        "class": "AnswerRelevancyMetric",
        "threshold": 0.3,
        "params": {"strict_mode": False}
    },
    "faithfulness": {
        "class": "FaithfulnessMetric",
        "threshold": 0.5,
        "params": {"strict_mode": False}
    },

    # 自定义指标 (离线模式)
    "intent_accuracy": {
        "class": "IntentAccuracyMetric",
        "threshold": 0.5,
        "params": {}
    },
    "tool_selection": {
        "class": "ToolSelectionMetric",
        "threshold": 0.5,
        "params": {}
    },
    "human_review_decision": {
        "class": "HumanReviewDecisionMetric",
        "threshold": 0.5,
        "params": {}
    },

    # G-Eval 指标 (需要 LLM API)
    "response_quality": {
        "class": "GEval",
        "params": {
            "name": "Response Quality",
            "criteria": RESPONSE_QUALITY_RUBRIC
        }
    },
    "intent_recognition": {
        "class": "GEval",
        "params": {
            "name": "Intent Recognition",
            "criteria": INTENT_RECOGNITION_RUBRIC
        }
    },
    "tool_usage": {
        "class": "GEval",
        "params": {
            "name": "Tool Usage",
            "criteria": TOOL_USAGE_RUBRIC
        }
    },
    "human_review_judgment": {
        "class": "GEval",
        "params": {
            "name": "Human Review Judgment",
            "criteria": HUMAN_REVIEW_RUBRIC
        }
    }
}

# =============================================================================
# 测试场景分类
# =============================================================================

SCENARIO_CATEGORIES = {
    "order_inquiry": {
        "description": "订单状态查询",
        "typical_intent": "order_status",
        "requires_tools": ["lookup_order"],
        "requires_human_review": False,
        "examples": [
            "Where is my order #A100?",
            "Can you check the status of my order?",
            "我的订单什么时候发货？"
        ]
    },
    "refund_request": {
        "description": "退款申请",
        "typical_intent": "order_status",
        "requires_tools": ["lookup_order", "create_refund_case"],
        "requires_human_review": True,
        "examples": [
            "I want to request a refund",
            "The product I received is damaged",
            "Please refund my order #B200"
        ]
    },
    "cancellation": {
        "description": "订单取消",
        "typical_intent": "order_status",
        "requires_tools": ["lookup_order", "create_refund_case"],
        "requires_human_review": True,
        "examples": [
            "I need to cancel order #A100",
            "Please cancel my order immediately",
            "I no longer need this order"
        ]
    },
    "complaint": {
        "description": "投诉处理",
        "typical_intent": "order_status",
        "requires_tools": ["lookup_order", "create_refund_case"],
        "requires_human_review": True,
        "examples": [
            "I want to complain about my order",
            "The delivery was late and the package was damaged",
            "I am very unhappy with your service"
        ]
    },
    "greeting": {
        "description": "问候/闲聊",
        "typical_intent": "greeting",
        "requires_tools": [],
        "requires_human_review": False,
        "examples": [
            "Hello, I need some help",
            "Hi there",
            "你好"
        ]
    },
    "thanks": {
        "description": "感谢/结束语",
        "typical_intent": "thanks",
        "requires_tools": [],
        "requires_human_review": False,
        "examples": [
            "Thank you so much!",
            "Thanks for your help!",
            "You're welcome"
        ]
    },
    "general_question": {
        "description": "一般性问题",
        "typical_intent": "generic_question",
        "requires_tools": [],
        "requires_human_review": False,
        "examples": [
            "What are your business hours?",
            "How can I track my package?",
            "你们的客服电话是多少？"
        ]
    },
    "multi_intent": {
        "description": "多意图场景",
        "typical_intent": "order_status",  # 主要意图
        "requires_tools": ["lookup_order", "create_refund_case"],
        "requires_human_review": True,
        "examples": [
            "My order #A100 is late and I want a refund",
            "Check order #B200 and cancel it if possible",
            "I received the wrong item in order #C300, please help"
        ]
    }
}

# =============================================================================
# 评测结果评估标准
# =============================================================================

EVALUATION_THRESHOLDS = {
    # 通过阈值
    "pass": {
        "intent_accuracy": 0.8,
        "tool_selection": 0.7,
        "human_review_decision": 0.8,
        "answer_relevancy": 0.3,  # 需要 LLM
        "faithfulness": 0.5,  # 需要 LLM
    },
    # 优秀阈值
    "excellent": {
        "intent_accuracy": 0.95,
        "tool_selection": 0.9,
        "human_review_decision": 0.95,
        "answer_relevancy": 0.5,  # 需要 LLM
        "faithfulness": 0.7,  # 需要 LLM
    }
}
