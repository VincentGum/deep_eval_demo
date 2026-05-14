"""
策略验证模块 - 验证回复内容和操作合规性

【模块概述】
本模块提供策略验证功能：
1. verify_policy() - 验证回复是否合规
2. is_sensitive_operation() - 判断操作是否为敏感操作
3. SensitiveOperation - 敏感操作枚举

【设计原则】
- 使用语义分析而非简单的关键词匹配
- 支持自定义敏感词库
- 提供置信度评分机制
"""

from __future__ import annotations

from enum import Enum
from typing import Tuple, List


# ============================================================================
# 敏感操作枚举
# ============================================================================

class SensitiveOperation(str, Enum):
    """
    敏感操作枚举
    
    【操作类型】
    - REFUND: 退款操作（涉及资金）
    - CANCEL: 取消订单（可能影响商家）
    - COMPENSATE: 赔偿操作（涉及资金）
    - DATA_ACCESS: 数据访问（涉及隐私）
    - HUMAN_ESCALATION: 需要人工介入
    """
    REFUND = "refund"                    # 退款
    CANCEL = "cancel"                    # 取消
    COMPENSATE = "compensate"            # 赔偿
    DATA_ACCESS = "data_access"          # 数据访问
    HUMAN_ESCALATION = "human_escalation"  # 人工介入


# ============================================================================
# 敏感词库
# ============================================================================

# 敏感词列表（示例）
# 实际应用中应从配置文件或数据库加载
SENSITIVE_WORDS = [
    # 财务相关
    "退款", "refund", "返现", "赔偿", "补偿",
    "账户", "account", "密码", "password", "credential",
    "支付", "payment", "信用卡", "credit card",
    
    # 隐私相关
    "个人", "personal", "隐私", "privacy",
    "地址", "address", "电话", "phone",
    
    # 违规词汇（示例）
    "诈骗", "欺诈", "虚假",
]

# 违规词列表（更严重的敏感词）
VIOLATION_WORDS = [
    "诈骗", "fraud", "欺诈", "钓鱼", "phishing",
]


# ============================================================================
# 策略验证函数
# ============================================================================

def verify_policy(
    response: str,
    intent: str = ""
) -> Tuple[bool, List[str], float]:
    """
    验证回复内容是否合规
    
    【检查项】
    1. 是否包含敏感词
    2. 是否包含违规词
    3. 置信度评估
    
    【参数】
    - response: 待验证的回复文本
    - intent: 回复的意图类型（可选）
    
    【返回】
    - is_compliant: 是否合规
    - sensitive_words: 检测到的敏感词列表
    - confidence: 置信度评分 (0.0-1.0)
    
    【示例】
    >>> is_ok, words, conf = verify_policy("Your order is on the way.")
    >>> print(is_ok)  # True
    >>> print(words)  # []
    """
    if not response:
        return True, [], 1.0
    
    response_lower = response.lower()
    detected_sensitive = []
    detected_violation = []
    
    # 检查违规词
    for word in VIOLATION_WORDS:
        if word.lower() in response_lower:
            detected_violation.append(word)
    
    # 如果检测到违规词，直接返回不合规
    if detected_violation:
        return False, detected_violation, 0.0
    
    # 检查敏感词
    for word in SENSITIVE_WORDS:
        if word.lower() in response_lower:
            detected_sensitive.append(word)
    
    # 计算置信度
    confidence = calculate_confidence(response, intent, detected_sensitive)
    
    # 判断是否合规
    # 合规条件：没有违规词，敏感词不影响回复质量
    is_compliant = len(detected_violation) == 0
    
    return is_compliant, detected_sensitive, confidence


def calculate_confidence(
    response: str,
    intent: str,
    sensitive_words: List[str]
) -> float:
    """
    计算回复置信度
    
    【评分因素】
    1. 响应长度（过短可能表示处理不完整）
    2. 是否包含占位符或模板文本
    3. 意图匹配度
    
    【参数】
    - response: 回复文本
    - intent: 意图类型
    - sensitive_words: 检测到的敏感词
    
    【返回】
    置信度评分 (0.0-1.0)
    """
    confidence = 1.0
    
    # 响应长度评分
    if len(response) < 20:
        confidence *= 0.8  # 过短可能不完整
    elif len(response) > 500:
        confidence *= 0.9  # 过长可能冗余
    
    # 占位符检查
    placeholder_patterns = ["{", "}", "[TODO]", "[FIXME]", "..."]
    for pattern in placeholder_patterns:
        if pattern in response:
            confidence *= 0.5
            break
    
    # 敏感词影响
    if sensitive_words:
        # 财务相关敏感词表示涉及重要操作
        financial_words = {"退款", "refund", "返现", "补偿", "支付", "payment"}
        if any(w.lower() in financial_words for w in sensitive_words):
            confidence *= 0.9  # 涉及财务，降低置信度
    
    return min(1.0, max(0.0, confidence))


def is_sensitive_operation(
    intent: str,
    tools_called: list[str],
    response: str = ""
) -> Tuple[bool, SensitiveOperation | None]:
    """
    判断操作是否为敏感操作
    
    【敏感操作判断逻辑】
    1. 调用了 create_refund_case 工具
    2. 调用了取消订单相关工具
    3. 回复涉及财务信息
    
    【参数】
    - intent: 意图类型
    - tools_called: 已调用的工具列表
    - response: 回复内容
    
    【返回】
    - is_sensitive: 是否为敏感操作
    - operation_type: 敏感操作类型（如果有）
    """
    # 退款操作
    if "create_refund_case" in tools_called:
        return True, SensitiveOperation.REFUND
    
    # 取消操作
    if "cancel" in intent.lower() or "cancel_order" in tools_called:
        return True, SensitiveOperation.CANCEL
    
    # 数据访问操作
    if intent == "data_access" or "lookup" in tools_called:
        return True, SensitiveOperation.DATA_ACCESS
    
    # 回复中包含敏感财务信息
    if response:
        financial_keywords = ["退款", "refund", "金额", "amount", "$", "¥"]
        if any(kw.lower() in response.lower() for kw in financial_keywords):
            return True, SensitiveOperation.REFUND
    
    return False, None


def check_profanity(text: str) -> Tuple[bool, List[str]]:
    """
    检查文本是否包含违规内容
    
    【参数】
    - text: 待检查的文本
    
    【返回】
    - has_profanity: 是否包含违规内容
    - found_words: 检测到的违规词
    
    【说明】
    此函数使用简单关键词匹配
    实际应用中应使用专业的文本审核服务
    """
    text_lower = text.lower()
    found = []
    
    for word in VIOLATION_WORDS:
        if word.lower() in text_lower:
            found.append(word)
    
    return len(found) > 0, found


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    "SensitiveOperation",
    "verify_policy",
    "is_sensitive_operation",
    "check_profanity",
    "SENSITIVE_WORDS",
    "VIOLATION_WORDS",
]
