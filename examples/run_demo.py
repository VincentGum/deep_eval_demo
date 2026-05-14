#!/usr/bin/env python3
"""
Customer Agent 演示脚本 - 客服智能体 PEV 架构演示

本脚本演示如何使用 Customer Agent 处理各种客服场景。
Agent 基于 PEV (Plan-Execute-Verify) 架构：
- Plan: 分析用户意图，决定调用哪些工具
- Execute: 调用工具执行操作
- Verify: 验证响应是否符合策略（敏感词、置信度等）
- Human Review: 敏感操作需要人工审批

运行方式：
    PYTHONPATH=src python examples/run_demo.py

测试场景：
1. 订单状态查询
2. 退款请求（需要人工审批）
3. 物流查询
4. 通用问候
5. 敏感操作（取消订单，需要人工审批）
"""

from __future__ import annotations

import sys
from pathlib import Path

# 将 src 目录添加到 Python 路径，以便直接运行此脚本
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from customer_agent import invoke_customer_agent


def main():
    """
    运行客服智能体演示。
    
    依次执行多个测试用例，展示 Agent 处理不同场景的能力：
    - 订单状态查询
    - 退款请求（触发人工审批）
    - 物流信息查询
    - 通用对话
    - 敏感操作（取消订单）
    
    每个用例都会打印：
    - 用户输入
    - Agent 回复
    - 调试信息（意图、工具、政策决策等）
    """

    print("=" * 60)
    print("PEV Customer Support Agent - Demo")
    print("=" * 60)
    print()

    # 测试用例列表
    # 包含不同类型的客服场景，用于演示 Agent 的各项能力
    test_cases = [
        # 场景1：订单状态查询
        # 用户询问订单 #A100 的状态
        # 预期：Agent 调用 lookup_order 工具查询订单
        {
            "name": "Order Status Inquiry",  # 测试名称
            "message": "Hi, can you tell me the status of my order #A100?",  # 用户输入
        },
        # 场景2：退款请求（包含敏感操作）
        # 用户因商品损坏要求退款
        # 预期：触发人工审批流程
        {
            "name": "Refund Request",
            "message": "I want to request a refund for order #B200, the item was damaged.",
        },
        # 场景3：物流查询
        # 用户询问订单 #C300 的物流信息
        {
            "name": "Delivery Question",
            "message": "Where is my order #C300?",
        },
        # 场景4：通用问候
        # 用户寻求帮助但没有具体问题
        {
            "name": "General Greeting",
            "message": "Hello, I need some help.",
        },
        # 场景5：敏感操作 - 取消订单
        # 用户要求立即取消订单（敏感词"cancel"触发人工审批）
        {
            "name": "Sensitive: Cancellation",
            "message": "I need to cancel order #A100 immediately.",
        },
    ]

    # 逐个执行测试用例
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─' * 60}")
        print(f"Test {i}: {test_case['name']}")  # 打印测试名称
        print(f"User: {test_case['message']}")  # 打印用户输入
        print(f"{'─' * 60}")

        # 调用 Agent 处理用户请求
        result = invoke_customer_agent(test_case["message"])

        # 显示 Agent 回复
        print(f"\nAgent Response:")
        print(result["response"])  # 打印回复内容

        # 显示调试信息
        print(f"\n[Debug Info]")
        print(f"  Intent: {result['intent']}")  # 用户意图
        print(f"  Needs Human Review: {result['human_review']}")  # 是否需要人工审批

        # 显示意图分析结果
        if result.get("intent"):
            print(f"  Confidence: {result['confidence']:.2f}")  # 置信度
            print(f"  Tools Used: {result['tools_called']}")  # 使用的工具

        # 显示人工审批原因
        if result.get("human_review_reason"):
            print(f"  Human Review Reason: {result['human_review_reason']}")

        # 显示错误信息（如果有）
        if result.get("error"):
            print(f"  Error: {result['error']}")

        print()

    # 演示结束
    print("=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    # 当直接运行此脚本时执行 main 函数
    main()
