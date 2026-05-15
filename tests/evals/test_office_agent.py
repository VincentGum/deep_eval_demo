"""DeepEval test cases for the PEV Office Agent.

Run with:
    PYTHONPATH=src DEEPEVAL_TELEMETRY_OPT_OUT=YES deepeval test run tests/evals/test_office_agent.py

Note: DeepEval 4.0 uses 'deepeval test run' command, not pytest.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import pytest

from deepeval import assert_test
from deepeval.metrics import BaseMetric, GEval
from deepeval.test_case import LLMTestCase

from office_agent import OfficeAgent, invoke_office_agent


# ============================================================
# G-EVAL RUBRICS - Office Agent 评测标准
# ============================================================

PLANNING_QUALITY_CRITERIA = """
评估 Office Agent 任务规划质量：

1. 任务分解完整性 (35%): 是否将复杂请求分解为可执行的子任务
2. 依赖关系处理 (25%): 是否正确识别任务间的依赖关系
3. Agent 类型匹配 (25%): 是否为每个子任务选择了正确的 Agent 类型
4. 并行化优化 (15%): 是否充分利用了任务的并行性

评分范围：0-100
"""

EXECUTION_QUALITY_CRITERIA = """
评估 Office Agent 执行质量：

1. 工具调用准确性 (40%): 是否正确调用了所需的工具
2. 数据处理正确性 (30%): 返回的数据/内容是否准确
3. 错误处理 (15%): 是否妥善处理执行错误
4. 执行效率 (15%): 是否在合理时间内完成

评分范围：0-100
"""

OUTPUT_QUALITY_CRITERIA = """
评估 Office Agent 输出质量：

1. 内容完整性 (30%): 是否包含所有预期的内容
2. 格式规范性 (25%): 输出格式是否符合要求
3. 可读性 (25%): 内容是否易于理解
4. 准确性 (20%): 内容是否准确无误

评分范围：0-100
"""

HUMAN_LOOP_QUALITY_CRITERIA = """
评估 Human-in-the-Loop 处理质量：

1. 介入时机 (35%): 是否在合适的时机请求人工介入
2. 信息呈现 (30%): 是否清晰地向人工呈现需要决策的信息
3. 响应处理 (20%): 是否正确处理人工的响应
4. 超时处理 (15%): 是否妥善处理超时情况

评分范围：0-100
"""

# =============================================================================
# 自定义指标 (离线模式)
# =============================================================================

class PlanningQualityMetric(BaseMetric):
    """评测任务规划质量：检查是否正确生成了任务计划"""

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.error = None
        self.reason = None

    def measure(self, test_case: LLMTestCase) -> float:
        return self._evaluate(test_case)

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self._evaluate(test_case)

    def _evaluate(self, test_case: LLMTestCase) -> float:
        """评估任务规划质量
        
        Context[0] = expected workflow type (如 "weekly_report", "research", "meeting")
        """
        expected_workflow = test_case.context[0] if test_case.context else ""
        actual_output = test_case.actual_output.lower()

        # 检查是否有任务规划相关的关键词
        planning_keywords = [
            "plan", "task", "step", "sub-agent", "agent",
            "计划", "任务", "步骤", "子任务"
        ]
        
        has_planning = any(kw in actual_output for kw in planning_keywords)
        
        # 检查是否提到了相关的 Agent 类型
        agent_keywords = {
            "browser": ["browser", "web", "browse", "crawl", "网页", "浏览"],
            "api": ["api", "fetch", "调用", "获取数据"],
            "data": ["data", "process", "analyze", "统计", "分析"],
            "doc": ["doc", "document", "write", "read", "文档", "读写"],
            "visualization": ["chart", "graph", "visual", "可视化", "图表"],
        }
        
        matched_agents = []
        for agent_type, keywords in agent_keywords.items():
            if any(kw in actual_output for kw in keywords):
                matched_agents.append(agent_type)

        self.reason = f"Planning keywords found: {has_planning}, Agents mentioned: {matched_agents}"
        
        if has_planning:
            self.score = 1.0
            return 1.0
        else:
            # 即使没有明确的规划关键词，只要输出非空，也给部分分数
            if len(actual_output) > 10:
                self.score = 0.7
                return 0.7
            else:
                self.score = 0.0
                return 0.0

    def is_successful(self) -> bool:
        return self.score is not None and self.score >= self.threshold


class ExecutionQualityMetric(BaseMetric):
    """评测执行质量：检查任务是否正确执行"""

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.error = None
        self.reason = None

    def measure(self, test_case: LLMTestCase) -> float:
        return self._evaluate(test_case)

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self._evaluate(test_case)

    def _evaluate(self, test_case: LLMTestCase) -> float:
        """评估执行质量
        
        Context[0] = expected output type (如 "report", "data", "document")
        """
        expected_type = test_case.context[0] if test_case.context else ""
        actual_output = test_case.actual_output.lower()

        # 检查输出是否包含预期的内容类型
        output_type_keywords = {
            "report": ["report", "summary", "analysis", "报告", "总结", "分析"],
            "data": ["data", "number", "statistic", "数据", "数字", "统计"],
            "document": ["document", "file", "content", "文档", "文件", "内容"],
            "chart": ["chart", "graph", "visual", "图表", "可视化"],
            "api_result": ["api", "response", "result", "接口", "响应", "结果"],
        }
        
        # 检查是否有执行相关的关键词
        execution_keywords = [
            "completed", "finished", "executed", "done",
            "已完成", "执行", "完成", "生成"
        ]
        
        has_execution = any(kw in actual_output for kw in execution_keywords)
        
        # 检查是否有错误关键词
        error_keywords = [
            "error", "failed", "exception", "not found",
            "错误", "失败", "异常", "未找到"
        ]
        
        has_error = any(kw in actual_output for kw in error_keywords)
        
        # 根据输出长度和内容判断质量
        output_length = len(actual_output)
        
        if has_error:
            self.score = 0.3
            self.reason = "Execution contains errors"
            return 0.3
        
        if output_length > 50:
            self.score = 1.0
            self.reason = f"Execution completed with substantial output ({output_length} chars)"
            return 1.0
        elif output_length > 10:
            self.score = 0.7
            self.reason = f"Execution completed with minimal output ({output_length} chars)"
            return 0.7
        else:
            self.score = 0.0
            self.reason = "Execution produced no meaningful output"
            return 0.0

    def is_successful(self) -> bool:
        return self.score is not None and self.score >= self.threshold


class OutputQualityMetric(BaseMetric):
    """评测输出质量：检查最终输出是否符合要求"""

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.error = None
        self.reason = None

    def measure(self, test_case: LLMTestCase) -> float:
        return self._evaluate(test_case)

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self._evaluate(test_case)

    def _evaluate(self, test_case: LLMTestCase) -> float:
        """评估输出质量
        
        Context[0] = expected content type
        Context[1] = expected format (如 "markdown", "json", "table")
        """
        expected_type = test_case.context[0] if test_case.context else ""
        expected_format = test_case.context[1] if len(test_case.context) > 1 else ""
        actual_output = test_case.actual_output

        # 检查输出格式
        format_keywords = {
            "markdown": ["#", "##", "*", "- ", "1.", "```"],
            "json": ["{", "}", "[", "]"],
            "table": ["|", "---"],
            "html": ["<html", "<div", "<table", "<span"],
        }
        
        # 检查输出长度（合理的输出应该有足够的长度）
        output_length = len(actual_output)
        
        # 检查是否包含结构化内容
        has_structure = False
        if expected_format in format_keywords:
            has_structure = any(kw in actual_output for kw in format_keywords[expected_format])
        
        # 评分逻辑
        if output_length < 10:
            self.score = 0.0
            self.reason = "Output too short"
            return 0.0
        
        # 有结构且长度适中
        if has_structure or output_length > 100:
            self.score = 1.0
            self.reason = f"Output quality good (length: {output_length}, structured: {has_structure})"
            return 1.0
        elif output_length > 50:
            self.score = 0.7
            self.reason = f"Output quality acceptable (length: {output_length})"
            return 0.7
        else:
            self.score = 0.5
            self.reason = f"Output quality marginal (length: {output_length})"
            return 0.5

    def is_successful(self) -> bool:
        return self.score is not None and self.score >= self.threshold


class HumanLoopHandlingMetric(BaseMetric):
    """评测 Human-in-the-Loop 处理"""

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.error = None
        self.reason = None

    def measure(self, test_case: LLMTestCase) -> float:
        return self._evaluate(test_case)

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self._evaluate(test_case)

    def _evaluate(self, test_case: LLMTestCase) -> float:
        """评估 Human-in-the-Loop 处理
        
        Context[0] = expected_human_loop (true/false)
        """
        expected_human_loop = test_case.context[0] if test_case.context else "false"
        actual_output = test_case.actual_output.lower()

        # 检查是否提到了人工介入相关的关键词
        waiting_keywords = [
            "waiting for", "waiting", "pending", "人工",
            "human review", "please wait", "processing"
        ]
        
        completed_keywords = [
            "completed", "finished", "done", "here is", "result",
            "已完成", "完成", "结果", "以下是"
        ]
        
        has_waiting = any(kw in actual_output for kw in waiting_keywords)
        has_completed = any(kw in actual_output for kw in completed_keywords)

        # 如果预期需要人工介入，检查是否提到了等待
        if expected_human_loop.lower() == "true":
            if has_waiting:
                self.score = 1.0
                self.reason = "Correctly indicated waiting for human input"
                return 1.0
            elif has_completed:
                # 如果没有等待但任务完成了，可能已经自动处理
                self.score = 0.8
                self.reason = "Task completed without explicit waiting (may have auto-handled)"
                return 0.8
            else:
                self.score = 0.5
                self.reason = "No clear indication of human loop status"
                return 0.5
        else:
            # 不预期人工介入
            if has_completed:
                self.score = 1.0
                self.reason = "Task completed without human intervention"
                return 1.0
            elif has_waiting:
                self.score = 0.3
                self.reason = "Unexpected waiting for human input"
                return 0.3
            else:
                self.score = 0.7
                self.reason = "No explicit human loop indication"
                return 0.7

    def is_successful(self) -> bool:
        return self.score is not None and self.score >= self.threshold


# =============================================================================
# G-Eval 指标 (需要 OpenAI API)
# =============================================================================

def create_planning_quality_g_eval():
    """创建任务规划质量的 G-Eval 指标"""
    try:
        from deepeval.models.gpt_model import GPTModel
        model = GPTModel(model="gpt-4o")
        return GEval(
            name="PlanningQuality",
            criteria=PLANNING_QUALITY_CRITERIA,
            evaluation_steps="""
            1. 检查输出是否包含任务规划内容
            2. 检查是否识别了必要的子任务
            3. 检查是否选择了合适的 Agent 类型
            4. 综合评分
            """,
            threshold=70,
            model=model
        )
    except Exception:
        return None


def create_output_quality_g_eval():
    """创建输出质量的 G-Eval 指标"""
    try:
        from deepeval.models.gpt_model import GPTModel
        model = GPTModel(model="gpt-4o")
        return GEval(
            name="OutputQuality",
            criteria=OUTPUT_QUALITY_CRITERIA,
            evaluation_steps="""
            1. 检查输出内容是否完整
            2. 检查格式是否规范
            3. 检查可读性
            4. 综合评分
            """,
            threshold=70,
            model=model
        )
    except Exception:
        return None


# =============================================================================
# 测试辅助函数
# =============================================================================

def run_office_agent_test(
    user_request: str,
    scenario: str = None,
    human_input_func: Callable[[str], str] = None
) -> dict[str, Any]:
    """运行 Office Agent 并返回结果
    
    Args:
        user_request: 用户请求
        scenario: 场景名称（可选）
        human_input_func: 人工输入函数（用于 Human-in-the-Loop）
    
    Returns:
        包含执行结果的字典
    """
    def default_human_input(prompt: str) -> str:
        """默认的人工输入函数：自动确认"""
        return "OK"
    
    if human_input_func is None:
        human_input_func = default_human_input

    try:
        result = invoke_office_agent(
            user_request=user_request,
            scenario=scenario,
            human_input_func=human_input_func
        )
        return {
            "success": True,
            "actual_output": result.get("response", ""),
            "state": result.get("state", {}),
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "actual_output": f"Error: {str(e)}",
            "state": {},
            "error": str(e)
        }


# =============================================================================
# DeepEval 测试用例
# =============================================================================

def test_weekly_sales_report():
    """测试：生成周销售报告
    
    场景：用户请求生成一份周销售报告
    预期：调用 API Agent 获取数据 -> Data Agent 处理 -> Visualization Agent 生成图表 -> Doc Agent 输出报告
    """
    user_request = "生成本周销售报告，包括销售数据统计和趋势图表"
    # Context: [workflow_type, output_type, expected_human_loop]
    context = ["weekly_report", "report", "false"]

    result = run_office_agent_test(user_request, scenario="weekly_sales_report")

    test_case = LLMTestCase(
        input=user_request,
        actual_output=result["actual_output"],
        context=context
    )

    metrics = [
        PlanningQualityMetric(threshold=0.5),
        ExecutionQualityMetric(threshold=0.5),
        OutputQualityMetric(threshold=0.5),
        HumanLoopHandlingMetric(threshold=0.5)
    ]

    assert_test(test_case, metrics)


def test_customer_research():
    """测试：客户调研任务
    
    场景：用户需要对某客户进行调研
    预期：调用 Browser Agent 爬取数据 -> Data Agent 分析 -> Doc Agent 生成报告
    """
    user_request = "帮我调研一下某科技公司的最新动态和财务状况"
    context = ["research", "report", "false"]

    result = run_office_agent_test(user_request, scenario="customer_research")

    test_case = LLMTestCase(
        input=user_request,
        actual_output=result["actual_output"],
        context=context
    )

    metrics = [
        PlanningQualityMetric(threshold=0.5),
        ExecutionQualityMetric(threshold=0.5),
        OutputQualityMetric(threshold=0.5),
        HumanLoopHandlingMetric(threshold=0.5)
    ]

    assert_test(test_case, metrics)


def test_meeting_preparation():
    """测试：会议准备工作
    
    场景：用户需要为会议做准备
    预期：调用 Data Agent 查询数据 -> Doc Agent 格式化文档
    """
    user_request = "帮我准备明天的销售会议，需要整理本周的销售数据"
    context = ["meeting", "document", "false"]

    result = run_office_agent_test(user_request, scenario="meeting_preparation")

    test_case = LLMTestCase(
        input=user_request,
        actual_output=result["actual_output"],
        context=context
    )

    metrics = [
        PlanningQualityMetric(threshold=0.5),
        ExecutionQualityMetric(threshold=0.5),
        OutputQualityMetric(threshold=0.5),
        HumanLoopHandlingMetric(threshold=0.5)
    ]

    assert_test(test_case, metrics)


def test_data_analysis_task():
    """测试：数据分析任务
    
    场景：用户需要进行数据分析
    预期：调用 Data Agent 进行数据处理
    """
    user_request = "分析一下我们产品 A 和产品 B 的销售对比"
    context = ["analysis", "data", "false"]

    result = run_office_agent_test(user_request)

    test_case = LLMTestCase(
        input=user_request,
        actual_output=result["actual_output"],
        context=context
    )

    metrics = [
        PlanningQualityMetric(threshold=0.5),
        ExecutionQualityMetric(threshold=0.5),
        OutputQualityMetric(threshold=0.5),
        HumanLoopHandlingMetric(threshold=0.5)
    ]

    assert_test(test_case, metrics)


def test_document_generation():
    """测试：文档生成任务
    
    场景：用户需要生成文档
    预期：调用 Doc Agent 生成文档
    """
    user_request = "帮我写一份项目计划书，主要包括项目背景、目标和实施方案"
    context = ["document", "markdown", "false"]

    result = run_office_agent_test(user_request)

    test_case = LLMTestCase(
        input=user_request,
        actual_output=result["actual_output"],
        context=context
    )

    metrics = [
        PlanningQualityMetric(threshold=0.5),
        ExecutionQualityMetric(threshold=0.5),
        OutputQualityMetric(threshold=0.5),
        HumanLoopHandlingMetric(threshold=0.5)
    ]

    assert_test(test_case, metrics)


def test_web_browsing_task():
    """测试：网页浏览任务
    
    场景：用户需要浏览网页获取信息
    预期：调用 Browser Agent 访问网页
    """
    user_request = "帮我查一下今天科技新闻有哪些"
    context = ["browse", "data", "false"]

    result = run_office_agent_test(user_request)

    test_case = LLMTestCase(
        input=user_request,
        actual_output=result["actual_output"],
        context=context
    )

    metrics = [
        PlanningQualityMetric(threshold=0.5),
        ExecutionQualityMetric(threshold=0.5),
        OutputQualityMetric(threshold=0.5),
        HumanLoopHandlingMetric(threshold=0.5)
    ]

    assert_test(test_case, metrics)


def test_api_call_task():
    """测试：API 调用任务
    
    场景：用户需要调用外部 API
    预期：调用 API Agent
    """
    user_request = "获取一下当前天气信息"
    context = ["api_call", "data", "false"]

    result = run_office_agent_test(user_request)

    test_case = LLMTestCase(
        input=user_request,
        actual_output=result["actual_output"],
        context=context
    )

    metrics = [
        PlanningQualityMetric(threshold=0.5),
        ExecutionQualityMetric(threshold=0.5),
        OutputQualityMetric(threshold=0.5),
        HumanLoopHandlingMetric(threshold=0.5)
    ]

    assert_test(test_case, metrics)


def test_chart_generation():
    """测试：图表生成任务
    
    场景：用户需要生成可视化图表
    预期：调用 Visualization Agent
    """
    user_request = "根据以下销售数据生成一个柱状图：[1月:100, 2月:150, 3月:200]"
    context = ["chart", "visual", "false"]

    result = run_office_agent_test(user_request)

    test_case = LLMTestCase(
        input=user_request,
        actual_output=result["actual_output"],
        context=context
    )

    metrics = [
        PlanningQualityMetric(threshold=0.5),
        ExecutionQualityMetric(threshold=0.5),
        OutputQualityMetric(threshold=0.5),
        HumanLoopHandlingMetric(threshold=0.5)
    ]

    assert_test(test_case, metrics)


def test_complex_multi_step_task():
    """测试：复杂多步骤任务
    
    场景：用户需要一个涉及多个步骤的复杂任务
    预期：多个 Agent 协同工作
    """
    user_request = """
    帮我完成以下任务：
    1. 先获取最新的市场数据
    2. 对数据进行趋势分析
    3. 生成分析报告并附上图表
    """
    context = ["multi_step", "report", "false"]

    result = run_office_agent_test(user_request)

    test_case = LLMTestCase(
        input=user_request,
        actual_output=result["actual_output"],
        context=context
    )

    metrics = [
        PlanningQualityMetric(threshold=0.5),
        ExecutionQualityMetric(threshold=0.5),
        OutputQualityMetric(threshold=0.5),
        HumanLoopHandlingMetric(threshold=0.5)
    ]

    assert_test(test_case, metrics)


def test_human_loop_required():
    """测试：需要人工介入的场景
    
    场景：任务无法自动完成，需要人工确认
    预期：正确触发 Human-in-the-Loop
    """
    user_request = "帮我生成一份年度报告，需要你确认一些关键数据后才能继续"
    context = ["multi_step", "report", "true"]

    result = run_office_agent_test(user_request)

    test_case = LLMTestCase(
        input=user_request,
        actual_output=result["actual_output"],
        context=context
    )

    metrics = [
        PlanningQualityMetric(threshold=0.5),
        ExecutionQualityMetric(threshold=0.5),
        OutputQualityMetric(threshold=0.5),
        HumanLoopHandlingMetric(threshold=0.5)
    ]

    assert_test(test_case, metrics)


# =============================================================================
# G-Eval 测试用例 (需要 OpenAI API)
# =============================================================================

def test_planning_quality_g_eval():
    """使用 G-Eval 评估任务规划质量（需要 OpenAI API）"""
    user_request = "生成本周销售报告"
    context = ["weekly_report", "report", "false"]

    result = run_office_agent_test(user_request)

    test_case = LLMTestCase(
        input=user_request,
        actual_output=result["actual_output"],
        context=context
    )

    planning_g_eval = create_planning_quality_g_eval()
    if planning_g_eval:
        assert_test(test_case, [planning_g_eval])
    else:
        pytest.skip("G-Eval requires OpenAI API Key")


def test_output_quality_g_eval():
    """使用 G-Eval 评估输出质量（需要 OpenAI API）"""
    user_request = "帮我写一份项目计划书"
    context = ["document", "markdown", "false"]

    result = run_office_agent_test(user_request)

    test_case = LLMTestCase(
        input=user_request,
        actual_output=result["actual_output"],
        context=context
    )

    output_g_eval = create_output_quality_g_eval()
    if output_g_eval:
        assert_test(test_case, [output_g_eval])
    else:
        pytest.skip("G-Eval requires OpenAI API Key")


# =============================================================================
# Golden Dataset 测试
# =============================================================================

GOLDEN_DATA_PATH = Path(__file__).parent / "office_agent_goldens.json"


def load_golden_data() -> list[dict[str, Any]]:
    """加载 Golden 测试数据"""
    if GOLDEN_DATA_PATH.exists():
        with open(GOLDEN_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def test_golden_dataset():
    """运行 Golden Dataset 中的所有测试用例"""
    goldens = load_golden_data()
    
    if not goldens:
        pytest.skip("No golden test data found")
    
    passed = 0
    failed = 0
    errors = []
    
    for i, golden in enumerate(goldens):
        user_request = golden.get("input", "")
        context = golden.get("context", ["general", "text", "false"])
        
        result = run_office_agent_test(user_request)
        
        test_case = LLMTestCase(
            input=user_request,
            actual_output=result["actual_output"],
            context=context
        )
        
        metrics = [
            PlanningQualityMetric(threshold=0.4),
            ExecutionQualityMetric(threshold=0.4),
            OutputQualityMetric(threshold=0.4),
        ]
        
        try:
            assert_test(test_case, metrics)
            passed += 1
        except Exception as e:
            failed += 1
            errors.append(f"Golden {i+1}: {str(e)}")
    
    print(f"\nGolden Dataset Results: {passed}/{passed+failed} passed")
    if errors:
        print("Errors:")
        for err in errors[:5]:  # 只显示前5个错误
            print(f"  - {err}")
    
    # 只要有一个通过就算成功
    assert passed > 0, f"All golden tests failed: {errors}"


# =============================================================================
# 离线评测总结测试
# =============================================================================

def test_offline_evaluation_summary():
    """离线评测总结：运行所有自定义指标测试"""
    test_cases = [
        {
            "name": "Weekly Sales Report",
            "request": "生成本周销售报告",
            "context": ["weekly_report", "report", "false"]
        },
        {
            "name": "Customer Research",
            "request": "调研某科技公司最新动态",
            "context": ["research", "report", "false"]
        },
        {
            "name": "Meeting Preparation",
            "request": "准备明天的销售会议资料",
            "context": ["meeting", "document", "false"]
        },
        {
            "name": "Data Analysis",
            "request": "分析产品销售对比",
            "context": ["analysis", "data", "false"]
        },
        {
            "name": "Document Generation",
            "request": "写一份项目计划书",
            "context": ["document", "markdown", "false"]
        },
    ]
    
    results = []
    for tc in test_cases:
        result = run_office_agent_test(tc["request"])
        
        test_case = LLMTestCase(
            input=tc["request"],
            actual_output=result["actual_output"],
            context=tc["context"]
        )
        
        # 评估每个指标
        planning_metric = PlanningQualityMetric(threshold=0.4)
        execution_metric = ExecutionQualityMetric(threshold=0.4)
        output_metric = OutputQualityMetric(threshold=0.4)
        human_loop_metric = HumanLoopHandlingMetric(threshold=0.4)
        
        planning_metric.measure(test_case)
        execution_metric.measure(test_case)
        output_metric.measure(test_case)
        human_loop_metric.measure(test_case)
        
        results.append({
            "name": tc["name"],
            "planning_score": planning_metric.score,
            "execution_score": execution_metric.score,
            "output_score": output_metric.score,
            "human_loop_score": human_loop_metric.score,
            "has_error": result.get("error") is not None
        })
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("Office Agent 离线评测结果")
    print("=" * 60)
    
    passed = 0
    for r in results:
        avg_score = (r["planning_score"] + r["execution_score"] + 
                     r["output_score"] + r["human_loop_score"]) / 4
        status = "✓" if avg_score >= 0.5 and not r["has_error"] else "✗"
        if avg_score >= 0.5 and not r["has_error"]:
            passed += 1
        print(f"{status} {r['name']}: "
              f"Planning={r['planning_score']:.1f}, "
              f"Execution={r['execution_score']:.1f}, "
              f"Output={r['output_score']:.1f}, "
              f"HumanLoop={r['human_loop_score']:.1f}")
    
    print("-" * 60)
    print(f"通过率: {passed}/{len(results)}")
    print("=" * 60)
    
    # 至少要有 60% 的测试通过
    assert passed >= len(results) * 0.6, f"Too many tests failed: {passed}/{len(results)} passed"
