"""
验证 Agent - 监控任务执行并验证完成状态

【模块概述】
Verify Agent 是 PEV 架构中的 "V"（Verify）节点，负责：
1. 验证任务输出 - 对比实际结果与预期结果
2. 进度检查 - 判断整体任务是否完成
3. 缺失识别 - 识别需要补充的信息
4. 决策生成 - 决定下一步操作（继续/完成/需要人工）

【验证决策】
- CONTINUE: 继续执行，等待更多任务完成
- COMPLETED: 所有任务已完成
- NEEDS_HUMAN_INPUT: 需要人工输入
- FAILED: 执行失败
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .base import Task, TaskPlan, TaskResult, TaskStatus


# ============================================================================
# 验证决策枚举
# ============================================================================

class VerifyDecision(str, Enum):
    """
    验证决策枚举
    
    【决策说明】
    - CONTINUE: 继续执行（还有待处理任务）
    - COMPLETED: 全部完成（任务计划成功）
    - NEEDS_HUMAN_INPUT: 需要人工输入
    - FAILED: 执行失败
    """
    CONTINUE = "continue"                       # 继续
    COMPLETED = "completed"                     # 完成
    NEEDS_HUMAN_INPUT = "needs_human_input"     # 需要人工
    FAILED = "failed"                          # 失败


# ============================================================================
# 验证报告
# ============================================================================

@dataclass
class VerificationReport:
    """
    验证报告 - 记录验证结果
    
    【字段说明】
    - decision: 验证决策
    - task_id: 关联的任务 ID（如果有）
    - reason: 决策原因
    - expected_vs_actual: 预期 vs 实际对比
    - suggested_action: 建议的操作
    - missing_info: 缺失的信息列表
    - confidence: 置信度 (0.0-1.0)
    - timestamp: 验证时间
    """
    decision: VerifyDecision
    task_id: str | None
    reason: str
    expected_vs_actual: dict[str, Any] | None = None
    suggested_action: str | None = None
    missing_info: list[str] = field(default_factory=list)
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# 验证 Agent
# ============================================================================

class VerifyAgent:
    """
    验证 Agent - 检查任务完成状态
    
    【验证流程】
    1. 接收任务结果
    2. 对比预期 vs 实际
    3. 计算匹配度
    4. 生成验证报告
    """
    
    def __init__(self):
        self._verification_history: list[VerificationReport] = []
    
    def verify_task(
        self,
        task: Task,
        actual_result: TaskResult,
        context: dict[str, Any] | None = None
    ) -> VerificationReport:
        """
        验证单个任务的结果
        
        【验证逻辑】
        1. 检查是否失败
        2. 检查是否等待人工输入
        3. 对比预期输出与实际输出
        4. 计算匹配率
        
        Args:
            task: 原始任务（含预期输出）
            actual_result: 实际执行结果
            context: 可选的验证上下文
        
        Returns:
            验证报告
        """
        report = VerificationReport(
            decision=VerifyDecision.CONTINUE,
            task_id=task.id,
            reason="",
        )
        
        # 1. 检查任务是否失败
        if actual_result.status == TaskStatus.FAILED:
            report.decision = VerifyDecision.FAILED
            report.reason = f"Task {task.id} failed: {actual_result.error}"
            report.confidence = 1.0
            self._verification_history.append(report)
            return report
        
        # 2. 检查是否等待人工输入
        if actual_result.status == TaskStatus.WAITING_HUMAN_INPUT:
            report.decision = VerifyDecision.NEEDS_HUMAN_INPUT
            report.reason = f"Task {task.id} requires human input"
            report.missing_info = context.get("missing_info", []) if context else []
            report.suggested_action = "Request information from user"
            report.confidence = 1.0
            self._verification_history.append(report)
            return report
        
        # 3. 验证输出
        expected = task.expected_output.lower() if task.expected_output else ""
        actual = str(actual_result.output).lower() if actual_result.output else ""
        
        if expected and actual:
            # 计算关键词重叠度
            expected_terms = set(expected.replace(",", " ").split())
            actual_terms = set(actual.replace(",", " ").split())
            overlap = expected_terms & actual_terms
            
            # 计算匹配率
            match_ratio = len(overlap) / len(expected_terms) if expected_terms else 1.0
            
            if match_ratio >= 0.5:
                report.decision = VerifyDecision.CONTINUE
                report.reason = f"Task {task.id} output matches expected ({match_ratio:.0%})"
                report.confidence = match_ratio
            else:
                report.decision = VerifyDecision.NEEDS_HUMAN_INPUT
                report.reason = f"Task {task.id} output incomplete"
                report.expected_vs_actual = {
                    "expected": task.expected_output,
                    "actual": actual_result.output,
                    "match_ratio": match_ratio,
                }
                report.missing_info = self._identify_missing_info(task, actual_result)
                report.suggested_action = "Request clarification from user"
                report.confidence = match_ratio
        else:
            # 没有预期输出，假设成功
            report.decision = VerifyDecision.CONTINUE
            report.reason = f"Task {task.id} completed (no explicit verification)"
            report.confidence = 1.0
        
        self._verification_history.append(report)
        return report
    
    def verify_plan_completion(
        self,
        plan: TaskPlan,
        context: dict[str, Any] | None = None
    ) -> VerificationReport:
        """
        验证整个任务计划是否完成
        
        Args:
            plan: 任务计划
            context: 可选的验证上下文
        
        Returns:
            验证报告
        """
        report = VerificationReport(
            decision=VerifyDecision.CONTINUE,
            task_id=None,
            reason="",
        )
        
        # 1. 检查是否有失败的任务
        failed_tasks = [t for t in plan.tasks if t.status == TaskStatus.FAILED]
        if failed_tasks:
            report.decision = VerifyDecision.FAILED
            report.reason = f"{len(failed_tasks)} task(s) failed"
            report.missing_info = [t.id for t in failed_tasks]
            return report
        
        # 2. 检查是否有任务等待人工输入
        waiting_tasks = [t for t in plan.tasks if t.status == TaskStatus.WAITING_HUMAN_INPUT]
        if waiting_tasks:
            report.decision = VerifyDecision.NEEDS_HUMAN_INPUT
            report.task_id = waiting_tasks[0].id
            report.reason = f"{len(waiting_tasks)} task(s) waiting for human input"
            report.suggested_action = "Request information from user"
            return report
        
        # 3. 检查是否还有待处理的任务
        pending_tasks = [t for t in plan.tasks if t.status == TaskStatus.PENDING]
        running_tasks = [t for t in plan.tasks if t.status == TaskStatus.RUNNING]
        
        if pending_tasks or running_tasks:
            report.decision = VerifyDecision.CONTINUE
            report.reason = f"{len(pending_tasks)} pending, {len(running_tasks)} running"
            return report
        
        # 4. 所有任务都已完成
        report.decision = VerifyDecision.COMPLETED
        report.reason = f"All {len(plan.tasks)} tasks completed"
        report.confidence = 1.0
        
        return report
    
    def _identify_missing_info(
        self,
        task: Task,
        result: TaskResult
    ) -> list[str]:
        """
        识别缺失的信息
        
        【实现逻辑】
        1. 对比预期输出中的关键项
        2. 检查实际输出中是否包含
        3. 返回缺失项列表
        
        Args:
            task: 原始任务
            result: 执行结果
        
        Returns:
            缺失的信息列表
        """
        missing = []
        
        expected = task.expected_output or ""
        actual = str(result.output) if result.output else ""
        
        # 简单的关键词检查
        key_indicators = ["order", "id", "status", "date", "number"]
        for indicator in key_indicators:
            if indicator.lower() in expected.lower() and indicator.lower() not in actual.lower():
                missing.append(f"Missing: {indicator}")
        
        return missing[:3]  # 最多返回3项
    
    def generate_summary(self, plan: TaskPlan) -> str:
        """
        生成验证摘要
        
        Args:
            plan: 任务计划
        
        Returns:
            格式化的摘要字符串
        """
        summary = [
            "=" * 60,
            "VERIFICATION SUMMARY",
            "=" * 60,
            f"Plan: {plan.id}",
            f"Total Tasks: {len(plan.tasks)}",
            "",
        ]
        
        # 统计各状态任务数
        status_counts = {}
        for task in plan.tasks:
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        summary.append("Status Breakdown:")
        for status, count in sorted(status_counts.items()):
            summary.append(f"  - {status}: {count}")
        
        summary.append("")
        
        # 详细任务列表
        summary.append("Task Details:")
        for task in plan.tasks:
            status_icon = "✓" if task.status == TaskStatus.COMPLETED else "✗"
            summary.append(f"  {status_icon} {task.id}: {task.description}")
            summary.append(f"     Status: {task.status.value}")
            if task.result and task.result.error:
                summary.append(f"     Error: {task.result.error}")
        
        summary.append("=" * 60)
        
        return "\n".join(summary)


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    "VerifyDecision",
    "VerificationReport",
    "VerifyAgent",
]
