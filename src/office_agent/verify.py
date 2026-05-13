"""Verify Agent - Monitors task execution and validates completion.

The Verify Agent receives progress reports from sub-agents,
compares actual results against expected outcomes, and decides
whether to continue, request human input, or complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .base import Task, TaskPlan, TaskResult, TaskStatus


class VerifyDecision(str, Enum):
    """Decision made by the verify agent."""
    CONTINUE = "continue"
    COMPLETED = "completed"
    NEEDS_HUMAN_INPUT = "needs_human_input"
    FAILED = "failed"


@dataclass
class VerificationReport:
    """Report from verification."""
    decision: VerifyDecision
    task_id: str | None
    reason: str
    expected_vs_actual: dict[str, Any] | None = None
    suggested_action: str | None = None
    missing_info: list[str] = field(default_factory=list)
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


class VerifyAgent:
    """Agent responsible for verifying task completion.

    The Verify Agent:
    1. Receives progress reports from sub-agents
    2. Compares actual outputs against expected outputs
    3. Determines if the overall task is complete
    4. Identifies when human input is needed
    """

    def __init__(self):
        self._verification_history: list[VerificationReport] = []

    def verify_task(
        self,
        task: Task,
        actual_result: TaskResult,
        context: dict[str, Any] | None = None
    ) -> VerificationReport:
        """Verify a single task result.

        Args:
            task: The original task with expected output
            actual_result: The actual execution result
            context: Optional verification context

        Returns:
            VerificationReport with decision
        """
        report = VerificationReport(
            decision=VerifyDecision.CONTINUE,
            task_id=task.id,
            reason="",
        )

        # Check if task failed
        if actual_result.status == TaskStatus.FAILED:
            report.decision = VerifyDecision.FAILED
            report.reason = f"Task {task.id} failed: {actual_result.error}"
            report.confidence = 1.0
            self._verification_history.append(report)
            return report

        # Check if task is waiting for human input
        if actual_result.status == TaskStatus.WAITING_HUMAN_INPUT:
            report.decision = VerifyDecision.NEEDS_HUMAN_INPUT
            report.reason = f"Task {task.id} requires human input"
            report.missing_info = context.get("missing_info", []) if context else []
            report.suggested_action = "Request information from user"
            report.confidence = 1.0
            self._verification_history.append(report)
            return report

        # Verify output against expected
        expected = task.expected_output.lower() if task.expected_output else ""
        actual = str(actual_result.output).lower() if actual_result.output else ""

        # Check for key terms in expected output
        if expected and actual:
            expected_terms = set(expected.replace(",", " ").split())
            actual_terms = set(actual.replace(",", " ").split())
            overlap = expected_terms & actual_terms

            # Calculate match percentage
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
            # No expected output specified, assume success
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
        """Verify if the entire task plan is complete.

        Args:
            plan: The task plan
            context: Optional verification context

        Returns:
            VerificationReport with overall decision
        """
        report = VerificationReport(
            decision=VerifyDecision.CONTINUE,
            task_id=None,
            reason="",
        )

        # Check for failed tasks
        failed_tasks = [
            task for task in plan.tasks
            if task.status == TaskStatus.FAILED
        ]

        if failed_tasks:
            report.decision = VerifyDecision.FAILED
            report.reason = f"{len(failed_tasks)} task(s) failed"
            report.missing_info = [
                f"Task {t.id}: {t.result.error}"
                for t in failed_tasks
                if t.result and t.result.error
            ]
            self._verification_history.append(report)
            return report

        # Check for pending tasks
        pending_tasks = [
            task for task in plan.tasks
            if task.status == TaskStatus.PENDING
        ]

        if pending_tasks:
            report.decision = VerifyDecision.CONTINUE
            report.reason = f"{len(pending_tasks)} task(s) still pending"
            self._verification_history.append(report)
            return report

        # All tasks completed
        waiting_tasks = [
            task for task in plan.tasks
            if task.status == TaskStatus.WAITING_HUMAN_INPUT
        ]

        if waiting_tasks:
            report.decision = VerifyDecision.NEEDS_HUMAN_INPUT
            report.reason = f"{len(waiting_tasks)} task(s) waiting for human input"
            report.missing_info = [task.id for task in waiting_tasks]
            report.suggested_action = "Request information from user"
            self._verification_history.append(report)
            return report

        # All tasks successfully completed
        report.decision = VerifyDecision.COMPLETED
        report.reason = f"All {len(plan.tasks)} tasks completed successfully"

        # Calculate overall confidence
        if plan.tasks:
            results = [t.result for t in plan.tasks if t.result]
            if results:
                avg_confidence = sum(
                    1.0 if r.is_successful() else 0.0
                    for r in results
                ) / len(results)
                report.confidence = avg_confidence

        self._verification_history.append(report)
        return report

    def _identify_missing_info(
        self,
        task: Task,
        actual_result: TaskResult
    ) -> list[str]:
        """Identify what information is missing.

        Args:
            task: The original task
            actual_result: The actual result

        Returns:
            List of missing information items
        """
        missing = []

        # Check input requirements
        required_inputs = task.input_data.get("required_fields", [])
        for field_name in required_inputs:
            if not task.input_data.get(field_name):
                missing.append(f"Missing required field: {field_name}")

        # Check expected output
        if task.expected_output:
            expected_terms = set(task.expected_output.lower().replace(",", " ").split())
            actual_terms = set(str(actual_result.output).lower().replace(",", " ").split())
            missing_terms = expected_terms - actual_terms

            if missing_terms:
                missing.extend([f"Missing content: {term}" for term in list(missing_terms)[:3]])

        return missing

    def generate_summary(self, plan: TaskPlan) -> str:
        """Generate a human-readable summary of verification results.

        Args:
            plan: The task plan

        Returns:
            Summary string
        """
        summary_lines = [
            "## Verification Summary",
            "",
            f"**Plan**: {plan.id}",
            f"**User Request**: {plan.user_request}",
            "",
            "### Task Results:",
        ]

        for task in plan.tasks:
            status_icon = "✓" if task.status == TaskStatus.COMPLETED else "✗" if task.status == TaskStatus.FAILED else "○"
            summary_lines.append(f"- [{status_icon}] {task.id}: {task.description}")
            summary_lines.append(f"  Status: {task.status.value}")

            if task.result:
                if task.result.output:
                    output_preview = str(task.result.output)[:100]
                    summary_lines.append(f"  Output: {output_preview}...")
                if task.result.error:
                    summary_lines.append(f"  Error: {task.result.error}")

        # Overall decision
        overall = self.verify_plan_completion(plan)
        summary_lines.extend([
            "",
            f"**Overall Decision**: {overall.decision.value}",
            f"**Reason**: {overall.reason}",
            f"**Confidence**: {overall.confidence:.0%}",
        ])

        return "\n".join(summary_lines)

    def get_history(self) -> list[VerificationReport]:
        """Get verification history."""
        return self._verification_history.copy()
