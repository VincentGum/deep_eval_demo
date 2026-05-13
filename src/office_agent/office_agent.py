"""Office Agent - Main entry point and workflow orchestration.

This module provides the main interface for the Office Agent system,
orchestrating the PEV workflow with Planner, Executor, Verifier, and Human-in-the-Loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from .base import Task, TaskPlan, TaskStatus
from .planner import PlannerAgent, MockReasoningModel
from .executor import TaskExecutor, ExecutionContext, TaskProgress
from .verify import VerifyAgent, VerifyDecision, VerificationReport
from .human_loop import HumanInTheLoop, HumanInputRequest, HumanInputResult


class WorkflowState(str, Enum):
    """Workflow execution state."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    WAITING_HUMAN = "waiting_human"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowProgress:
    """Progress information for the workflow."""
    state: WorkflowState
    message: str
    plan_progress: dict[str, Any] = field(default_factory=dict)
    current_task: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    success: bool
    plan: TaskPlan
    final_state: WorkflowState
    output: dict[str, Any] = field(default_factory=dict)
    human_inputs: list[HumanInputResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    completed_at: datetime = field(default_factory=datetime.now)


class OfficeAgent:
    """Main Office Agent orchestrator.

    The Office Agent coordinates:
    1. Planning: Using Planner Agent to create task plans
    2. Execution: Using Task Executor to run tasks in parallel
    3. Verification: Using Verify Agent to check completion
    4. Human-in-the-Loop: Requesting human input when needed
    """

    def __init__(
        self,
        planner: PlannerAgent | None = None,
        executor: TaskExecutor | None = None,
        verifier: VerifyAgent | None = None,
        human_loop: HumanInTheLoop | None = None,
        progress_callback: Callable[[WorkflowProgress], None] | None = None,
    ):
        """
        Args:
            planner: Planner agent (creates task plans)
            executor: Task executor (runs tasks)
            verifier: Verify agent (checks completion)
            human_loop: Human-in-the-loop handler
            progress_callback: Callback for progress updates
        """
        self.planner = planner or PlannerAgent()
        self.executor = executor or TaskExecutor()
        self.verifier = verifier or VerifyAgent()
        self.human_loop = human_loop or HumanInTheLoop(default_timeout=60)
        self.progress_callback = progress_callback

        self._current_plan: TaskPlan | None = None
        self._current_context: ExecutionContext | None = None
        self._human_inputs: list[HumanInputResult] = []

    def execute(
        self,
        user_request: str,
        context: dict[str, Any] | None = None
    ) -> WorkflowResult:
        """Execute a user request through the PEV workflow.

        Args:
            user_request: The user's request
            context: Optional context information

        Returns:
            WorkflowResult with the execution outcome
        """
        start_time = datetime.now()
        self._human_inputs = []

        # Step 1: PLANNING
        self._report_progress(WorkflowState.PLANNING, "Creating task plan...")
        plan, reasoning_steps = self.planner.create_plan(user_request, context)
        self._current_plan = plan

        plan_explanation = self.planner.explain_plan(plan)
        self._report_progress(
            WorkflowState.PLANNING,
            f"Task plan created with {len(plan.tasks)} tasks",
            {"explanation": plan_explanation}
        )

        # Initialize execution context
        self._current_context = ExecutionContext(plan=plan)

        # Step 2: EXECUTING (with verification)
        max_iterations = len(plan.tasks) * 2 + 5  # Safety limit
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Check if all tasks are done
            if plan.all_completed():
                break

            # Execute pending tasks
            self._report_progress(
                WorkflowState.EXECUTING,
                f"Executing tasks (iteration {iteration})..."
            )

            # Execute tasks in parallel
            self._current_context = self.executor.execute_plan(
                plan,
                self._current_context
            )

            # Step 3: VERIFYING
            self._report_progress(WorkflowState.VERIFYING, "Verifying progress...")

            verification = self.verifier.verify_plan_completion(plan)

            if verification.decision == VerifyDecision.COMPLETED:
                self._report_progress(
                    WorkflowState.COMPLETED,
                    "All tasks completed successfully",
                    {"confidence": verification.confidence}
                )
                break

            elif verification.decision == VerifyDecision.NEEDS_HUMAN_INPUT:
                # Request human input
                self._report_progress(
                    WorkflowState.WAITING_HUMAN,
                    f"Waiting for human input: {verification.reason}"
                )

                # Find tasks that need input
                waiting_tasks = [
                    task for task in plan.tasks
                    if task.status == TaskStatus.WAITING_HUMAN_INPUT
                ]

                if waiting_tasks:
                    task = waiting_tasks[0]
                    human_result = self._request_human_input(
                        task,
                        verification.missing_info or []
                    )
                    self._human_inputs.append(human_result)

                    if human_result.status.value == "received":
                        # Resume task with input
                        task.status = TaskStatus.PENDING
                        task.input_data["human_input"] = human_result.response
                    else:
                        # Timeout or cancelled - fail or skip
                        task.status = TaskStatus.FAILED
                        task.result = None

            elif verification.decision == VerifyDecision.FAILED:
                self._report_progress(
                    WorkflowState.FAILED,
                    f"Workflow failed: {verification.reason}",
                    {"errors": verification.missing_info}
                )
                break

            else:
                # CONTINUE - some tasks still pending
                pending = sum(1 for t in plan.tasks if t.status == TaskStatus.PENDING)
                if pending == 0:
                    # No pending tasks but not complete - might be waiting
                    continue

        # Final verification
        final_verification = self.verifier.verify_plan_completion(plan)

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        return WorkflowResult(
            success=final_verification.decision == VerifyDecision.COMPLETED,
            plan=plan,
            final_state=WorkflowState.COMPLETED if final_verification.decision == VerifyDecision.COMPLETED else WorkflowState.FAILED,
            output={
                "summary": self.verifier.generate_summary(plan),
                "results": {
                    task.id: {
                        "status": task.status.value,
                        "output": task.result.output if task.result else None,
                        "error": task.result.error if task.result else None,
                    }
                    for task in plan.tasks
                },
            },
            human_inputs=self._human_inputs,
            errors=[verification.reason] if final_verification.decision == VerifyDecision.FAILED else [],
            execution_time_seconds=execution_time,
        )

    def _report_progress(
        self,
        state: WorkflowState,
        message: str,
        extra: dict[str, Any] | None = None
    ) -> None:
        """Report workflow progress."""
        progress = WorkflowProgress(
            state=state,
            message=message,
            plan_progress=self._current_plan.get_execution_summary() if self._current_plan else {},
        )

        if self.progress_callback:
            self.progress_callback(progress)

    def _request_human_input(
        self,
        task: Task,
        missing_info: list[str]
    ) -> HumanInputResult:
        """Request human input for a task.

        Args:
            task: Task requiring input
            missing_info: List of missing information

        Returns:
            HumanInputResult
        """
        # Build question
        question = f"Task '{task.description}' needs more information.\n"
        if missing_info:
            question += "Missing: " + ", ".join(missing_info[:3])
        question += "\nPlease provide the required information."

        # Create input request
        request = HumanInputRequest(
            id=f"input_{task.id}",
            task_id=task.id,
            question=question,
            context={"task_description": task.description},
            timeout_seconds=task.timeout_seconds,
        )

        # Generate and display prompt
        prompt = self.human_loop.generate_prompt(request)
        print("\n" + prompt)

        # Request input
        return self.human_loop.request_input(
            task_id=task.id,
            question=question,
            context={"task_description": task.description},
            timeout_seconds=task.timeout_seconds,
        )

    def get_current_plan(self) -> TaskPlan | None:
        """Get the current task plan."""
        return self._current_plan

    def get_verification_summary(self) -> str | None:
        """Get verification summary if available."""
        if self._current_plan:
            return self.verifier.generate_summary(self._current_plan)
        return None


# ============================================================================
# Convenience Functions
# ============================================================================

def invoke_office_agent(
    user_request: str,
    context: dict[str, Any] | None = None,
    progress_callback: Callable[[WorkflowProgress], None] | None = None,
) -> WorkflowResult:
    """Invoke the Office Agent with a user request.

    This is the main entry point for using the Office Agent.

    Args:
        user_request: The user's request
        context: Optional context (user info, available tools, etc.)
        progress_callback: Optional callback for progress updates

    Returns:
        WorkflowResult with the execution outcome

    Example:
        >>> result = invoke_office_agent(
        ...     "Generate a weekly sales report",
        ...     {"user": "Alice", "department": "Sales"}
        ... )
        >>> print(result.success)
        True
        >>> print(result.output["summary"])
    """
    agent = OfficeAgent(progress_callback=progress_callback)
    return agent.execute(user_request, context)


def build_office_agent_graph(**kwargs) -> OfficeAgent:
    """Build an Office Agent instance with custom configuration.

    Args:
        **kwargs: Configuration options passed to OfficeAgent

    Returns:
        Configured OfficeAgent instance
    """
    return OfficeAgent(**kwargs)
