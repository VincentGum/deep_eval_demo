"""Task Executor - Handles parallel task execution.

The Task Executor coordinates the execution of tasks by sub-agents,
manages dependencies, and tracks progress.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .base import (
    Task,
    TaskStatus,
    TaskResult,
    TaskPlan,
    BaseSubAgent,
    ExecutionResult,
)
from .sub_agents.registry import find_agent_for_task


@dataclass
class TaskProgress:
    """Progress information for a task."""
    task_id: str
    status: TaskStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress_percent: float = 0.0
    message: str = ""


@dataclass
class ExecutionContext:
    """Context for task execution."""
    plan: TaskPlan
    completed_tasks: dict[str, TaskResult] = field(default_factory=dict)
    task_progress: dict[str, TaskProgress] = field(default_factory=dict)
    shared_data: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def add_result(self, task_id: str, result: TaskResult) -> None:
        """Add a task result."""
        self.completed_tasks[task_id] = result
        self.task_progress[task_id] = TaskProgress(
            task_id=task_id,
            status=result.status,
            completed_at=datetime.now(),
            progress_percent=100.0 if result.is_successful() else 0.0,
            message=result.error or "Completed",
        )

        # Store output in shared data for dependent tasks
        if result.is_successful():
            self.shared_data[task_id] = result.output


class TaskExecutor:
    """Executes tasks in parallel using appropriate sub-agents."""

    def __init__(
        self,
        progress_callback: Callable[[TaskProgress], None] | None = None,
        max_workers: int = 3,
    ):
        self.progress_callback = progress_callback
        self.max_workers = max_workers
        self._lock = threading.Lock()

    def execute_plan(
        self,
        plan: TaskPlan,
        context: ExecutionContext | None = None
    ) -> ExecutionContext:
        """Execute all tasks in a plan.

        Args:
            plan: TaskPlan to execute
            context: Optional execution context

        Returns:
            ExecutionContext with results
        """
        if context is None:
            context = ExecutionContext(plan=plan)

        # Initialize progress tracking for NEW tasks only
        for task in plan.tasks:
            if task.id not in context.task_progress:
                context.task_progress[task.id] = TaskProgress(
                    task_id=task.id,
                    status=task.status,  # Preserve existing status
                )

        # Build completed set from context
        completed: set[str] = set(context.completed_tasks.keys())

        # Execute tasks in parallel where possible
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures: dict[str, Future] = {}

            # Track which tasks are ready to execute (using context)
            while True:
                # Re-check completed from context on each iteration
                completed = set(context.completed_tasks.keys())

                # Find tasks ready to execute
                pending_tasks = [
                    task for task in plan.tasks
                    if task.status == TaskStatus.PENDING
                    and task.can_execute(completed)
                ]

                # Submit pending tasks
                for task in pending_tasks:
                    if task.id not in futures:
                        task.status = TaskStatus.RUNNING
                        context.task_progress[task.id].status = TaskStatus.RUNNING
                        context.task_progress[task.id].started_at = datetime.now()

                        future = executor.submit(self._execute_task, task, context)
                        futures[task.id] = future

                # Check for completed futures
                done_futures = {tid: f for tid, f in futures.items() if f.done()}
                for task_id, future in done_futures.items():
                    result = future.result()
                    task = plan.get_task(task_id)
                    if task:
                        task.status = result.status
                        task.result = result
                        context.task_progress[task_id].status = result.status
                        context.task_progress[task_id].completed_at = datetime.now()
                        context.add_result(task_id, result)

                        # Update shared data for dependent tasks
                        if result.is_successful():
                            context.shared_data[task_id] = result.output

                    del futures[task_id]

                # Report progress
                if self.progress_callback:
                    for task_id, progress in context.task_progress.items():
                        self.progress_callback(progress)

                # Check if we're done
                if plan.all_completed() or (not pending_tasks and not futures):
                    break

                # Small delay to prevent busy waiting
                time.sleep(0.1)

        return context

    def _execute_task(
        self,
        task: Task,
        context: ExecutionContext
    ) -> TaskResult:
        """Execute a single task.

        Args:
            task: Task to execute
            context: Execution context

        Returns:
            TaskResult
        """
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        # Report initial progress
        if self.progress_callback:
            self.progress_callback(TaskProgress(
                task_id=task.id,
                status=TaskStatus.RUNNING,
                started_at=datetime.now(),
                progress_percent=10.0,
                message="Finding agent...",
            ))

        # Find appropriate agent
        agent = find_agent_for_task(task)
        if not agent:
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=f"No agent found for capability: {task.capability_required}",
                completed_at=datetime.now(),
            )

        # Report agent found
        if self.progress_callback:
            self.progress_callback(TaskProgress(
                task_id=task.id,
                status=TaskStatus.RUNNING,
                started_at=task.started_at,
                progress_percent=30.0,
                message=f"Agent {agent.name} found",
            ))

        # Prepare input data (include results from dependencies)
        input_data = task.input_data.copy()
        for dep_id in task.depends_on:
            if dep_id in context.shared_data:
                input_data[f"from_{dep_id}"] = context.shared_data[dep_id]

        # Update task input
        task.input_data = input_data

        # Execute
        try:
            # Simulate some work time
            time.sleep(0.2)

            exec_result = agent.execute(task)

            if exec_result.success:
                return TaskResult(
                    task_id=task.id,
                    status=TaskStatus.COMPLETED,
                    output=exec_result.output,
                    evidence={"agent": agent.name},
                    completed_at=datetime.now(),
                )
            else:
                return TaskResult(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    error=exec_result.error,
                    completed_at=datetime.now(),
                )
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=str(e),
                completed_at=datetime.now(),
            )

    def execute_single_task(
        self,
        task: Task,
        context: ExecutionContext | None = None
    ) -> tuple[TaskResult, ExecutionContext]:
        """Execute a single task (useful for streaming scenarios).

        Args:
            task: Task to execute
            context: Optional execution context

        Returns:
            Tuple of (TaskResult, updated ExecutionContext)
        """
        if context is None:
            plan = TaskPlan(
                id="temp",
                user_request="",
                tasks=[task],
            )
            context = ExecutionContext(plan=plan)

        result = self._execute_task(task, context)
        task.status = result.status
        task.result = result
        context.add_result(task.id, result)

        return result, context
