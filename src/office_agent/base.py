"""Base classes and interfaces for Office Agent.

This module provides shared abstractions that can be reused across
different agent implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


# ============================================================================
# Task & Plan Definitions
# ============================================================================

class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_HUMAN_INPUT = "waiting_human_input"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    status: TaskStatus
    output: Any = None
    error: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_successful(self) -> bool:
        return self.status == TaskStatus.COMPLETED and self.error is None


@dataclass
class Task:
    """A single task in a task plan."""
    id: str
    description: str
    capability_required: AgentCapability | str
    input_data: dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    depends_on: list[str] = field(default_factory=list)
    timeout_seconds: int = 300
    status: TaskStatus = TaskStatus.PENDING
    result: TaskResult | None = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def can_execute(self, completed_tasks: set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_tasks for dep in self.depends_on)


@dataclass
class TaskPlan:
    """A plan containing multiple tasks."""
    id: str
    user_request: str
    tasks: list[Task]
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_task(self, task_id: str) -> Task | None:
        """Get task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def get_pending_tasks(self, completed: set[str]) -> list[Task]:
        """Get tasks that are ready to execute."""
        return [
            task for task in self.tasks
            if task.status == TaskStatus.PENDING and task.can_execute(completed)
        ]

    def all_completed(self) -> bool:
        """Check if all tasks are completed."""
        return all(task.status == TaskStatus.COMPLETED for task in self.tasks)

    def get_execution_summary(self) -> dict[str, Any]:
        """Get summary of task execution."""
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)
        pending = sum(1 for t in self.tasks if t.status == TaskStatus.PENDING)
        waiting = sum(1 for t in self.tasks if t.status == TaskStatus.WAITING_HUMAN_INPUT)

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "waiting_human_input": waiting,
            "completion_rate": completed / total if total > 0 else 0,
        }


# ============================================================================
# Agent Capability Definitions
# ============================================================================

class AgentCapability(str, Enum):
    """Capabilities that sub-agents can provide."""
    # Web & Browser
    BROWSER_NAVIGATE = "browser_navigate"
    BROWSER_SCRAPE = "browser_scrape"
    BROWSER_FILL_FORM = "browser_fill_form"

    # API
    API_CALL = "api_call"
    API_AUTH = "api_auth"
    API_PAGINATE = "api_paginate"

    # Document
    DOC_READ = "doc_read"
    DOC_WRITE = "doc_write"
    DOC_PARSE = "doc_parse"
    DOC_CONVERT = "doc_convert"

    # Data
    DATA_QUERY = "data_query"
    DATA_TRANSFORM = "data_transform"
    DATA_AGGREGATE = "data_aggregate"
    DATA_EXPORT = "data_export"

    # Visualization
    CHART_CREATE = "chart_create"
    TABLE_CREATE = "table_create"
    REPORT_GENERATE = "report_generate"

    # Communication
    EMAIL_SEND = "email_send"
    NOTIFICATION_SEND = "notification_send"

    # File System
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_LIST = "file_list"


# ============================================================================
# Base Classes
# ============================================================================

@dataclass
class ExecutionResult:
    """Result of executing an agent or tool."""
    success: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseSubAgent(ABC):
    """Abstract base class for sub-agents.

    Sub-agents are specialized agents that can execute specific tasks
    based on their capabilities.

    Example:
        class BrowserAgent(BaseSubAgent):
            @property
            def capabilities(self) -> list[AgentCapability]:
                return [AgentCapability.BROWSER_NAVIGATE, AgentCapability.BROWSER_SCRAPE]

            def execute(self, task: Task) -> ExecutionResult:
                # Implementation
                ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name."""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> list[AgentCapability]:
        """Capabilities this agent provides."""
        ...

    @abstractmethod
    def can_handle(self, task: Task) -> bool:
        """Check if this agent can handle the given task."""
        ...

    @abstractmethod
    def execute(self, task: Task) -> ExecutionResult:
        """Execute the given task.

        Args:
            task: Task to execute

        Returns:
            ExecutionResult with success status and output
        """
        ...

    def validate_input(self, task: Task) -> tuple[bool, str | None]:
        """Validate task input before execution.

        Args:
            task: Task to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ["input_data"]
        for field_name in required_fields:
            if not hasattr(task, field_name):
                return False, f"Missing required field: {field_name}"
        return True, None


class BaseTool(ABC):
    """Abstract base class for tools."""

    @property
    def name(self) -> str:
        """Tool name."""
        return self.__class__.__name__

    @property
    def description(self) -> str:
        """Tool description."""
        return ""

    @abstractmethod
    def invoke(self, **kwargs) -> Any:
        """Invoke the tool with given parameters."""
        ...

    def validate(self, **kwargs) -> tuple[bool, str | None]:
        """Validate parameters before invocation."""
        return True, None


# ============================================================================
# Shared Utilities
# ============================================================================

def create_task_id(prefix: str = "task") -> str:
    """Create a unique task ID."""
    import uuid
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def format_task_result(task: Task) -> str:
    """Format task result as readable string."""
    lines = [
        f"Task: {task.id}",
        f"Description: {task.description}",
        f"Status: {task.status.value}",
        f"Capability: {task.capability_required}",
    ]

    if task.result:
        if task.result.output:
            lines.append(f"Output: {task.result.output}")
        if task.result.error:
            lines.append(f"Error: {task.result.error}")

    return "\n".join(lines)


# ============================================================================
# Agent Registry
# ============================================================================

class AgentRegistry:
    """Registry for managing sub-agents.

    This allows dynamic registration and lookup of sub-agents
    based on their capabilities.
    """

    def __init__(self):
        self._agents: dict[str, BaseSubAgent] = {}
        self._capability_map: dict[AgentCapability, list[BaseSubAgent]] = {}

    def register(self, agent: BaseSubAgent) -> None:
        """Register a sub-agent."""
        self._agents[agent.name] = agent
        for capability in agent.capabilities:
            if capability not in self._capability_map:
                self._capability_map[capability] = []
            if agent not in self._capability_map[capability]:
                self._capability_map[capability].append(agent)

    def get_agent(self, name: str) -> BaseSubAgent | None:
        """Get agent by name."""
        return self._agents.get(name)

    def find_agents_for_task(self, task: Task) -> list[BaseSubAgent]:
        """Find agents that can handle a task."""
        capability = task.capability_required
        if isinstance(capability, str):
            try:
                capability = AgentCapability(capability)
            except ValueError:
                return []

        return self._capability_map.get(capability, [])

    def list_agents(self) -> list[BaseSubAgent]:
        """List all registered agents."""
        return list(self._agents.values())

    def get_capabilities(self) -> list[AgentCapability]:
        """List all available capabilities."""
        return list(self._capability_map.keys())
