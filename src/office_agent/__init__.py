"""Office Agent - AI Office Assistant with PEV Architecture.

This module provides an AI office assistant that uses PEV (Plan-Execute-Verify)
architecture with parallel task execution and human-in-the-loop support.
"""

from .base import (
    Task,
    TaskStatus,
    TaskResult,
    TaskPlan,
    AgentCapability,
    BaseSubAgent,
    ExecutionResult,
)
from .planner import PlannerAgent
from .executor import TaskExecutor
from .verify import VerifyAgent
from .human_loop import HumanInTheLoop
from .office_agent import (
    OfficeAgent,
    WorkflowState,
    WorkflowProgress,
    WorkflowResult,
    invoke_office_agent,
    build_office_agent_graph,
)
from .scenarios import (
    OfficeScenario,
    WEEKLY_SALES_REPORT,
    CUSTOMER_RESEARCH,
    MEETING_PREPARATION,
    list_scenarios,
    get_scenario,
    run_scenario,
    demo_all,
)

__all__ = [
    # Base classes
    "Task",
    "TaskStatus",
    "TaskResult",
    "TaskPlan",
    "AgentCapability",
    "BaseSubAgent",
    "ExecutionResult",
    # Core agents
    "PlannerAgent",
    "TaskExecutor",
    "VerifyAgent",
    "HumanInTheLoop",
    "OfficeAgent",
    # Workflow types
    "WorkflowState",
    "WorkflowProgress",
    "WorkflowResult",
    # Entry point
    "invoke_office_agent",
    "build_office_agent_graph",
    # Scenarios
    "OfficeScenario",
    "WEEKLY_SALES_REPORT",
    "CUSTOMER_RESEARCH",
    "MEETING_PREPARATION",
    "list_scenarios",
    "get_scenario",
    "run_scenario",
    "demo_all",
]
