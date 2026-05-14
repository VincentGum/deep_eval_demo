"""Office Scenarios - Common office automation use cases.

This module provides pre-configured office scenarios that demonstrate
the Office Agent's capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import Task, TaskPlan, TaskPriority, AgentCapability
from .planner import PlannerAgent, MockReasoningModel
from .office_agent import OfficeAgent, WorkflowResult


@dataclass
class OfficeScenario:
    """A pre-configured office automation scenario."""
    name: str
    description: str
    user_request: str
    expected_tasks: list[str]
    context: dict[str, Any]


# ============================================================================
# Scenario 1: Weekly Sales Report Generation
# ============================================================================

WEEKLY_SALES_REPORT = OfficeScenario(
    name="Weekly Sales Report",
    description="Generate a comprehensive weekly sales report with charts and statistics.",
    user_request=(
        "Generate a weekly sales report for this week. "
        "I need to fetch sales data from the API, calculate totals and averages, "
        "create a bar chart showing sales by product, and save everything to a report document."
    ),
    expected_tasks=[
        "Fetch sales data from API",
        "Aggregate data (sum, average)",
        "Create sales chart",
        "Generate report document",
    ],
    context={
        "department": "Sales",
        "period": "this_week",
        "output_format": "markdown",
    },
)


# ============================================================================
# Scenario 2: Customer Research Report
# ============================================================================

CUSTOMER_RESEARCH = OfficeScenario(
    name="Customer Research Report",
    description="Research customer data, scrape competitor websites, and compile findings.",
    user_request=(
        "I need to research our top 5 customers. "
        "Please look up their information from our database, "
        "visit their websites to understand their business, "
        "and create a comprehensive research report."
    ),
    expected_tasks=[
        "Query customer data from database",
        "Browse customer websites",
        "Scrape competitor information",
        "Compile research report",
    ],
    context={
        "customer_count": 5,
        "include_competitors": True,
        "output_format": "document",
    },
)


# ============================================================================
# Scenario 3: Meeting Preparation Pack
# ============================================================================

MEETING_PREPARATION = OfficeScenario(
    name="Meeting Preparation",
    description="Prepare a comprehensive meeting pack with agenda, attendees, and relevant data.",
    user_request=(
        "Prepare a meeting pack for my quarterly review meeting. "
        "I need the agenda, attendee list, previous meeting notes, "
        "current project status from our tracking system, "
        "and some key metrics. Format everything in a nice document."
    ),
    expected_tasks=[
        "Fetch meeting details",
        "Get attendee information",
        "Retrieve previous meeting notes",
        "Query project status",
        "Fetch key metrics",
        "Generate formatted document",
    ],
    context={
        "meeting_type": "quarterly_review",
        "include_calendar": True,
        "output_format": "document",
    },
)


# ============================================================================
# Scenario Factory
# ============================================================================

def get_scenario(scenario_name: str) -> OfficeScenario | None:
    """Get a scenario by name."""
    scenarios = {
        "weekly_sales_report": WEEKLY_SALES_REPORT,
        "customer_research": CUSTOMER_RESEARCH,
        "meeting_preparation": MEETING_PREPARATION,
    }
    return scenarios.get(scenario_name.lower().replace(" ", "_"))


def list_scenarios() -> list[OfficeScenario]:
    """List all available scenarios."""
    return [WEEKLY_SALES_REPORT, CUSTOMER_RESEARCH, MEETING_PREPARATION]


def run_scenario(
    scenario: OfficeScenario,
    verbose: bool = True
) -> WorkflowResult:
    """Run a scenario and return the result.

    Args:
        scenario: The scenario to run
        verbose: Whether to print progress

    Returns:
        WorkflowResult
    """
    if verbose:
        print("=" * 60)
        print(f"SCENARIO: {scenario.name}")
        print("=" * 60)
        print(f"Description: {scenario.description}")
        print(f"Request: {scenario.user_request}")
        print("=" * 60)
        print()

    # Progress callback for verbose mode
    def progress_callback(progress):
        if verbose:
            print(f"[{progress.state.value.upper()}] {progress.message}")

    # Create agent and execute
    agent = OfficeAgent(progress_callback=progress_callback)
    result = agent.execute(scenario.user_request, scenario.context)

    if verbose:
        print()
        print("=" * 60)
        print("RESULT SUMMARY")
        print("=" * 60)
        print(f"Success: {result.success}")
        print(f"Execution Time: {result.execution_time_seconds:.2f}s")
        print(f"Tasks Completed: {sum(1 for t in result.plan.tasks if t.status.value == 'completed')}/{len(result.plan.tasks)}")

        if result.human_inputs:
            print(f"Human Inputs: {len(result.human_inputs)}")

        if result.errors:
            print(f"Errors: {result.errors}")

        print()
        print(result.output.get("summary", ""))

    return result


# ============================================================================
# Demo Functions
# ============================================================================

def demo_weekly_sales_report():
    """Demo: Weekly Sales Report Generation."""
    print("\n" + "=" * 60)
    print("DEMO: Weekly Sales Report Generation")
    print("=" * 60)
    return run_scenario(WEEKLY_SALES_REPORT, verbose=True)


def demo_customer_research():
    """Demo: Customer Research Report."""
    print("\n" + "=" * 60)
    print("DEMO: Customer Research Report")
    print("=" * 60)
    return run_scenario(CUSTOMER_RESEARCH, verbose=True)


def demo_meeting_preparation():
    """Demo: Meeting Preparation Pack."""
    print("\n" + "=" * 60)
    print("DEMO: Meeting Preparation Pack")
    print("=" * 60)
    return run_scenario(MEETING_PREPARATION, verbose=True)


def demo_all():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("RUNNING ALL OFFICE AGENT DEMOS")
    print("=" * 60)

    results = []

    for scenario in list_scenarios():
        result = run_scenario(scenario, verbose=False)
        results.append((scenario.name, result.success))

    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"  [{status}] {name}")

    return results
