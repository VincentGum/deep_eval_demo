"""Planner Agent - High-level reasoning for task plan generation.

The Planner Agent analyzes user requests and generates structured task plans
using a mock high-level reasoning model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import (
    Task,
    TaskPlan,
    TaskPriority,
    AgentCapability,
    create_task_id,
)


# ============================================================================
# Mock High-Level Reasoning Model
# ============================================================================

@dataclass
class TaskTemplate:
    """Template for common office tasks."""
    name: str
    description: str
    capability: AgentCapability
    keywords: tuple[str, ...]
    default_priority: TaskPriority = TaskPriority.MEDIUM


# Predefined task templates for office scenarios
TASK_TEMPLATES: list[TaskTemplate] = [
    # Browser tasks
    TaskTemplate(
        name="Navigate to URL",
        description="Open a web browser and navigate to a specific URL",
        capability=AgentCapability.BROWSER_NAVIGATE,
        keywords=("open", "navigate", "go to", "visit", "browse", "website", "网页", "打开"),
    ),
    TaskTemplate(
        name="Scrape Web Content",
        description="Extract information from a webpage",
        capability=AgentCapability.BROWSER_SCRAPE,
        keywords=("scrape", "extract", "get content", "fetch data", "获取", "抓取"),
    ),

    # API tasks
    TaskTemplate(
        name="Call API Endpoint",
        description="Make an API call to fetch or submit data",
        capability=AgentCapability.API_CALL,
        keywords=("api", "call", "request", "fetch", "get data", "接口", "调用"),
    ),

    # Document tasks
    TaskTemplate(
        name="Read Document",
        description="Read content from a document file",
        capability=AgentCapability.DOC_READ,
        keywords=("read", "open", "load", "parse", "read file", "读取", "打开文件"),
    ),
    TaskTemplate(
        name="Write Document",
        description="Write content to a document file",
        capability=AgentCapability.DOC_WRITE,
        keywords=("write", "save", "create", "generate", "export", "写入", "保存", "生成"),
    ),

    # Data tasks
    TaskTemplate(
        name="Query Data",
        description="Query data from a database or data source",
        capability=AgentCapability.DATA_QUERY,
        keywords=("query", "select", "find", "search", "lookup", "查询", "搜索"),
    ),
    TaskTemplate(
        name="Transform Data",
        description="Transform or process data",
        capability=AgentCapability.DATA_TRANSFORM,
        keywords=("transform", "process", "convert", "clean", "format", "转换", "处理"),
    ),
    TaskTemplate(
        name="Aggregate Data",
        description="Aggregate or summarize data (sum, average, count, etc.)",
        capability=AgentCapability.DATA_AGGREGATE,
        keywords=("sum", "average", "count", "aggregate", "total", "summarize", "统计", "汇总"),
    ),

    # Visualization tasks
    TaskTemplate(
        name="Create Chart",
        description="Create a chart or graph",
        capability=AgentCapability.CHART_CREATE,
        keywords=("chart", "graph", "plot", "visualize", "draw", "图", "图表", "可视化"),
    ),
    TaskTemplate(
        name="Create Table",
        description="Create a formatted table",
        capability=AgentCapability.TABLE_CREATE,
        keywords=("table", "grid", "matrix", "表格", "列表"),
    ),
    TaskTemplate(
        name="Generate Report",
        description="Generate a formatted report",
        capability=AgentCapability.REPORT_GENERATE,
        keywords=("report", "summary", "document", "report", "报告", "总结"),
    ),

    # Communication tasks
    TaskTemplate(
        name="Send Email",
        description="Send an email notification",
        capability=AgentCapability.EMAIL_SEND,
        keywords=("email", "send", "notify", "mail", "邮件", "发送"),
    ),
]


@dataclass
class ReasoningStep:
    """A step in the reasoning process."""
    thought: str
    action: str
    observation: str | None = None


class MockReasoningModel:
    """Mock high-level reasoning model for task planning.

    This simulates a reasoning model (like o1) by analyzing user requests
    and generating structured task plans based on pattern matching.
    """

    def __init__(self):
        self.templates = TASK_TEMPLATES

    def analyze_request(self, user_request: str) -> list[ReasoningStep]:
        """Analyze user request and generate reasoning steps.

        Returns:
            List of reasoning steps leading to task plan
        """
        steps = []
        request_lower = user_request.lower()

        # Step 1: Identify the goal
        steps.append(ReasoningStep(
            thought=f"User request: '{user_request}'",
            action="Understanding the user's goal",
            observation="This appears to be an office automation task"
        ))

        # Step 2: Identify required capabilities
        capabilities_needed = self._identify_capabilities(request_lower)
        steps.append(ReasoningStep(
            thought=f"Required capabilities: {capabilities_needed}",
            action="Identifying required capabilities",
            observation=f"Need to handle: {', '.join(capabilities_needed)}"
        ))

        # Step 3: Determine dependencies
        dependencies = self._analyze_dependencies(request_lower, capabilities_needed)
        steps.append(ReasoningStep(
            thought=f"Task dependencies: {dependencies}",
            action="Analyzing task dependencies",
            observation="Some tasks may need to run sequentially"
        ))

        return steps

    def _identify_capabilities(self, request: str) -> list[str]:
        """Identify required capabilities from request."""
        capabilities = []
        for template in self.templates:
            if any(kw in request for kw in template.keywords):
                if template.capability.value not in capabilities:
                    capabilities.append(template.capability.value)
        return capabilities

    def _analyze_dependencies(self, request: str, capabilities: list[str]) -> dict[str, list[str]]:
        """Analyze dependencies between tasks."""
        # Data tasks usually depend on query tasks
        dependencies = {}
        if AgentCapability.DATA_QUERY.value in capabilities:
            if AgentCapability.API_CALL.value not in dependencies:
                dependencies[AgentCapability.API_CALL.value] = []
        return dependencies

    def generate_task_plan(
        self,
        user_request: str,
        context: dict[str, Any] | None = None
    ) -> TaskPlan:
        """Generate a task plan from user request.

        Args:
            user_request: The user's request
            context: Optional context information

        Returns:
            TaskPlan with ordered tasks
        """
        request_lower = user_request.lower()
        tasks = []
        task_counter = 1

        # Detect task templates based on keywords
        matched_templates = []
        for template in self.templates:
            if any(kw in request_lower for kw in template.keywords):
                matched_templates.append(template)

        # If no templates matched, create a generic task
        if not matched_templates:
            matched_templates.append(TaskTemplate(
                name="Process Request",
                description=user_request,
                capability=AgentCapability.API_CALL,
                keywords=(),
            ))

        # Generate tasks based on matched templates
        for template in matched_templates:
            task = Task(
                id=f"task_{task_counter:03d}",
                description=template.description,
                capability_required=template.capability,
                expected_output=self._generate_expected_output(template, user_request),
                priority=self._determine_priority(template, request_lower),
                input_data=self._extract_input_data(template, user_request, context),
            )
            tasks.append(task)
            task_counter += 1

        # Add data aggregation task if multiple data sources detected
        if self._needs_aggregation(request_lower):
            aggregation_task = Task(
                id=f"task_{task_counter:03d}",
                description="Aggregate and summarize the collected data",
                capability_required=AgentCapability.DATA_AGGREGATE,
                expected_output="Summary statistics and trends",
                priority=TaskPriority.MEDIUM,
                depends_on=[t.id for t in tasks],
            )
            tasks.append(aggregation_task)
            task_counter += 1

        # Add report generation task if report requested
        if self._needs_report(request_lower):
            report_task = Task(
                id=f"task_{task_counter:03d}",
                description="Generate final report with visualizations",
                capability_required=AgentCapability.REPORT_GENERATE,
                expected_output="Formatted report document",
                priority=TaskPriority.HIGH,
                depends_on=[t.id for t in tasks if t.capability_required != AgentCapability.REPORT_GENERATE],
            )
            tasks.append(report_task)

        return TaskPlan(
            id=create_task_id("plan"),
            user_request=user_request,
            tasks=tasks,
            metadata={"reasoning_steps": self.analyze_request(user_request)},
        )

    def _generate_expected_output(self, template: TaskTemplate, request: str) -> str:
        """Generate expected output description."""
        if template.capability == AgentCapability.BROWSER_SCRAPE:
            return "Extracted data from webpage"
        elif template.capability == AgentCapability.API_CALL:
            return "API response data"
        elif template.capability == AgentCapability.DATA_AGGREGATE:
            return "Aggregated statistics"
        elif template.capability == AgentCapability.CHART_CREATE:
            return "Chart image or embedded visualization"
        elif template.capability == AgentCapability.REPORT_GENERATE:
            return "Formatted report document"
        return "Task completed successfully"

    def _determine_priority(self, template: TaskTemplate, request: str) -> TaskPriority:
        """Determine task priority."""
        urgent_keywords = ("urgent", "asap", "immediately", "紧急", "马上", "立即")
        if any(kw in request for kw in urgent_keywords):
            return TaskPriority.URGENT
        return template.default_priority

    def _extract_input_data(
        self,
        template: TaskTemplate,
        request: str,
        context: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Extract input data for the task."""
        import re
        data = {}

        # Extract URLs
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, request)
        if urls:
            data["url"] = urls[0]

        # Extract file paths
        path_pattern = r'[A-Za-z]:\\[^\s]+|/[^/\s]+'
        paths = re.findall(path_pattern, request)
        if paths:
            data["file_path"] = paths[0]

        # Add capability-specific default parameters
        # These will be used if not provided, avoiding missing parameter errors
        data.update(self._get_default_params(template.capability, request))

        # Add context if available
        if context:
            # Merge context, but capability defaults take precedence
            for key, value in context.items():
                if key not in data:
                    data[key] = value

        return data

    def _get_default_params(
        self,
        capability: AgentCapability,
        request: str
    ) -> dict[str, Any]:
        """Get default parameters for a capability type.

        This simulates what a real LLM would generate as parameters,
        providing sensible defaults for common scenarios.
        """
        defaults = {}

        if capability == AgentCapability.BROWSER_NAVIGATE:
            # Default to example URL or placeholder
            defaults["url"] = "https://www.example.com"
            defaults["action"] = "navigate"

        elif capability == AgentCapability.BROWSER_SCRAPE:
            # Default to example URL
            defaults["url"] = "https://www.example.com"
            defaults["selectors"] = ["h1", "p", "article"]

        elif capability == AgentCapability.API_CALL:
            # Default API endpoint
            defaults["endpoint"] = "/api/data"
            defaults["method"] = "GET"
            defaults["params"] = {}

        elif capability == AgentCapability.DOC_READ:
            # Default document path
            defaults["file_path"] = "/tmp/document.txt"
            defaults["format"] = "text"

        elif capability == AgentCapability.DOC_WRITE:
            # Default output path with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            defaults["file_path"] = f"/tmp/output_{timestamp}.txt"
            defaults["format"] = "text"
            defaults["content"] = ""

        elif capability == AgentCapability.DATA_QUERY:
            # Default database query
            defaults["table"] = "data"
            defaults["filters"] = {}

        elif capability == AgentCapability.DATA_AGGREGATE:
            # Default aggregation parameters
            defaults["operations"] = ["sum", "average", "count"]

        elif capability == AgentCapability.DATA_TRANSFORM:
            # Default transformation
            defaults["transform_type"] = "normalize"

        elif capability == AgentCapability.CHART_CREATE:
            # Default chart parameters
            defaults["chart_type"] = "bar"
            defaults["title"] = "Data Visualization"
            defaults["x_axis"] = "Category"
            defaults["y_axis"] = "Value"

        elif capability == AgentCapability.TABLE_CREATE:
            # Default table parameters
            defaults["headers"] = ["Column A", "Column B", "Column C"]
            defaults["rows"] = []

        elif capability == AgentCapability.REPORT_GENERATE:
            # Default report parameters
            from datetime import datetime
            defaults["title"] = "Report"
            defaults["format"] = "markdown"
            defaults["include_charts"] = True

        elif capability == AgentCapability.EMAIL_SEND:
            # Default email parameters
            defaults["to"] = "recipient@example.com"
            defaults["subject"] = "Report"
            defaults["body"] = ""

        return defaults

    def _needs_aggregation(self, request: str) -> bool:
        """Check if task needs data aggregation."""
        agg_keywords = ("total", "sum", "average", "count", "summarize", "统计", "汇总")
        return any(kw in request for kw in agg_keywords)

    def _needs_report(self, request: str) -> bool:
        """Check if task needs report generation."""
        report_keywords = ("report", "summary", "document", "export", "报告", "总结", "生成")
        return any(kw in request for kw in report_keywords)


# ============================================================================
# Planner Agent
# ============================================================================

class PlannerAgent:
    """Planner Agent that generates task plans.

    The Planner Agent uses the MockReasoningModel to analyze user requests
    and generate structured task plans with proper dependencies.
    """

    def __init__(self, reasoning_model: MockReasoningModel | None = None):
        self.reasoning_model = reasoning_model or MockReasoningModel()

    def create_plan(
        self,
        user_request: str,
        context: dict[str, Any] | None = None
    ) -> tuple[TaskPlan, list[ReasoningStep]]:
        """Create a task plan from user request.

        Args:
            user_request: The user's request
            context: Optional context (user info, available tools, etc.)

        Returns:
            Tuple of (TaskPlan, ReasoningSteps)
        """
        reasoning_steps = self.reasoning_model.analyze_request(user_request)
        task_plan = self.reasoning_model.generate_task_plan(user_request, context)

        return task_plan, reasoning_steps

    def explain_plan(self, plan: TaskPlan) -> str:
        """Generate human-readable explanation of the plan."""
        lines = [
            f"## Task Plan: {plan.id}",
            f"**User Request**: {plan.user_request}",
            "",
            "### Tasks:",
        ]

        for i, task in enumerate(plan.tasks, 1):
            deps = f" (depends on: {', '.join(task.depends_on)})" if task.depends_on else ""
            lines.append(f"{i}. [{task.priority.value.upper()}] {task.description}{deps}")
            lines.append(f"   - Capability: {task.capability_required.value}")
            lines.append(f"   - Expected: {task.expected_output}")
            lines.append("")

        summary = plan.get_execution_summary()
        lines.extend([
            "### Execution Summary:",
            f"- Total Tasks: {summary['total']}",
            f"- Expected Duration: ~{summary['total'] * 2} minutes",
        ])

        return "\n".join(lines)
