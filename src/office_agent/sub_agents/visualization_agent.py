"""Visualization Agent - Handles chart and table creation."""

from __future__ import annotations

from typing import Any

from ..base import (
    BaseSubAgent,
    ExecutionResult,
    Task,
    AgentCapability,
)


class VisualizationAgent(BaseSubAgent):
    """Agent for creating visualizations.

    Capabilities:
    - Create charts (bar, line, pie, etc.)
    - Create tables
    - Generate reports
    """

    @property
    def name(self) -> str:
        return "visualization_agent"

    @property
    def capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability.CHART_CREATE,
            AgentCapability.TABLE_CREATE,
            AgentCapability.REPORT_GENERATE,
        ]

    def can_handle(self, task: Task) -> bool:
        capability = task.capability_required
        if isinstance(capability, str):
            capability = AgentCapability(capability)
        return capability in self.capabilities

    def execute(self, task: Task) -> ExecutionResult:
        """Execute a visualization task."""
        try:
            capability = task.capability_required
            if isinstance(capability, str):
                capability = AgentCapability(capability)

            if capability == AgentCapability.CHART_CREATE:
                return self._create_chart(task)
            elif capability == AgentCapability.TABLE_CREATE:
                return self._create_table(task)
            elif capability == AgentCapability.REPORT_GENERATE:
                return self._generate_report(task)
            else:
                return ExecutionResult(
                    success=False,
                    error=f"Unknown capability: {capability}",
                )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
            )

    def _create_chart(self, task: Task) -> ExecutionResult:
        """Create a chart."""
        chart_type = task.input_data.get("type", "bar")
        data = task.input_data.get("data", {})
        title = task.input_data.get("title", "Chart")
        labels = task.input_data.get("labels", [])

        # Generate mock chart data
        chart_config = {
            "type": chart_type,
            "title": title,
            "data": data,
            "labels": labels,
        }

        # Generate ASCII representation for demo
        ascii_chart = self._generate_ascii_chart(chart_type, data, labels)

        return ExecutionResult(
            success=True,
            output={
                "chart_type": chart_type,
                "title": title,
                "config": chart_config,
                "ascii_preview": ascii_chart,
                "format": "png",  # or svg, html
            },
            metadata={"action": "create_chart"},
        )

    def _create_table(self, task: Task) -> ExecutionResult:
        """Create a table."""
        headers = task.input_data.get("headers", [])
        rows = task.input_data.get("rows", [])
        title = task.input_data.get("title", "Table")

        # Generate markdown table
        table_md = self._generate_markdown_table(headers, rows, title)

        return ExecutionResult(
            success=True,
            output={
                "title": title,
                "headers": headers,
                "row_count": len(rows),
                "markdown": table_md,
                "format": "markdown",  # or html, csv
            },
            metadata={"action": "create_table"},
        )

    def _generate_report(self, task: Task) -> ExecutionResult:
        """Generate a formatted report."""
        sections = task.input_data.get("sections", [])
        title = task.input_data.get("title", "Report")
        include_charts = task.input_data.get("include_charts", True)

        # Build report structure
        report = self._generate_markdown_report(title, sections, include_charts)

        return ExecutionResult(
            success=True,
            output={
                "title": title,
                "sections": sections,
                "content": report,
                "word_count": len(report.split()),
                "format": "markdown",  # or pdf, html, docx
            },
            metadata={"action": "generate_report"},
        )

    def _generate_ascii_chart(
        self,
        chart_type: str,
        data: dict,
        labels: list
    ) -> str:
        """Generate ASCII art chart."""
        if not data:
            return "[No data to display]"

        max_val = max(data.values()) if data else 1
        lines = []

        if chart_type == "bar":
            for label, value in data.items():
                bar_len = int((value / max_val) * 40)
                bar = "█" * bar_len
                lines.append(f"{label:12} | {bar} {value}")
        elif chart_type == "horizontal_bar":
            for label, value in data.items():
                bar_len = int((value / max_val) * 40)
                bar = "█" * bar_len
                lines.append(f"{bar} {value:5} | {label}")
        else:
            # Default to simple representation
            for label, value in data.items():
                lines.append(f"{label}: {value}")

        return "\n".join(lines)

    def _generate_markdown_table(
        self,
        headers: list,
        rows: list,
        title: str
    ) -> str:
        """Generate markdown table."""
        lines = [f"## {title}\n"]

        if not headers:
            return "\n".join(lines) + "[No data]"

        # Header
        lines.append(f"| {' | '.join(str(h) for h in headers)} |")
        lines.append(f"| {' | '.join(['---'] * len(headers))} |")

        # Rows
        for row in rows:
            lines.append(f"| {' | '.join(str(c) for c in row)} |")

        return "\n".join(lines)

    def _generate_markdown_report(
        self,
        title: str,
        sections: list,
        include_charts: bool
    ) -> str:
        """Generate markdown report."""
        lines = [
            f"# {title}",
            "",
            f"*Generated on: ...*",
            "",
        ]

        for section in sections:
            section_title = section.get("title", "Section")
            section_content = section.get("content", "")
            section_data = section.get("data", {})

            lines.extend([
                f"## {section_title}",
                "",
                section_content,
                "",
            ])

            # Add data visualization if present
            if include_charts and section_data:
                if "table" in section:
                    table = section["table"]
                    lines.append(self._generate_markdown_table(
                        table.get("headers", []),
                        table.get("rows", []),
                        ""
                    ))
                    lines.append("")

                if "chart" in section:
                    chart = section["chart"]
                    lines.append("```")
                    lines.append(self._generate_ascii_chart(
                        chart.get("type", "bar"),
                        chart.get("data", {}),
                        chart.get("labels", [])
                    ))
                    lines.append("```")
                    lines.append("")

        return "\n".join(lines)
