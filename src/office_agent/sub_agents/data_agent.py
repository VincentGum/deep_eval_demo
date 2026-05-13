"""Data Agent - Handles data processing and analysis tasks."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from ..base import (
    BaseSubAgent,
    ExecutionResult,
    Task,
    AgentCapability,
)


class DataAgent(BaseSubAgent):
    """Agent for handling data operations.

    Capabilities:
    - Query data from various sources
    - Transform and process data
    - Aggregate and summarize data
    - Export data to various formats
    """

    # Mock data stores
    _mock_data = {
        "sales": {
            "Q1": {"Widget A": 1000, "Widget B": 800, "Widget C": 500},
            "Q2": {"Widget A": 1200, "Widget B": 900, "Widget C": 600},
            "Q3": {"Widget A": 1100, "Widget B": 950, "Widget C": 700},
            "Q4": {"Widget A": 1300, "Widget B": 1100, "Widget C": 800},
        },
        "employees": [
            {"name": "Alice", "dept": "Sales", "salary": 75000, "experience": 5},
            {"name": "Bob", "dept": "Engineering", "salary": 95000, "experience": 8},
            {"name": "Charlie", "dept": "Marketing", "salary": 70000, "experience": 3},
            {"name": "Diana", "dept": "Sales", "salary": 80000, "experience": 6},
            {"name": "Eve", "dept": "Engineering", "salary": 90000, "experience": 7},
        ],
    }

    def __init__(self):
        self.query_history: list[str] = []

    @property
    def name(self) -> str:
        return "data_agent"

    @property
    def capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability.DATA_QUERY,
            AgentCapability.DATA_TRANSFORM,
            AgentCapability.DATA_AGGREGATE,
            AgentCapability.DATA_EXPORT,
        ]

    def can_handle(self, task: Task) -> bool:
        capability = task.capability_required
        if isinstance(capability, str):
            capability = AgentCapability(capability)
        return capability in self.capabilities

    def execute(self, task: Task) -> ExecutionResult:
        """Execute a data task."""
        try:
            capability = task.capability_required
            if isinstance(capability, str):
                capability = AgentCapability(capability)

            if capability == AgentCapability.DATA_QUERY:
                return self._query(task)
            elif capability == AgentCapability.DATA_TRANSFORM:
                return self._transform(task)
            elif capability == AgentCapability.DATA_AGGREGATE:
                return self._aggregate(task)
            elif capability == AgentCapability.DATA_EXPORT:
                return self._export(task)
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

    def _query(self, task: Task) -> ExecutionResult:
        """Query data from a source."""
        source = task.input_data.get("source", "sales")
        filters = task.input_data.get("filters", {})

        self.query_history.append(source)

        data = self._mock_data.get(source, {})

        # Apply filters (mock)
        if filters:
            # Simple filtering logic
            if source == "employees" and filters.get("dept"):
                data = [e for e in data if e.get("dept") == filters["dept"]]

        return ExecutionResult(
            success=True,
            output=data,
            metadata={
                "action": "query",
                "source": source,
                "filters": filters,
            },
        )

    def _transform(self, task: Task) -> ExecutionResult:
        """Transform data."""
        data = task.input_data.get("data", {})
        operations = task.input_data.get("operations", [])

        transformed = data.copy()

        for op in operations:
            if op == "normalize":
                # Simple normalization
                if isinstance(transformed, dict):
                    total = sum(transformed.values()) if all(isinstance(v, (int, float)) for v in transformed.values()) else 1
                    transformed = {k: v/total for k, v in transformed.items()}
            elif op == "sort":
                if isinstance(transformed, dict):
                    transformed = dict(sorted(transformed.items(), key=lambda x: x[1], reverse=True))
            elif op == "filter_empty":
                if isinstance(transformed, dict):
                    transformed = {k: v for k, v in transformed.items() if v}

        return ExecutionResult(
            success=True,
            output=transformed,
            metadata={"action": "transform", "operations": operations},
        )

    def _aggregate(self, task: Task) -> ExecutionResult:
        """Aggregate data."""
        data = task.input_data.get("data", {})
        agg_type = task.input_data.get("aggregation", "sum")
        group_by = task.input_data.get("group_by")

        result = {}

        if isinstance(data, dict):
            if agg_type == "sum":
                result["total"] = sum(v for v in data.values() if isinstance(v, (int, float)))
            elif agg_type == "average":
                nums = [v for v in data.values() if isinstance(v, (int, float))]
                result["average"] = sum(nums) / len(nums) if nums else 0
            elif agg_type == "count":
                result["count"] = len(data)
            elif agg_type == "max":
                result["max"] = max(v for v in data.values() if isinstance(v, (int, float))) if data else None
            elif agg_type == "min":
                result["min"] = min(v for v in data.values() if isinstance(v, (int, float))) if data else None

            # Also provide per-key breakdown
            result["breakdown"] = data

        elif isinstance(data, list):
            if agg_type == "sum":
                result["total"] = sum(e.get("salary", 0) for e in data)
            elif agg_type == "average":
                result["average"] = sum(e.get("salary", 0) for e in data) / len(data) if data else 0
            elif agg_type == "count":
                result["count"] = len(data)

            # Group by if specified
            if group_by:
                grouped = {}
                for item in data:
                    key = item.get(group_by, "unknown")
                    if key not in grouped:
                        grouped[key] = []
                    grouped[key].append(item)
                result["grouped_by"] = grouped

        return ExecutionResult(
            success=True,
            output=result,
            metadata={
                "action": "aggregate",
                "aggregation": agg_type,
                "group_by": group_by,
            },
        )

    def _export(self, task: Task) -> ExecutionResult:
        """Export data to a format."""
        data = task.input_data.get("data", {})
        format_type = task.input_data.get("format", "json")

        if format_type == "json":
            exported = json.dumps(data, indent=2, default=str)
        elif format_type == "csv":
            # Simple CSV conversion
            if isinstance(data, dict):
                lines = ["key,value"]
                for k, v in data.items():
                    lines.append(f"{k},{v}")
                exported = "\n".join(lines)
            else:
                exported = "[CSV conversion not supported for this data type]"
        elif format_type == "markdown":
            if isinstance(data, dict):
                lines = ["| Key | Value |", "| --- | --- |"]
                for k, v in data.items():
                    lines.append(f"| {k} | {v} |")
                exported = "\n".join(lines)
            else:
                exported = str(data)
        else:
            exported = str(data)

        return ExecutionResult(
            success=True,
            output={
                "format": format_type,
                "data": exported,
                "size": len(exported),
            },
            metadata={"action": "export"},
        )
