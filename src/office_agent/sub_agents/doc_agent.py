"""Document Agent - Handles document read/write operations."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..base import (
    BaseSubAgent,
    ExecutionResult,
    Task,
    AgentCapability,
)


class MockFileSystem:
    """Mock file system for offline demo."""

    _files: dict[str, str] = {
        "/documents/report_template.md": """# Weekly Report Template

## Summary
{summary}

## Metrics
{metrics}

## Action Items
{action_items}
""",
        "/documents/meeting_notes.md": """# Meeting Notes - {date}

## Attendees
{attendees}

## Agenda
{agenda}

## Notes
{notes}

## Next Steps
{next_steps}
""",
        "/documents/sales_data.json": json.dumps({
            "product": "Widget A",
            "units_sold": 4600,
            "revenue": 137954,
            "regions": ["North", "South", "East", "West"],
        }, indent=2),
    }

    def __init__(self):
        self._written_files: dict[str, str] = {}

    def read(self, path: str) -> str | None:
        """Read file content."""
        return self._files.get(path) or self._written_files.get(path)

    def write(self, path: str, content: str) -> bool:
        """Write content to file."""
        self._written_files[path] = content
        return True

    def list_files(self, directory: str = "/documents") -> list[str]:
        """List files in directory."""
        all_files = list(self._files.keys()) + list(self._written_files.keys())
        return [f for f in all_files if f.startswith(directory)]


class DocAgent(BaseSubAgent):
    """Agent for handling document operations.

    Capabilities:
    - Read documents (markdown, JSON, CSV, etc.)
    - Write documents
    - Parse document content
    - Convert between formats
    """

    def __init__(self):
        self._fs = MockFileSystem()

    @property
    def name(self) -> str:
        return "doc_agent"

    @property
    def capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability.DOC_READ,
            AgentCapability.DOC_WRITE,
            AgentCapability.DOC_PARSE,
            AgentCapability.DOC_CONVERT,
        ]

    def can_handle(self, task: Task) -> bool:
        capability = task.capability_required
        if isinstance(capability, str):
            capability = AgentCapability(capability)
        return capability in self.capabilities

    def execute(self, task: Task) -> ExecutionResult:
        """Execute a document task."""
        try:
            capability = task.capability_required
            if isinstance(capability, str):
                capability = AgentCapability(capability)

            if capability == AgentCapability.DOC_READ:
                return self._read(task)
            elif capability == AgentCapability.DOC_WRITE:
                return self._write(task)
            elif capability == AgentCapability.DOC_PARSE:
                return self._parse(task)
            elif capability == AgentCapability.DOC_CONVERT:
                return self._convert(task)
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

    def _read(self, task: Task) -> ExecutionResult:
        """Read a document."""
        # Support both 'path' and 'file_path' for compatibility
        path = task.input_data.get("path") or task.input_data.get("file_path", "")
        if not path:
            return ExecutionResult(
                success=False,
                error="No file path provided",
            )

        content = self._fs.read(path)
        if content is None:
            return ExecutionResult(
                success=False,
                error=f"File not found: {path}",
            )

        return ExecutionResult(
            success=True,
            output={
                "path": path,
                "content": content,
                "size": len(content),
            },
            metadata={"action": "read"},
        )

    def _write(self, task: Task) -> ExecutionResult:
        """Write a document."""
        # Support both 'path' and 'file_path' for compatibility
        path = task.input_data.get("path") or task.input_data.get("file_path", "")
        content = task.input_data.get("content", "")
        template_data = task.input_data.get("template_data", {})

        if not path:
            return ExecutionResult(
                success=False,
                error="No file path provided",
            )

        # Apply template if provided
        if template_data and "{content}" in content:
            content = content.format(**template_data)

        success = self._fs.write(path, content)

        return ExecutionResult(
            success=success,
            output={
                "path": path,
                "written": success,
                "size": len(content) if success else 0,
            },
            metadata={"action": "write"},
        )

    def _parse(self, task: Task) -> ExecutionResult:
        """Parse document content."""
        content = task.input_data.get("content", "")
        format_type = task.input_data.get("format", "auto")

        parsed = {}
        if format_type in ("auto", "json") and content.strip().startswith("{"):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                pass

        if not parsed:
            # Try to extract structured data from text
            lines = content.strip().split("\n")
            parsed = {"lines": len(lines), "content": content}

        return ExecutionResult(
            success=True,
            output=parsed,
            metadata={"action": "parse", "format": format_type},
        )

    def _convert(self, task: Task) -> ExecutionResult:
        """Convert document between formats."""
        source_format = task.input_data.get("source_format", "markdown")
        target_format = task.input_data.get("target_format", "html")
        content = task.input_data.get("content", "")

        # Mock conversion
        if target_format == "html":
            converted = f"<html><body><pre>{content}</pre></body></html>"
        elif target_format == "pdf":
            converted = "[PDF binary data - mock]"
        else:
            converted = content

        return ExecutionResult(
            success=True,
            output={
                "source_format": source_format,
                "target_format": target_format,
                "converted_content": converted,
            },
            metadata={"action": "convert"},
        )
