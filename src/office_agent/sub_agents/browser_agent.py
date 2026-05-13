"""Browser Agent - Handles web browsing and scraping tasks."""

from __future__ import annotations

import re
from typing import Any

from ..base import (
    BaseSubAgent,
    ExecutionResult,
    Task,
    AgentCapability,
)


class MockBrowser:
    """Mock browser for offline demo.

    In a real implementation, this would use Playwright or Selenium.
    """

    def __init__(self):
        self.current_url: str | None = None
        self.history: list[str] = []

    def navigate(self, url: str) -> str:
        """Simulate navigation to URL."""
        self.current_url = url
        self.history.append(url)
        return f"[Browser] Navigated to {url}"

    def scrape(self, selector: str | None = None) -> dict[str, Any]:
        """Simulate scraping content from current page."""
        if not self.current_url:
            return {"error": "No page loaded"}

        # Mock scraped data based on URL
        mock_data = self._get_mock_page_data(self.current_url)
        return {
            "url": self.current_url,
            "title": f"Page - {self.current_url}",
            "content": mock_data.get("content", ""),
            "links": mock_data.get("links", []),
            "tables": mock_data.get("tables", []),
        }

    def fill_form(self, fields: dict[str, str]) -> str:
        """Simulate filling a form."""
        return f"[Browser] Form filled with fields: {list(fields.keys())}"

    def _get_mock_page_data(self, url: str) -> dict[str, Any]:
        """Get mock data based on URL pattern."""
        if "sales" in url.lower():
            return {
                "content": "Sales Data Report",
                "tables": [
                    {"headers": ["Product", "Q1", "Q2", "Q3", "Q4"],
                     "rows": [["Widget A", "1000", "1200", "1100", "1300"],
                             ["Widget B", "800", "900", "950", "1100"],
                             ["Widget C", "500", "600", "700", "800"]]},
                ],
                "links": ["/products", "/orders", "/reports"],
            }
        elif "news" in url.lower():
            return {
                "content": "Latest News Headlines",
                "tables": [],
                "links": ["/article/1", "/article/2", "/article/3"],
            }
        else:
            return {
                "content": f"Generic page content from {url}",
                "tables": [],
                "links": ["/page1", "/page2"],
            }


class BrowserAgent(BaseSubAgent):
    """Agent for handling browser-related tasks.

    Capabilities:
    - Navigate to URLs
    - Scrape web content
    - Fill and submit forms
    """

    def __init__(self):
        self._browser = MockBrowser()

    @property
    def name(self) -> str:
        return "browser_agent"

    @property
    def capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability.BROWSER_NAVIGATE,
            AgentCapability.BROWSER_SCRAPE,
            AgentCapability.BROWSER_FILL_FORM,
        ]

    def can_handle(self, task: Task) -> bool:
        capability = task.capability_required
        if isinstance(capability, str):
            capability = AgentCapability(capability)
        return capability in self.capabilities

    def execute(self, task: Task) -> ExecutionResult:
        """Execute a browser task."""
        try:
            capability = task.capability_required
            if isinstance(capability, str):
                capability = AgentCapability(capability)

            if capability == AgentCapability.BROWSER_NAVIGATE:
                return self._navigate(task)
            elif capability == AgentCapability.BROWSER_SCRAPE:
                return self._scrape(task)
            elif capability == AgentCapability.BROWSER_FILL_FORM:
                return self._fill_form(task)
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

    def _navigate(self, task: Task) -> ExecutionResult:
        """Navigate to a URL."""
        url = task.input_data.get("url", "")
        if not url:
            return ExecutionResult(
                success=False,
                error="No URL provided",
            )

        result = self._browser.navigate(url)
        return ExecutionResult(
            success=True,
            output={
                "message": result,
                "url": url,
                "history": self._browser.history.copy(),
            },
            metadata={"action": "navigate"},
        )

    def _scrape(self, task: Task) -> ExecutionResult:
        """Scrape content from current page."""
        selector = task.input_data.get("selector")
        data = self._browser.scrape(selector)

        if "error" in data:
            return ExecutionResult(
                success=False,
                error=data["error"],
            )

        return ExecutionResult(
            success=True,
            output=data,
            metadata={
                "action": "scrape",
                "url": data.get("url"),
            },
        )

    def _fill_form(self, task: Task) -> ExecutionResult:
        """Fill a form."""
        fields = task.input_data.get("fields", {})
        result = self._browser.fill_form(fields)

        return ExecutionResult(
            success=True,
            output={"message": result, "fields": fields},
            metadata={"action": "fill_form"},
        )
