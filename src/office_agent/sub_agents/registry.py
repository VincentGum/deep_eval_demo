"""Agent Registry - Manages sub-agents for task execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import (
    AgentRegistry as BaseRegistry,
    BaseSubAgent,
)

if TYPE_CHECKING:
    from ..base import Task

from .browser_agent import BrowserAgent
from .api_agent import ApiAgent
from .doc_agent import DocAgent
from .data_agent import DataAgent
from .visualization_agent import VisualizationAgent


# Singleton registry instance
_registry: BaseRegistry | None = None


def get_default_registry() -> BaseRegistry:
    """Get or create the default agent registry."""
    global _registry
    if _registry is None:
        _registry = BaseRegistry()

        # Register default agents
        _registry.register(BrowserAgent())
        _registry.register(ApiAgent())
        _registry.register(DocAgent())
        _registry.register(DataAgent())
        _registry.register(VisualizationAgent())

    return _registry


def find_agent_for_task(task: "Task") -> BaseSubAgent | None:
    """Find the best agent for a given task."""
    registry = get_default_registry()
    agents = registry.find_agents_for_task(task)
    return agents[0] if agents else None


def list_available_capabilities() -> list[str]:
    """List all available capabilities from registered agents."""
    registry = get_default_registry()
    return [cap.value for cap in registry.get_capabilities()]
