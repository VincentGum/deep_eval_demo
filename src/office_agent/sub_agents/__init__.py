"""Sub Agents - Specialized agents for different capabilities.

Each sub-agent is responsible for executing specific types of tasks
based on its capabilities.
"""

from .browser_agent import BrowserAgent
from .api_agent import ApiAgent
from .doc_agent import DocAgent
from .data_agent import DataAgent
from .visualization_agent import VisualizationAgent
from .registry import get_default_registry

__all__ = [
    "BrowserAgent",
    "ApiAgent",
    "DocAgent",
    "DataAgent",
    "VisualizationAgent",
    "get_default_registry",
]
