"""
Office Agent 子模块 - 包含所有子 Agent 实现

【模块结构】
```
office_agent/
└── sub_agents/
    ├── __init__.py         # 本文件 - 统一导出
    ├── registry.py         # Agent 注册表
    ├── browser_agent.py    # 浏览器 Agent
    ├── api_agent.py        # API Agent
    ├── doc_agent.py        # 文档 Agent
    ├── data_agent.py       # 数据 Agent
    └── visualization_agent.py  # 可视化 Agent
```

【子 Agent 列表】
| Agent | 能力 | 用途 |
|-------|------|------|
| BrowserAgent | BROWSER_* | 网页浏览、内容爬取 |
| ApiAgent | API_CALL | 外部 API 调用 |
| DocAgent | DOC_* | 文档读写、格式转换 |
| DataAgent | DATA_* | 数据处理、统计分析 |
| VisualizationAgent | VISUALIZATION_* | 图表生成 |

【使用方式】
```python
from .registry import get_registry

# 获取注册表
registry = get_registry()

# 按能力查找 Agent
agent = registry.find_agent_by_capability(AgentCapability.API_CALL)

# 执行任务
result = agent.execute(task)
```

【扩展新的 Agent】
1. 在单独的文件中实现 Agent 类（继承 BaseSubAgent）
2. 在 registry.py 的 _auto_register() 中注册
3. 在 base.py 的 AgentCapability 中添加新的能力枚举
"""

from __future__ import annotations

# 统一导出所有子 Agent 和相关组件
from .registry import Registry, get_registry
from .browser_agent import BrowserAgent, MockBrowser
from .api_agent import ApiAgent, MockAPIClient
from .doc_agent import DocAgent, MockFileSystem
from .data_agent import DataAgent, MockDataProcessor
from .visualization_agent import VisualizationAgent, MockChartGenerator


# ============================================================================
# 快捷导入函数
# ============================================================================

def get_all_agents():
    """
    获取所有已注册的 Agent 列表
    
    Returns:
        Agent 实例列表
    """
    registry = get_registry()
    return list(registry._agents.values())


def get_agent_by_name(name: str):
    """
    根据名称获取 Agent
    
    Args:
        name: Agent 名称
    
    Returns:
        Agent 实例，如果不存在返回 None
    """
    registry = get_registry()
    return registry.get_agent(name)


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # 注册表
    "Registry",
    "get_registry",
    # Agent 类
    "BrowserAgent",
    "ApiAgent",
    "DocAgent",
    "DataAgent",
    "VisualizationAgent",
    # Mock 组件（用于测试）
    "MockBrowser",
    "MockAPIClient",
    "MockFileSystem",
    "MockDataProcessor",
    "MockChartGenerator",
    # 快捷函数
    "get_all_agents",
    "get_agent_by_name",
]
