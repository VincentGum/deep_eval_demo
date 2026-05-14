"""
Agent 注册表 - 管理所有子 Agent 的注册和发现

【功能说明】
Registry 用于：
1. 子 Agent 注册 - 收集所有可用的子 Agent
2. Agent 查找 - 根据能力类型找到合适的 Agent
3. 能力查询 - 列出所有支持的能力
4. Agent 列表 - 获取所有已注册的 Agent

【使用方式】
```python
from .registry import get_registry

# 获取注册表单例
registry = get_registry()

# 根据能力查找 Agent
agent = registry.find_agent_by_capability(AgentCapability.API_CALL)

# 获取所有支持特定能力的 Agent
agents = registry.find_agents_by_capability(AgentCapability.DATA_STATISTICS)
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import BaseSubAgent, AgentCapability

# 避免循环导入
if TYPE_CHECKING:
    from ..base import ExecutionResult, Task


# ============================================================================
# Agent 注册表
# ============================================================================

class Registry:
    """
    Agent 注册表 - 管理所有子 Agent
    
    【设计模式】
    使用单例模式 + 注册表模式：
    - get_registry(): 获取全局单例
    - register(): 注册 Agent 实例
    - find_agent_by_capability(): 按能力查找
    
    【自动注册】
    Registry 初始化时会自动注册所有已实现的 Agent：
    - BrowserAgent: 浏览器操作
    - ApiAgent: API 调用
    - DocAgent: 文档操作
    - DataAgent: 数据处理
    - VisualizationAgent: 可视化生成
    """
    
    # 类级别的单例实例
    _instance: "Registry | None" = None
    
    def __init__(self):
        # Agent 实例字典: {agent_name: agent_instance}
        self._agents: dict[str, BaseSubAgent] = {}
        
        # 能力索引: {capability: [agent_names]}
        self._capability_index: dict[AgentCapability, list[str]] = {}
        
        # 防止直接实例化（应使用 get_registry()）
        if Registry._instance is not None:
            raise RuntimeError("Use get_registry() to get the singleton instance")
        
        # 自动注册所有 Agent
        self._auto_register()
    
    def _auto_register(self):
        """
        自动注册所有已实现的 Agent
        
        【注册顺序】
        1. BrowserAgent
        2. ApiAgent
        3. DocAgent
        4. DataAgent
        5. VisualizationAgent
        
        【扩展方式】
        在此方法中添加新的 Agent 注册即可：
        ```python
        from .browser_agent import BrowserAgent
        self.register(BrowserAgent())
        ```
        """
        # 延迟导入避免循环依赖
        from .browser_agent import BrowserAgent
        from .api_agent import ApiAgent
        from .doc_agent import DocAgent
        from .data_agent import DataAgent
        from .visualization_agent import VisualizationAgent
        
        # 按优先级注册 Agent
        agents_to_register = [
            BrowserAgent(),
            ApiAgent(),
            DocAgent(),
            DataAgent(),
            VisualizationAgent(),
        ]
        
        for agent in agents_to_register:
            self.register(agent)
    
    def register(self, agent: BaseSubAgent) -> None:
        """
        注册一个 Agent 实例
        
        Args:
            agent: BaseSubAgent 的子类实例
        
        【示例】
        ```python
        registry.register(MyCustomAgent())
        ```
        """
        name = agent.name
        self._agents[name] = agent
        
        # 更新能力索引
        for capability in agent.capabilities:
            if capability not in self._capability_index:
                self._capability_index[capability] = []
            if name not in self._capability_index[capability]:
                self._capability_index[capability].append(name)
    
    def unregister(self, agent_name: str) -> bool:
        """
        取消注册一个 Agent
        
        Args:
            agent_name: Agent 名称
        
        Returns:
            是否成功取消注册
        """
        if agent_name not in self._agents:
            return False
        
        agent = self._agents.pop(agent_name)
        
        # 从能力索引中移除
        for capability in agent.capabilities:
            if capability in self._capability_index:
                if agent_name in self._capability_index[capability]:
                    self._capability_index[capability].remove(agent_name)
        
        return True
    
    def get_agent(self, agent_name: str) -> BaseSubAgent | None:
        """
        根据名称获取 Agent
        
        Args:
            agent_name: Agent 名称
        
        Returns:
            Agent 实例，如果不存在返回 None
        """
        return self._agents.get(agent_name)
    
    def find_agent_by_capability(
        self,
        capability: AgentCapability,
    ) -> BaseSubAgent | None:
        """
        根据能力查找第一个可用的 Agent
        
        Args:
            capability: 需要的能力
        
        Returns:
            第一个支持该能力的 Agent 实例
        """
        agent_names = self._capability_index.get(capability, [])
        if not agent_names:
            return None
        return self._agents.get(agent_names[0])
    
    def find_agents_by_capability(
        self,
        capability: AgentCapability,
    ) -> list[BaseSubAgent]:
        """
        根据能力查找所有可用的 Agent
        
        Args:
            capability: 需要的能力
        
        Returns:
            所有支持该能力的 Agent 列表
        """
        agent_names = self._capability_index.get(capability, [])
        return [self._agents[name] for name in agent_names if name in self._agents]
    
    def list_all_agents(self) -> list[dict]:
        """
        列出所有已注册的 Agent
        
        Returns:
            Agent 信息列表，每个元素包含：
            - name: Agent 名称
            - capabilities: 支持的能力列表
        """
        return [
            {
                "name": agent.name,
                "capabilities": [str(cap) for cap in agent.capabilities],
            }
            for agent in self._agents.values()
        ]
    
    def list_all_capabilities(self) -> list[str]:
        """
        列出所有已注册的能力
        
        Returns:
            能力名称列表
        """
        return [str(cap) for cap in self._capability_index.keys()]
    
    def __repr__(self) -> str:
        return f"Registry(agents={len(self._agents)}, capabilities={len(self._capability_index)})"


# ============================================================================
# 单例访问函数
# ============================================================================

def get_registry() -> Registry:
    """
    获取 Registry 单例
    
    Returns:
        全局唯一的 Registry 实例
    
    【使用示例】
    ```python
    from .registry import get_registry
    
    registry = get_registry()
    agent = registry.find_agent_by_capability(AgentCapability.API_CALL)
    ```
    """
    if Registry._instance is None:
        Registry._instance = Registry()
    return Registry._instance


# ============================================================================
# 导出
# ============================================================================

__all__ = ["Registry", "get_registry"]
