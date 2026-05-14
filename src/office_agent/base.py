"""
Office Agent 基础模块 - 定义共享的数据结构和基类

【模块概述】
本模块定义了 Office Agent 系统的基础组件：
1. Task, TaskPlan, TaskStatus - 任务相关的数据类
2. AgentCapability - Agent 能力枚举
3. BaseSubAgent - 子 Agent 的抽象基类
4. AgentRegistry - Agent 注册表

【设计原则】
- 使用 dataclass 简化数据类定义
- 使用 Enum 提供类型安全的枚举
- 使用 ABC 实现抽象基类
- 单例模式管理 Agent 注册表
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar, Generic


# ============================================================================
# 任务状态枚举
# ============================================================================

class TaskStatus(str, Enum):
    """
    任务执行状态枚举
    
    【状态说明】
    - PENDING: 等待执行
    - RUNNING: 正在执行
    - COMPLETED: 执行成功
    - FAILED: 执行失败
    - WAITING_HUMAN_INPUT: 等待人工输入
    - SKIPPED: 跳过（依赖任务失败）
    """
    PENDING = "pending"                           # 等待执行
    RUNNING = "running"                           # 执行中
    COMPLETED = "completed"                       # 已完成
    FAILED = "failed"                             # 失败
    WAITING_HUMAN_INPUT = "waiting_human_input"   # 等待人工输入
    SKIPPED = "skipped"                           # 已跳过


class TaskPriority(str, Enum):
    """
    任务优先级枚举
    
    【优先级说明】
    - LOW: 低优先级，可延后
    - MEDIUM: 中优先级，正常处理
    - HIGH: 高优先级，尽快处理
    - URGENT: 紧急，需要立即处理
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


# ============================================================================
# 任务相关数据类
# ============================================================================

@dataclass
class Task:
    """
    任务数据类 - 表示一个可执行的工作单元
    
    【字段说明】
    - id: 任务唯一标识
    - description: 任务描述
    - capability_required: 执行所需的能力
    - input_data: 输入数据
    - expected_output: 期望输出（用于验证）
    - status: 当前状态
    - priority: 优先级
    - depends_on: 依赖的任务 ID 列表
    - timeout_seconds: 超时时间（秒）
    - created_at: 创建时间
    - started_at: 开始执行时间
    - completed_at: 完成时间
    - result: 执行结果
    """
    # 基本信息
    id: str                                      # 任务 ID
    description: str                             # 任务描述
    capability_required: str                     # 所需能力
    input_data: dict[str, Any] = field(default_factory=dict)  # 输入参数
    expected_output: str | None = None           # 期望输出
    
    # 执行控制
    status: TaskStatus = TaskStatus.PENDING     # 当前状态
    priority: TaskPriority = TaskPriority.MEDIUM # 优先级
    depends_on: list[str] = field(default_factory=list)  # 依赖任务
    timeout_seconds: int = 300                  # 超时时间（秒）
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    
    # 执行结果
    output: Any = None
    result: "TaskResult | None" = None
    
    def can_execute(self, completed_tasks: set[str]) -> bool:
        """
        检查任务是否可以执行
        
        条件：所有依赖任务都已完成
        
        Args:
            completed_tasks: 已完成的任务 ID 集合
        
        Returns:
            是否可以执行
        """
        # 检查依赖
        for dep_id in self.depends_on:
            if dep_id not in completed_tasks:
                return False
        return True


@dataclass
class TaskResult:
    """
    任务执行结果
    
    【字段说明】
    - task_id: 关联的任务 ID
    - status: 执行状态
    - output: 输出数据
    - error: 错误信息（如果有）
    - evidence: 执行证据/日志
    - completed_at: 完成时间
    """
    task_id: str
    status: TaskStatus
    output: Any = None
    error: str | None = None
    evidence: dict[str, Any] | None = None
    completed_at: datetime = field(default_factory=datetime.now)
    
    def is_successful(self) -> bool:
        """判断是否成功执行"""
        return self.status == TaskStatus.COMPLETED


@dataclass
class TaskPlan:
    """
    任务计划 - 包含一组需要协调执行的任务
    
    【字段说明】
    - id: 计划 ID
    - user_request: 原始用户请求
    - tasks: 任务列表
    - context: 额外上下文
    """
    id: str
    user_request: str
    tasks: list[Task] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    
    def get_task(self, task_id: str) -> Task | None:
        """根据 ID 获取任务"""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def all_completed(self) -> bool:
        """检查所有任务是否完成"""
        if not self.tasks:
            return False
        return all(t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED) for t in self.tasks)
    
    def get_execution_summary(self) -> dict[str, Any]:
        """获取执行摘要"""
        return {
            "total": len(self.tasks),
            "completed": sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in self.tasks if t.status == TaskStatus.FAILED),
            "pending": sum(1 for t in self.tasks if t.status == TaskStatus.PENDING),
            "running": sum(1 for t in self.tasks if t.status == TaskStatus.RUNNING),
        }


# ============================================================================
# Agent 能力枚举
# ============================================================================

class AgentCapability(str, Enum):
    """
    Agent 能力枚举 - 定义子 Agent 可以执行的能力类型
    
    【能力分类】
    - 浏览器能力: 浏览、爬取、填表
    - API 能力: 调用外部 API
    - 数据能力: 查询、转换、聚合、导出
    - 文档能力: 读、写、解析、转换
    - 可视化能力: 创建图表、表格、报告
    - 通信能力: 发送邮件
    """
    # 浏览器能力
    BROWSER_NAVIGATE = "browser_navigate"     # 导航到 URL
    BROWSER_SCRAPE = "browser_scrape"        # 爬取网页内容
    BROWSER_FILL_FORM = "browser_fill_form"  # 填写表单
    
    # API 能力
    API_CALL = "api_call"                    # 调用 API
    
    # 数据能力
    DATA_QUERY = "data_query"                # 查询数据
    DATA_TRANSFORM = "data_transform"        # 转换数据
    DATA_AGGREGATE = "data_aggregate"        # 聚合数据
    DATA_EXPORT = "data_export"              # 导出数据
    
    # 文档能力
    DOC_READ = "doc_read"                   # 读取文档
    DOC_WRITE = "doc_write"                  # 写入文档
    DOC_PARSE = "doc_parse"                  # 解析文档
    DOC_CONVERT = "doc_convert"              # 转换文档格式
    
    # 可视化能力
    CHART_CREATE = "chart_create"            # 创建图表
    TABLE_CREATE = "table_create"            # 创建表格
    REPORT_GENERATE = "report_generate"      # 生成报告
    
    # 通信能力
    EMAIL_SEND = "email_send"                # 发送邮件


# ============================================================================
# 执行结果
# ============================================================================

@dataclass
class ExecutionResult:
    """
    执行结果 - 子 Agent 的返回值格式
    
    【字段说明】
    - success: 是否成功
    - output: 输出数据
    - error: 错误信息（如果有）
    - metadata: 额外元数据
    """
    success: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] | None = None


# ============================================================================
# 子 Agent 基类
# ============================================================================

class BaseSubAgent(ABC):
    """
    子 Agent 抽象基类
    
    所有子 Agent（BrowserAgent, ApiAgent, DocAgent 等）都必须继承此类
    并实现 can_handle() 和 execute() 方法。
    
    【设计模式】
    - 策略模式：不同 Agent 实现不同的执行策略
    - 模板方法：can_handle() 检查能力，execute() 执行逻辑
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Agent 名称
        
        Returns:
            Agent 的唯一名称标识
        """
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> list[AgentCapability]:
        """
        Agent 支持的能力列表
        
        Returns:
            该 Agent 支持的所有能力
        """
        pass
    
    @abstractmethod
    def can_handle(self, task: Task) -> bool:
        """
        检查 Agent 是否能处理给定任务
        
        Args:
            task: 待处理的任务
        
        Returns:
            是否能处理
        """
        pass
    
    @abstractmethod
    def execute(self, task: Task) -> ExecutionResult:
        """
        执行任务
        
        Args:
            task: 要执行的任务
        
        Returns:
            执行结果
        """
        pass


# ============================================================================
# Agent 注册表
# ============================================================================

class AgentRegistry:
    """
    Agent 注册表 - 管理所有子 Agent 的注册和查询
    
    【功能】
    - 注册子 Agent
    - 根据能力查找 Agent
    - 获取所有可用能力
    
    【使用方式】
    registry = AgentRegistry()
    registry.register(BrowserAgent())
    agents = registry.find_agents_for_task(task)
    """
    
    def __init__(self):
        self._agents: list[BaseSubAgent] = []
        self._capability_map: dict[AgentCapability, list[BaseSubAgent]] = {}
    
    def register(self, agent: BaseSubAgent) -> None:
        """
        注册一个子 Agent
        
        Args:
            agent: 要注册的 Agent
        """
        self._agents.append(agent)
        
        # 更新能力映射
        for cap in agent.capabilities:
            if cap not in self._capability_map:
                self._capability_map[cap] = []
            self._capability_map[cap].append(agent)
    
    def find_agents_for_task(self, task: Task) -> list[BaseSubAgent]:
        """
        查找能处理任务的 Agent
        
        Args:
            task: 目标任务
        
        Returns:
            能处理该任务的 Agent 列表
        """
        # 解析能力
        capability = task.capability_required
        if isinstance(capability, str):
            capability = AgentCapability(capability)
        
        return self._capability_map.get(capability, [])
    
    def get_capabilities(self) -> list[AgentCapability]:
        """获取所有已注册的能力"""
        return list(self._capability_map.keys())
    
    def list_agents(self) -> list[BaseSubAgent]:
        """列出所有已注册的 Agent"""
        return self._agents.copy()


# ============================================================================
# 辅助函数
# ============================================================================

def create_task_id(prefix: str = "task", counter: int = 1) -> str:
    """
    创建任务 ID
    
    Args:
        prefix: ID 前缀
        counter: 计数器
    
    Returns:
        格式化的任务 ID，如 "task_001"
    """
    return f"{prefix}_{counter:03d}"


def create_task(
    description: str,
    capability: AgentCapability | str,
    input_data: dict[str, Any] | None = None,
    expected_output: str | None = None,
    priority: TaskPriority = TaskPriority.MEDIUM,
    depends_on: list[str] | None = None,
) -> Task:
    """
    创建任务的便捷函数
    
    Args:
        description: 任务描述
        capability: 所需能力
        input_data: 输入数据
        expected_output: 期望输出
        priority: 优先级
        depends_on: 依赖任务
    
    Returns:
        创建的 Task 对象
    """
    capability_str = capability.value if isinstance(capability, AgentCapability) else capability
    
    return Task(
        id=create_task_id(),
        description=description,
        capability_required=capability_str,
        input_data=input_data or {},
        expected_output=expected_output,
        priority=priority,
        depends_on=depends_on or [],
    )


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # 枚举
    "TaskStatus",
    "TaskPriority",
    "AgentCapability",
    # 数据类
    "Task",
    "TaskResult",
    "TaskPlan",
    "ExecutionResult",
    # 基类
    "BaseSubAgent",
    "AgentRegistry",
    # 辅助函数
    "create_task_id",
    "create_task",
]
