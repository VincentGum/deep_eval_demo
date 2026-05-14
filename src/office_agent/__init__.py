"""Office Agent 模块 - AI 办公助手（PEV 架构）

本模块提供基于 PEV (Plan-Execute-Verify) 架构的 AI 办公助手，
支持并行任务执行和 Human-in-the-Loop 机制。

核心组件：
- PlannerAgent: 任务规划（分析用户请求，生成任务计划）
- TaskExecutor: 任务执行（并行调度多个子 Agent）
- VerifyAgent: 结果验证（对比实际结果与预期）
- HumanInTheLoop: 人工介入（复杂决策需要人工确认）

子 Agent 能力：
- Browser Agent: 网页浏览与数据爬取
- API Agent: 外部 API 调用
- Data Agent: 数据处理与统计分析
- Doc Agent: 文档读写
- Visualization Agent: 图表生成

使用示例：
    from office_agent import OfficeAgent, WEEKLY_SALES_REPORT
    
    agent = OfficeAgent()
    result = agent.execute(
        user_request="生成周销售报告",
        context={"period": "this_week"}
    )
"""

# ============================================================================
# 基础类型导出
# ============================================================================
# 
# 这些是 PEV 架构的核心数据结构

from .base import (
    Task,  # 单个任务定义
    TaskStatus,  # 任务状态枚举
    TaskResult,  # 任务执行结果
    TaskPlan,  # 任务计划（包含多个任务）
    AgentCapability,  # Agent 能力枚举
    BaseSubAgent,  # 子 Agent 基类
    ExecutionResult,  # 执行结果
)

# ============================================================================
# 核心 Agent 导出
# ============================================================================
# 
# PEV 架构的三个主要节点

from .planner import PlannerAgent  # Planner Agent（任务规划）
from .executor import TaskExecutor  # Task Executor（任务执行）
from .verify import VerifyAgent  # Verify Agent（结果验证）
from .human_loop import HumanInTheLoop  # Human-in-the-Loop

# ============================================================================
# 主入口和图构建导出
# ============================================================================

from .office_agent import (
    OfficeAgent,  # Office Agent 主类
    WorkflowState,  # 工作流状态
    WorkflowProgress,  # 工作流进度
    WorkflowResult,  # 工作流结果
    invoke_office_agent,  # 便捷调用函数
    build_office_agent_graph,  # 构建 LangGraph
)

# ============================================================================
# 预定义场景导出
# ============================================================================
# 
# 提供开箱即用的办公自动化场景

from .scenarios import (
    OfficeScenario,  # 场景定义类
    WEEKLY_SALES_REPORT,  # 周报生成场景
    CUSTOMER_RESEARCH,  # 客户调研场景
    MEETING_PREPARATION,  # 会议准备场景
    list_scenarios,  # 列出所有场景
    get_scenario,  # 获取指定场景
    run_scenario,  # 运行场景
    demo_all,  # 运行所有演示
)

# ============================================================================
# 模块公开 API
# ============================================================================

__all__ = [
    # ========== 基础类 ==========
    "Task",  # 单个任务
    "TaskStatus",  # 任务状态
    "TaskResult",  # 任务结果
    "TaskPlan",  # 任务计划
    "AgentCapability",  # Agent 能力
    "BaseSubAgent",  # 子 Agent 基类
    "ExecutionResult",  # 执行结果
    
    # ========== 核心 Agent ==========
    "PlannerAgent",  # 任务规划 Agent
    "TaskExecutor",  # 任务执行器
    "VerifyAgent",  # 结果验证 Agent
    "HumanInTheLoop",  # 人工介入机制
    "OfficeAgent",  # Office Agent 主类
    
    # ========== 工作流类型 ==========
    "WorkflowState",  # 工作流状态
    "WorkflowProgress",  # 工作流进度
    "WorkflowResult",  # 工作流结果
    
    # ========== 入口函数 ==========
    "invoke_office_agent",  # 便捷调用
    "build_office_agent_graph",  # 图构建
    
    # ========== 预定义场景 ==========
    "OfficeScenario",  # 场景类
    "WEEKLY_SALES_REPORT",  # 周报场景
    "CUSTOMER_RESEARCH",  # 客户调研场景
    "MEETING_PREPARATION",  # 会议准备场景
    "list_scenarios",  # 列出场景
    "get_scenario",  # 获取场景
    "run_scenario",  # 运行场景
    "demo_all",  # 演示所有场景
]
