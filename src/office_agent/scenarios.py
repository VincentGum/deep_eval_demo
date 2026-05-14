"""办公场景模块 - 预配置的办公自动化场景示例。

本模块提供预定义的办公场景，展示 Office Agent 的各种能力：
- 周报生成（销售数据分析 + 图表 + 文档）
- 客户调研（数据查询 + 网页浏览 + 竞品分析）
- 会议准备（日程 + 参会人 + 资料整理）

每个场景都是一个完整的端到端工作流，演示了 PEV 架构中
Planner、Executor、Verify 和 Human-in-the-Loop 的协作。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import Task, TaskPlan, TaskPriority, AgentCapability
from .planner import PlannerAgent, MockReasoningModel
from .office_agent import OfficeAgent, WorkflowResult


@dataclass
class OfficeScenario:
    """
    预配置的办公自动化场景。
    
    属性说明：
    - name: 场景名称（用于命令行选择）
    - description: 场景描述（用于帮助信息）
    - user_request: 用户请求文本（会发送给 Agent）
    - expected_tasks: 预期任务列表（用于验证）
    - context: 额外上下文信息（传递给 Agent）
    """
    name: str  # 场景名称
    description: str  # 场景描述
    user_request: str  # 用户请求
    expected_tasks: list[str]  # 预期任务列表
    context: dict[str, Any]  # 额外上下文


# ============================================================================
# 场景1：周销售报告生成
# ============================================================================
# 
# 演示场景：自动生成包含数据分析、图表和文档的周报
# 
# 涉及能力：
# - API Agent: 获取销售数据
# - Data Agent: 数据统计（汇总、平均值）
# - Visualization Agent: 生成销售图表
# - Doc Agent: 生成报告文档
#
# 工作流：API → Data → Visualization → Doc

WEEKLY_SALES_REPORT = OfficeScenario(
    name="Weekly Sales Report",  # 场景名称
    description="生成包含图表和统计数据的综合周销售报告。",  # 描述
    user_request=(
        "Generate a weekly sales report for this week. "  # 用户请求
        "I need to fetch sales data from the API, calculate totals and averages, "  # 获取销售数据
        "create a bar chart showing sales by product, and save everything to a report document."  # 生成图表和文档
    ),
    expected_tasks=[  # 预期任务流程
        "Fetch sales data from API",  # 1. 从 API 获取数据
        "Aggregate data (sum, average)",  # 2. 数据统计
        "Create sales chart",  # 3. 生成图表
        "Generate report document",  # 4. 生成文档
    ],
    context={  # 额外上下文
        "department": "Sales",  # 部门：销售部
        "period": "this_week",  # 周期：本周
        "output_format": "markdown",  # 输出格式
    },
)


# ============================================================================
# 场景2：客户调研报告
# ============================================================================
#
# 演示场景：综合客户信息收集和竞品分析
#
# 涉及能力：
# - API Agent: 查询客户数据库
# - Browser Agent: 访问客户网站
# - Data Agent: 数据整理
# - Doc Agent: 生成调研报告
#
# 工作流：API(Browser) → Data → Doc

CUSTOMER_RESEARCH = OfficeScenario(
    name="Customer Research Report",  # 场景名称
    description="调研客户数据、爬取竞品网站并整理发现。",  # 描述
    user_request=(
        "I need to research our top 5 customers. "  # 用户请求
        "Please look up their information from our database, "  # 查询数据库
        "visit their websites to understand their business, "  # 访问网站
        "and create a comprehensive research report."  # 生成报告
    ),
    expected_tasks=[  # 预期任务流程
        "Query customer data from database",  # 1. 查询客户数据
        "Browse customer websites",  # 2. 浏览客户网站
        "Scrape competitor information",  # 3. 爬取竞品信息
        "Compile research report",  # 4. 整理报告
    ],
    context={  # 额外上下文
        "customer_count": 5,  # 客户数量
        "include_competitors": True,  # 包含竞品
        "output_format": "document",  # 输出格式
    },
)


# ============================================================================
# 场景3：会议准备包
# ============================================================================
#
# 演示场景：自动整理会议所需的各种材料
#
# 涉及能力：
# - API Agent: 查询会议信息、获取数据
# - Data Agent: 数据统计
# - Doc Agent: 格式化文档
#
# 工作流：API → Data → Doc

MEETING_PREPARATION = OfficeScenario(
    name="Meeting Preparation",  # 场景名称
    description="准备包含议程、参会人和相关数据的会议材料包。",  # 描述
    user_request=(
        "Prepare a meeting pack for my quarterly review meeting. "  # 用户请求
        "I need the agenda, attendee list, previous meeting notes, "  # 议程和参会人
        "current project status from our tracking system, "  # 项目状态
        "and some key metrics. Format everything in a nice document."  # 关键指标
    ),
    expected_tasks=[  # 预期任务流程
        "Fetch meeting details",  # 1. 获取会议详情
        "Get attendee information",  # 2. 获取参会人信息
        "Retrieve previous meeting notes",  # 3. 获取上次会议记录
        "Query project status",  # 4. 查询项目状态
        "Fetch key metrics",  # 5. 获取关键指标
        "Generate formatted document",  # 6. 生成格式化文档
    ],
    context={  # 额外上下文
        "meeting_type": "quarterly_review",  # 会议类型：季度评审
        "include_calendar": True,  # 包含日程
        "output_format": "document",  # 输出格式
    },
)


# ============================================================================
# 场景工厂函数
# ============================================================================

def get_scenario(scenario_name: str) -> OfficeScenario | None:
    """
    根据名称获取场景实例。
    
    Args:
        scenario_name: 场景名称（支持多种格式：
                      "weekly_sales_report", "weekly-sales-report", 
                      "Weekly Sales Report" 等）
    
    Returns:
        OfficeScenario 对象，如果未找到则返回 None
    """
    # 场景注册表
    scenarios = {
        "weekly_sales_report": WEEKLY_SALES_REPORT,
        "customer_research": CUSTOMER_RESEARCH,
        "meeting_preparation": MEETING_PREPARATION,
    }
    # 统一格式：转小写 + 替换空格为下划线
    key = scenario_name.lower().replace(" ", "_")
    return scenarios.get(key)


def list_scenarios() -> list[OfficeScenario]:
    """
    列出所有可用的场景。
    
    Returns:
        OfficeScenario 对象列表
    """
    return [WEEKLY_SALES_REPORT, CUSTOMER_RESEARCH, MEETING_PREPARATION]


def run_scenario(
    scenario: OfficeScenario,
    verbose: bool = True
) -> WorkflowResult:
    """
    运行指定的场景并返回执行结果。
    
    这是场景执行的主入口函数，流程如下：
    1. 打印场景信息（verbose 模式）
    2. 创建进度回调函数
    3. 初始化 OfficeAgent
    4. 执行用户请求
    5. 打印执行结果摘要
    
    Args:
        scenario: 要执行的场景
        verbose: 是否打印详细进度信息
    
    Returns:
        WorkflowResult: 包含执行结果的 WorkflowResult 对象
                        - success: 是否成功
                        - execution_time_seconds: 执行耗时
                        - plan: 任务计划
                        - output: 输出结果
                        - errors: 错误列表（如果有）
    """
    if verbose:
        # 打印场景信息
        print("=" * 60)
        print(f"SCENARIO: {scenario.name}")
        print("=" * 60)
        print(f"Description: {scenario.description}")
        print(f"Request: {scenario.user_request}")
        print("=" * 60)
        print()

    # 进度回调函数：用于接收 Agent 执行过程中的状态更新
    def progress_callback(progress):
        if verbose:
            # 打印当前状态和消息
            print(f"[{progress.state.value.upper()}] {progress.message}")

    # 创建 Agent 并执行
    agent = OfficeAgent(progress_callback=progress_callback)
    result = agent.execute(scenario.user_request, scenario.context)

    if verbose:
        # 打印执行结果摘要
        print()
        print("=" * 60)
        print("RESULT SUMMARY")
        print("=" * 60)
        print(f"Success: {result.success}")  # 是否成功
        print(f"Execution Time: {result.execution_time_seconds:.2f}s")  # 执行耗时
        
        # 统计完成的任务数
        completed = sum(1 for t in result.plan.tasks if t.status.value == 'completed')
        print(f"Tasks Completed: {completed}/{len(result.plan.tasks)}")

        # 打印需要人工输入的数量
        if result.human_inputs:
            print(f"Human Inputs: {len(result.human_inputs)}")

        # 打印错误信息
        if result.errors:
            print(f"Errors: {result.errors}")

        print()
        print(result.output.get("summary", ""))  # 打印摘要

    return result


# ============================================================================
# 演示函数
# ============================================================================
# 
# 这些函数用于快速运行单个或所有演示场景
# 在 examples/run_office_agent.py 中被调用

def demo_weekly_sales_report():
    """演示：周销售报告生成"""
    print("\n" + "=" * 60)
    print("DEMO: Weekly Sales Report Generation")
    print("=" * 60)
    return run_scenario(WEEKLY_SALES_REPORT, verbose=True)


def demo_customer_research():
    """演示：客户调研报告"""
    print("\n" + "=" * 60)
    print("DEMO: Customer Research Report")
    print("=" * 60)
    return run_scenario(CUSTOMER_RESEARCH, verbose=True)


def demo_meeting_preparation():
    """演示：会议准备包"""
    print("\n" + "=" * 60)
    print("DEMO: Meeting Preparation Pack")
    print("=" * 60)
    return run_scenario(MEETING_PREPARATION, verbose=True)


def demo_all():
    """
    运行所有演示场景。
    
    依次执行 WEEKLY_SALES_REPORT, CUSTOMER_RESEARCH, MEETING_PREPARATION
    并打印每个场景的执行结果摘要。
    
    Returns:
        list[tuple[str, bool]]: 每个场景的名称和成功状态
    """
    print("\n" + "=" * 60)
    print("RUNNING ALL OFFICE AGENT DEMOS")
    print("=" * 60)

    results = []

    # 执行所有场景
    for scenario in list_scenarios():
        result = run_scenario(scenario, verbose=False)
        results.append((scenario.name, result.success))

    # 打印汇总
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"  [{status}] {name}")

    return results
