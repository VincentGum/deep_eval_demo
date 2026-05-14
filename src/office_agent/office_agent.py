"""
Office Agent 主入口 - 工作流编排器

【模块概述】
OfficeAgent 是 Office Agent 系统的核心编排器，协调：
1. Planner - 创建任务计划
2. Executor - 并行执行任务
3. Verifier - 验证完成状态
4. HumanLoop - 管理人工介入

【PEV 工作流】
```
User Request
    │
    ▼
┌─────────────────────────────────────────┐
│           PLANNER AGENT                  │
│  分析请求 → 创建任务计划                  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│           EXECUTOR                      │
│  并行执行 → 管理依赖 → 进度跟踪           │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│           VERIFY AGENT                  │
│  验证进度 → 检查完成 → 识别缺失          │
└─────────────────────────────────────────┘
    │
    ├── 完成 → END
    │
    └── 缺失 → HUMAN-IN-THE-LOOP → 补充 → 继续
```

【工作流状态】
- IDLE: 空闲
- PLANNING: 规划中
- EXECUTING: 执行中
- VERIFYING: 验证中
- WAITING_HUMAN: 等待人工
- COMPLETED: 已完成
- FAILED: 失败
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from .base import Task, TaskPlan, TaskStatus
from .planner import PlannerAgent, MockReasoningModel
from .executor import TaskExecutor, ExecutionContext, TaskProgress
from .verify import VerifyAgent, VerifyDecision, VerificationReport
from .human_loop import HumanInTheLoop, HumanInputRequest, HumanInputResult


# ============================================================================
# 工作流状态和结果
# ============================================================================

class WorkflowState(str, Enum):
    """
    工作流执行状态枚举
    
    【状态说明】
    - IDLE: 空闲/初始状态
    - PLANNING: 正在规划任务
    - EXECUTING: 正在执行任务
    - VERIFYING: 正在验证
    - WAITING_HUMAN: 等待人工输入
    - COMPLETED: 工作流完成
    - FAILED: 工作流失败
    """
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    WAITING_HUMAN = "waiting_human"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowProgress:
    """
    工作流进度信息
    
    【字段说明】
    - state: 当前状态
    - message: 进度消息
    - plan_progress: 任务计划进度
    - current_task: 当前执行的任务
    - timestamp: 时间戳
    """
    state: WorkflowState
    message: str
    plan_progress: dict[str, Any] = field(default_factory=dict)
    current_task: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WorkflowResult:
    """
    工作流执行结果
    
    【字段说明】
    - success: 是否成功
    - plan: 关联的任务计划
    - final_state: 最终状态
    - output: 输出数据
    - human_inputs: 人工输入记录
    - errors: 错误列表
    - execution_time_seconds: 执行时间
    - completed_at: 完成时间
    """
    success: bool
    plan: TaskPlan
    final_state: WorkflowState
    output: dict[str, Any] = field(default_factory=dict)
    human_inputs: list[HumanInputResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    completed_at: datetime = field(default_factory=datetime.now)


# ============================================================================
# Office Agent
# ============================================================================

class OfficeAgent:
    """
    Office Agent 主编排器
    
    【协调流程】
    1. 规划阶段 (PLANNING)
       - Planner Agent 分析用户请求
       - 生成任务计划
    
    2. 执行阶段 (EXECUTING)
       - TaskExecutor 并行执行任务
       - SubAgents 处理具体任务
    
    3. 验证阶段 (VERIFYING)
       - Verify Agent 检查进度
       - 决定下一步操作
    
    4. 人工介入 (WAITING_HUMAN)
       - HumanLoop 请求用户输入
       - 恢复执行
    """
    
    def __init__(
        self,
        planner: PlannerAgent | None = None,
        executor: TaskExecutor | None = None,
        verifier: VerifyAgent | None = None,
        human_loop: HumanInTheLoop | None = None,
        progress_callback: Callable[[WorkflowProgress], None] | None = None,
    ):
        """
        初始化 Office Agent
        
        Args:
            planner: 任务规划器
            executor: 任务执行器
            verifier: 验证器
            human_loop: 人工介入管理器
            progress_callback: 进度回调函数
        """
        self.planner = planner or PlannerAgent()
        self.executor = executor or TaskExecutor()
        self.verifier = verifier or VerifyAgent()
        self.human_loop = human_loop or HumanInTheLoop(default_timeout=60)
        self.progress_callback = progress_callback
        
        self._current_plan: TaskPlan | None = None
        self._current_context: ExecutionContext | None = None
        self._human_inputs: list[HumanInputResult] = []
    
    def execute(
        self,
        user_request: str,
        context: dict[str, Any] | None = None
    ) -> WorkflowResult:
        """
        执行用户请求
        
        【执行流程】
        1. 规划 → 创建任务计划
        2. 执行 → 并行执行任务
        3. 验证 → 检查完成状态
        4. 循环 → 直到完成或失败
        
        Args:
            user_request: 用户请求
            context: 可选的上下文
        
        Returns:
            工作流执行结果
        """
        start_time = datetime.now()
        self._human_inputs = []
        
        # ===== 阶段 1: 规划 =====
        self._report_progress(WorkflowState.PLANNING, "Creating task plan...")
        plan, reasoning_steps = self.planner.create_plan(user_request, context)
        self._current_plan = plan
        
        # 报告计划详情
        plan_explanation = self.planner.explain_plan(plan)
        self._report_progress(
            WorkflowState.PLANNING,
            f"Task plan created with {len(plan.tasks)} tasks",
            {"explanation": plan_explanation}
        )
        
        # 初始化执行上下文
        self._current_context = ExecutionContext(plan=plan)
        
        # ===== 阶段 2: 执行 + 验证（循环）=====
        max_iterations = len(plan.tasks) * 2 + 5  # 安全限制
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # 检查是否全部完成
            if plan.all_completed():
                break
            
            # 执行待处理任务
            self._report_progress(
                WorkflowState.EXECUTING,
                f"Executing tasks (iteration {iteration})..."
            )
            
            # 执行任务（使用线程池并行）
            self._current_context = self.executor.execute_plan(
                plan,
                self._current_context
            )
            
            # ===== 阶段 3: 验证 =====
            self._report_progress(WorkflowState.VERIFYING, "Verifying progress...")
            verification = self.verifier.verify_plan_completion(plan)
            
            # 处理验证决策
            if verification.decision == VerifyDecision.COMPLETED:
                # 全部完成
                self._report_progress(
                    WorkflowState.COMPLETED,
                    "All tasks completed successfully",
                    {"confidence": verification.confidence}
                )
                break
            
            elif verification.decision == VerifyDecision.NEEDS_HUMAN_INPUT:
                # 需要人工输入
                self._report_progress(
                    WorkflowState.WAITING_HUMAN,
                    f"Waiting for human input: {verification.reason}"
                )
                
                # 找出等待输入的任务
                waiting_tasks = [
                    task for task in plan.tasks
                    if task.status == TaskStatus.WAITING_HUMAN_INPUT
                ]
                
                if waiting_tasks:
                    task = waiting_tasks[0]
                    human_result = self._request_human_input(
                        task,
                        verification.missing_info or []
                    )
                    self._human_inputs.append(human_result)
                    
                    if human_result.status.value == "received":
                        # 收到输入，恢复任务
                        task.status = TaskStatus.PENDING
                        task.input_data["human_input"] = human_result.response
                    else:
                        # 超时或取消，标记失败
                        task.status = TaskStatus.FAILED
                        task.result = None
            
            elif verification.decision == VerifyDecision.FAILED:
                # 执行失败
                self._report_progress(
                    WorkflowState.FAILED,
                    f"Workflow failed: {verification.reason}",
                    {"errors": verification.missing_info}
                )
                break
            
            else:
                # CONTINUE - 还有待处理任务
                pending = sum(1 for t in plan.tasks if t.status == TaskStatus.PENDING)
                if pending == 0:
                    # 没有待处理但未完成，可能在等待
                    continue
        
        # 最终验证
        final_verification = self.verifier.verify_plan_completion(plan)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        return WorkflowResult(
            success=final_verification.decision == VerifyDecision.COMPLETED,
            plan=plan,
            final_state=WorkflowState.COMPLETED if final_verification.decision == VerifyDecision.COMPLETED else WorkflowState.FAILED,
            output={
                "summary": self.verifier.generate_summary(plan),
                "results": {
                    task.id: {
                        "status": task.status.value,
                        "output": task.result.output if task.result else None,
                        "error": task.result.error if task.result else None,
                    }
                    for task in plan.tasks
                },
            },
            human_inputs=self._human_inputs,
            errors=[verification.reason] if final_verification.decision == VerifyDecision.FAILED else [],
            execution_time_seconds=execution_time,
        )
    
    def _report_progress(
        self,
        state: WorkflowState,
        message: str,
        extra: dict[str, Any] | None = None
    ) -> None:
        """报告工作流进度"""
        progress = WorkflowProgress(
            state=state,
            message=message,
            plan_progress=self._current_plan.get_execution_summary() if self._current_plan else {},
        )
        
        if self.progress_callback:
            self.progress_callback(progress)
    
    def _request_human_input(
        self,
        task: Task,
        missing_info: list[str]
    ) -> HumanInputResult:
        """
        请求人工输入
        
        Args:
            task: 需要输入的任务
            missing_info: 缺失的信息列表
        
        Returns:
            输入结果
        """
        # 构建问题
        question = f"Task '{task.description}' needs more information.\n"
        if missing_info:
            question += "缺失: " + ", ".join(missing_info[:3])
        question += "\n请提供所需信息。"
        
        # 创建输入请求
        request = HumanInputRequest(
            id=f"input_{task.id}",
            task_id=task.id,
            question=question,
            context={"task_description": task.description},
            timeout_seconds=task.timeout_seconds,
        )
        
        # 生成提示
        prompt = self.human_loop.generate_prompt(request)
        print("\n" + prompt)
        
        # 请求输入
        return self.human_loop.request_input(
            task_id=task.id,
            question=question,
            context={"task_description": task.description},
            timeout_seconds=task.timeout_seconds,
        )
    
    def get_current_plan(self) -> TaskPlan | None:
        """获取当前任务计划"""
        return self._current_plan
    
    def get_verification_summary(self) -> str | None:
        """获取验证摘要"""
        if self._current_plan:
            return self.verifier.generate_summary(self._current_plan)
        return None


# ============================================================================
# 便捷函数
# ============================================================================

def create_office_agent(
    progress_callback: Callable[[WorkflowProgress], None] | None = None
) -> OfficeAgent:
    """
    创建 Office Agent 实例
    
    Args:
        progress_callback: 进度回调函数
    
    Returns:
        OfficeAgent 实例
    """
    return OfficeAgent(progress_callback=progress_callback)


def run_office_task(
    user_request: str,
    verbose: bool = True
) -> WorkflowResult:
    """
    运行办公任务
    
    Args:
        user_request: 用户请求
        verbose: 是否输出详细信息
    
    Returns:
        工作流结果
    """
    def progress_handler(progress: WorkflowProgress):
        if verbose:
            print(f"[{progress.state.value.upper()}] {progress.message}")
    
    agent = create_office_agent(progress_callback=progress_handler)
    return agent.execute(user_request)


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    "WorkflowState",
    "WorkflowProgress",
    "WorkflowResult",
    "OfficeAgent",
    "create_office_agent",
    "run_office_task",
]
