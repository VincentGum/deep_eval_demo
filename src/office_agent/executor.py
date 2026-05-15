"""
任务执行器 - 负责并行执行任务并管理执行上下文

【模块概述】
TaskExecutor 是 Office Agent 系统的核心执行组件，负责：
1. 并行执行 - 使用线程池并发执行多个任务
2. 依赖管理 - 确保任务按正确的依赖顺序执行
3. 进度跟踪 - 实时报告任务执行进度
4. 数据共享 - 在任务之间传递数据

【核心组件】

1. TaskProgress (任务进度)
   - 跟踪单个任务的执行进度
   - 记录开始/完成时间
   - 记录状态和消息

2. ExecutionContext (执行上下文)
   - 管理所有任务的执行状态
   - 存储已完成任务的结果
   - 维护共享数据供后续任务使用

3. TaskExecutor (任务执行器)
   - 使用 ThreadPoolExecutor 实现并行
   - 支持任务依赖解析
   - 回调机制报告进度
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .base import (
    Task,
    TaskStatus,
    TaskResult,
    TaskPlan,
    BaseSubAgent,
    ExecutionResult,
)
from .sub_agents.registry import Registry, get_registry


# ============================================================================
# 执行进度和上下文
# ============================================================================

@dataclass
class TaskProgress:
    """
    任务执行进度信息
    
    【字段说明】
    - task_id: 任务 ID
    - status: 当前状态
    - started_at: 开始时间
    - completed_at: 完成时间
    - progress_percent: 完成百分比 (0.0-100.0)
    - message: 进度消息
    """
    task_id: str
    status: TaskStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress_percent: float = 0.0
    message: str = ""


@dataclass
class ExecutionContext:
    """
    任务执行上下文 - 管理整个执行过程的状态
    
    【字段说明】
    - plan: 关联的任务计划
    - completed_tasks: 已完成任务的结果字典
    - task_progress: 任务进度字典
    - shared_data: 任务间共享的数据
    - errors: 执行过程中的错误列表
    """
    plan: TaskPlan
    completed_tasks: dict[str, TaskResult] = field(default_factory=dict)
    task_progress: dict[str, TaskProgress] = field(default_factory=dict)
    shared_data: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    
    def add_result(self, task_id: str, result: TaskResult) -> None:
        """
        添加任务执行结果
        
        Args:
            task_id: 任务 ID
            result: 执行结果
        """
        self.completed_tasks[task_id] = result
        self.task_progress[task_id] = TaskProgress(
            task_id=task_id,
            status=result.status,
            completed_at=datetime.now(),
            progress_percent=100.0 if result.is_successful() else 0.0,
            message=result.error or "Completed",
        )
        
        # 将输出存入共享数据，供后续任务使用
        if result.is_successful():
            self.shared_data[task_id] = result.output


# ============================================================================
# 任务执行器
# ============================================================================

class TaskExecutor:
    """
    任务执行器 - 并行执行任务
    
    【执行流程】
    1. 初始化：构建已完成任务集合
    2. 循环执行：
       a. 找出可执行的任务（依赖已满足）
       b. 提交任务到线程池
       c. 收集已完成的结果
       d. 更新共享数据
    3. 重复直到所有任务完成
    4. 返回执行上下文
    """
    
    def __init__(
        self,
        progress_callback: Callable[[TaskProgress], None] | None = None,
        max_workers: int = 3,
    ):
        """
        初始化执行器
        
        Args:
            progress_callback: 进度回调函数
            max_workers: 最大并发数
        """
        self.progress_callback = progress_callback
        self.max_workers = max_workers
        self._lock = threading.Lock()
    
    def execute_plan(
        self,
        plan: TaskPlan,
        context: ExecutionContext | None = None
    ) -> ExecutionContext:
        """
        执行任务计划
        
        Args:
            plan: 要执行的任务计划
            context: 可选的执行上下文（用于恢复执行）
        
        Returns:
            更新后的执行上下文
        """
        if context is None:
            context = ExecutionContext(plan=plan)
        
        # 初始化新任务的进度跟踪
        for task in plan.tasks:
            if task.id not in context.task_progress:
                context.task_progress[task.id] = TaskProgress(
                    task_id=task.id,
                    status=task.status,
                )
        
        # 构建已完成任务集合
        completed: set[str] = set(context.completed_tasks.keys())
        
        # 使用线程池并行执行
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures: dict[str, Future] = {}
            
            # 主循环：持续执行直到全部完成
            while True:
                # 从上下文重新获取已完成任务
                completed = set(context.completed_tasks.keys())
                
                # 找出可执行的任务
                pending_tasks = [
                    task for task in plan.tasks
                    if task.status == TaskStatus.PENDING
                    and task.can_execute(completed)
                ]
                
                # 提交待执行任务
                for task in pending_tasks:
                    if task.id not in futures:
                        task.status = TaskStatus.RUNNING
                        context.task_progress[task.id].status = TaskStatus.RUNNING
                        context.task_progress[task.id].started_at = datetime.now()
                        
                        # 提交到线程池执行
                        future = executor.submit(self._execute_task, task, context)
                        futures[task.id] = future
                
                # 检查已完成的 futures
                done_futures = {tid: f for tid, f in futures.items() if f.done()}
                for task_id, future in done_futures.items():
                    result = future.result()
                    task = plan.get_task(task_id)
                    
                    if task:
                        task.status = result.status
                        task.result = result
                        context.task_progress[task_id].status = result.status
                        context.task_progress[task_id].completed_at = datetime.now()
                        context.add_result(task_id, result)
                        
                        # 更新共享数据
                        if result.is_successful():
                            context.shared_data[task_id] = result.output
                    
                    del futures[task_id]
                
                # 报告进度
                if self.progress_callback:
                    for task_id, progress in context.task_progress.items():
                        self.progress_callback(progress)
                
                # 检查是否全部完成
                if plan.all_completed() or (not pending_tasks and not futures):
                    break
                
                # 小延迟防止忙等待
                time.sleep(0.1)
        
        return context
    
    def _execute_task(
        self,
        task: Task,
        context: ExecutionContext
    ) -> TaskResult:
        """
        执行单个任务
        
        【执行步骤】
        1. 查找合适的 Agent
        2. 准备输入数据（包含依赖任务的输出）
        3. 调用 Agent 执行
        4. 返回结果
        
        Args:
            task: 要执行的任务
            context: 执行上下文
        
        Returns:
            任务执行结果
        """
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        # 报告初始进度
        if self.progress_callback:
            self.progress_callback(TaskProgress(
                task_id=task.id,
                status=TaskStatus.RUNNING,
                started_at=datetime.now(),
                progress_percent=10.0,
                message="Finding agent...",
            ))
        
        # 查找合适的 Agent
        registry = get_registry()
        agent = registry.find_agent_by_capability(task.capability_required)
        if not agent:
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=f"No agent found for capability: {task.capability_required}",
                completed_at=datetime.now(),
            )
        
        # 报告找到 Agent
        if self.progress_callback:
            self.progress_callback(TaskProgress(
                task_id=task.id,
                status=TaskStatus.RUNNING,
                started_at=task.started_at,
                progress_percent=30.0,
                message=f"Agent {agent.name} found",
            ))
        
        # 准备输入数据（包含依赖任务的输出）
        input_data = task.input_data.copy()
        for dep_id in task.depends_on:
            if dep_id in context.shared_data:
                input_data[f"from_{dep_id}"] = context.shared_data[dep_id]
        
        # 更新任务输入
        task.input_data = input_data
        
        # 执行任务
        try:
            # 模拟一些处理时间
            time.sleep(0.2)
            
            # 调用 Agent 执行
            exec_result = agent.execute(task)
            
            if exec_result.success:
                return TaskResult(
                    task_id=task.id,
                    status=TaskStatus.COMPLETED,
                    output=exec_result.output,
                    evidence={"agent": agent.name},
                    completed_at=datetime.now(),
                )
            else:
                return TaskResult(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    error=exec_result.error,
                    completed_at=datetime.now(),
                )
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=str(e),
                completed_at=datetime.now(),
            )
    
    def execute_single_task(
        self,
        task: Task,
        context: ExecutionContext | None = None
    ) -> tuple[TaskResult, ExecutionContext]:
        """
        执行单个任务（适用于流式场景）
        
        Args:
            task: 要执行的任务
            context: 可选的执行上下文
        
        Returns:
            (TaskResult, updated ExecutionContext) 元组
        """
        if context is None:
            plan = TaskPlan(
                id="temp",
                user_request="",
                tasks=[task],
            )
            context = ExecutionContext(plan=plan)
        
        result = self._execute_task(task, context)
        task.status = result.status
        task.result = result
        context.add_result(task.id, result)
        
        return result, context


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    "TaskProgress",
    "ExecutionContext",
    "TaskExecutor",
]
