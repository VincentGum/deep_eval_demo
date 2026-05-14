"""
Human-in-the-Loop - 人工介入机制

【模块概述】
本模块提供 Human-in-the-Loop (HITL) 机制，用于：
1. 暂停执行 - 在关键时刻暂停任务
2. 请求输入 - 向用户请求必要信息
3. 超时管理 - 处理用户无响应的情况
4. 结果恢复 - 将人工输入整合到执行流程

【使用场景】
- 敏感操作确认（如退款、取消）
- 信息缺失时的补充
- 异常情况处理
- 决策验证

【核心组件】
1. HumanInputRequest - 输入请求
2. HumanInputResult - 输入结果
3. HumanInTheLoop - 主管理器
4. MockHumanInputHandler - Mock 实现
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from .base import Task, TaskStatus


# ============================================================================
# 输入状态枚举
# ============================================================================

class HumanInputStatus(str, Enum):
    """
    人工输入状态枚举
    
    【状态说明】
    - PENDING: 等待输入
    - RECEIVED: 已收到输入
    - TIMEOUT: 超时
    - CANCELLED: 已取消
    """
    PENDING = "pending"
    RECEIVED = "received"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


# ============================================================================
# 输入请求和结果
# ============================================================================

@dataclass
class HumanInputRequest:
    """
    人工输入请求
    
    【字段说明】
    - id: 请求唯一 ID
    - task_id: 关联的任务 ID
    - question: 要询问用户的问题
    - context: 额外上下文信息
    - options: 可选的选项列表
    - required: 是否必须输入
    - timeout_seconds: 超时时间（秒）
    - created_at: 创建时间
    - status: 当前状态
    - response: 收到的响应
    - responded_at: 响应时间
    """
    id: str
    task_id: str
    question: str
    context: dict[str, Any] = field(default_factory=dict)
    options: list[str] | None = None
    required: bool = True
    timeout_seconds: int = 300
    created_at: datetime = field(default_factory=datetime.now)
    status: HumanInputStatus = HumanInputStatus.PENDING
    response: str | None = None
    responded_at: datetime | None = None


@dataclass
class HumanInputResult:
    """
    人工输入结果
    
    【字段说明】
    - request_id: 请求 ID
    - task_id: 任务 ID
    - status: 输入状态
    - response: 用户响应内容
    - wait_time_seconds: 等待时间
    - timestamp: 结果时间
    """
    request_id: str
    task_id: str
    status: HumanInputStatus
    response: str | None
    wait_time_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# Human-in-the-Loop 管理器
# ============================================================================

class HumanInTheLoop:
    """
    Human-in-the-Loop 管理器
    
    【管理功能】
    - 创建和管理输入请求
    - 等待用户响应
    - 处理超时
    - 支持多个并发请求
    """
    
    def __init__(
        self,
        default_timeout: int = 300,
        auto_proceed_on_timeout: bool = False,
    ):
        """
        初始化 HITL 管理器
        
        Args:
            default_timeout: 默认超时时间（秒）
            auto_proceed_on_timeout: 超时后是否自动继续
        """
        self.default_timeout = default_timeout
        self.auto_proceed_on_timeout = auto_proceed_on_timeout
        
        self._pending_requests: dict[str, HumanInputRequest] = {}
        self._responses: dict[str, HumanInputResult] = {}
        self._lock = threading.Lock()
        
        # 默认输入处理器（可替换）
        self._input_handler: Callable[[HumanInputRequest], str | None] | None = None
    
    def set_input_handler(
        self,
        handler: Callable[[HumanInputRequest], str | None]
    ) -> None:
        """
        设置输入处理器
        
        Args:
            handler: 接收 HumanInputRequest，返回输入字符串的函数
        """
        self._input_handler = handler
    
    def request_input(
        self,
        task_id: str,
        question: str,
        context: dict[str, Any] | None = None,
        options: list[str] | None = None,
        timeout_seconds: int | None = None,
    ) -> HumanInputResult:
        """
        请求人工输入
        
        【请求流程】
        1. 创建请求
        2. 如果有处理器，直接调用
        3. 否则进入等待循环
        
        Args:
            task_id: 任务 ID
            question: 要询问的问题
            context: 额外上下文
            options: 可选选项
            timeout_seconds: 超时时间
        
        Returns:
            输入结果
        """
        timeout = timeout_seconds or self.default_timeout
        request_id = f"input_{task_id}_{int(time.time())}"
        
        request = HumanInputRequest(
            id=request_id,
            task_id=task_id,
            question=question,
            context=context or {},
            options=options,
            timeout_seconds=timeout,
        )
        
        with self._lock:
            self._pending_requests[request_id] = request
        
        start_time = time.time()
        
        # 如果有输入处理器，直接调用
        if self._input_handler:
            response = self._input_handler(request)
            if response is not None:
                request.status = HumanInputStatus.RECEIVED
                request.response = response
                request.responded_at = datetime.now()
                
                with self._lock:
                    self._pending_requests[request_id] = request
                
                return HumanInputResult(
                    request_id=request_id,
                    task_id=task_id,
                    status=HumanInputStatus.RECEIVED,
                    response=response,
                    wait_time_seconds=time.time() - start_time,
                )
        
        # 否则进入等待循环
        while (time.time() - start_time) < timeout:
            with self._lock:
                updated_request = self._pending_requests.get(request_id)
            
            if updated_request and updated_request.status == HumanInputStatus.RECEIVED:
                return HumanInputResult(
                    request_id=request_id,
                    task_id=task_id,
                    status=HumanInputStatus.RECEIVED,
                    response=updated_request.response,
                    wait_time_seconds=time.time() - start_time,
                )
            
            # 检查超时
            if (time.time() - start_time) >= timeout:
                request.status = HumanInputStatus.TIMEOUT
                with self._lock:
                    self._pending_requests[request_id] = request
                
                return HumanInputResult(
                    request_id=request_id,
                    task_id=task_id,
                    status=HumanInputStatus.TIMEOUT,
                    response=None if self.auto_proceed_on_timeout else None,
                    wait_time_seconds=timeout,
                )
            
            time.sleep(0.1)
        
        # 不会执行到这里
        request.status = HumanInputStatus.TIMEOUT
        return HumanInputResult(
            request_id=request_id,
            task_id=task_id,
            status=HumanInputStatus.TIMEOUT,
            response=None,
            wait_time_seconds=timeout,
        )
    
    def provide_input(
        self,
        request_id: str,
        response: str
    ) -> bool:
        """
        提供输入
        
        Args:
            request_id: 请求 ID
            response: 响应内容
        
        Returns:
            是否成功
        """
        with self._lock:
            if request_id in self._pending_requests:
                request = self._pending_requests[request_id]
                request.status = HumanInputStatus.RECEIVED
                request.response = response
                request.responded_at = datetime.now()
                self._pending_requests[request_id] = request
                return True
        return False
    
    def cancel_request(self, request_id: str) -> bool:
        """
        取消请求
        
        Args:
            request_id: 请求 ID
        
        Returns:
            是否成功
        """
        with self._lock:
            if request_id in self._pending_requests:
                request = self._pending_requests[request_id]
                request.status = HumanInputStatus.CANCELLED
                self._pending_requests[request_id] = request
                return True
        return False
    
    def get_pending_requests(self) -> list[HumanInputRequest]:
        """获取所有待处理的请求"""
        with self._lock:
            return [
                req for req in self._pending_requests.values()
                if req.status == HumanInputStatus.PENDING
            ]
    
    def get_request_status(self, request_id: str) -> HumanInputStatus | None:
        """获取请求状态"""
        with self._lock:
            request = self._pending_requests.get(request_id)
            return request.status if request else None
    
    def generate_prompt(self, request: HumanInputRequest) -> str:
        """
        生成人工可读的提示
        
        Args:
            request: 输入请求
        
        Returns:
            格式化的提示字符串
        """
        lines = [
            "=" * 60,
            "需要人工输入",
            "=" * 60,
            "",
            f"任务: {request.task_id}",
            f"问题: {request.question}",
            "",
        ]
        
        if request.context:
            lines.append("上下文:")
            for key, value in request.context.items():
                lines.append(f"  - {key}: {value}")
            lines.append("")
        
        if request.options:
            lines.append("选项:")
            for i, option in enumerate(request.options, 1):
                lines.append(f"  {i}. {option}")
            lines.append("")
        
        lines.extend([
            f"超时: {request.timeout_seconds} 秒",
            "=" * 60,
        ])
        
        return "\n".join(lines)


# ============================================================================
# Mock 输入处理器
# ============================================================================

class MockHumanInputHandler:
    """
    Mock 人工输入处理器
    
    用于演示和测试，模拟用户输入。
    """
    
    def __init__(self):
        self._responses: dict[str, str] = {}
        self._manual_mode = False
        self._manual_input: str | None = None
    
    def set_predefined_response(
        self,
        task_pattern: str,
        response: str
    ) -> None:
        """设置预定义的响应"""
        self._responses[task_pattern] = response
    
    def enable_manual_mode(self) -> None:
        """启用手动模式（会阻塞等待输入）"""
        self._manual_mode = True
    
    def set_manual_input(self, response: str) -> None:
        """设置手动输入"""
        self._manual_input = response
    
    def __call__(self, request: HumanInputRequest) -> str | None:
        """处理输入请求"""
        # 检查预定义响应
        for pattern, response in self._responses.items():
            if pattern in request.question.lower():
                return response
        
        # 检查手动输入
        if self._manual_input:
            return self._manual_input
        
        # 手动模式
        if self._manual_mode:
            print("\n" + "=" * 60)
            print(f"需要人工输入: {request.question}")
            if request.options:
                print("选项:")
                for i, opt in enumerate(request.options, 1):
                    print(f"  {i}. {opt}")
            print("=" * 60)
            return input("请输入: ")
        
        # 默认：自动批准
        return "approved"


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    "HumanInputStatus",
    "HumanInputRequest",
    "HumanInputResult",
    "HumanInTheLoop",
    "MockHumanInputHandler",
]
