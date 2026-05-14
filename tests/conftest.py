"""
Pytest 配置文件 - 客服智能体测试

本文件配置 pytest 测试环境，包括：
- 环境变量设置
- Python 路径配置
- 测试 fixtures 定义

使用方式：
    pytest tests/  # 运行所有测试
    pytest tests/evals/  # 只运行评测测试

环境要求：
- PYTHONPATH 必须包含 src 目录
- DEEPEVAL_TELEMETRY_OPT_OUT=YES（禁用遥测）
"""

from __future__ import annotations

import os  # 操作系统环境变量
import sys  # Python 路径操作
from pathlib import Path  # 路径处理

import pytest  # pytest 测试框架

# 确保 src 目录在 Python 路径中
# 这样可以正确导入 customer_agent 和 office_agent 模块
_SRC_PATH = Path(__file__).parent.parent / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    设置测试环境变量（会话级别，自动应用）。
    
    这个 fixture 会在整个测试会话开始前执行，
    并在会话结束后自动清理。
    
    设置的环境变量：
    - DEEPEVAL_TELEMETRY_OPT_OUT: 禁用 DeepEval 遥测（避免网络请求）
    - PYTHONPATH: 设置为 src 目录（确保模块导入正常）
    
    使用 @pytest.fixture 的参数：
    - scope="session": 整个测试会话只执行一次
    - autouse=True: 自动应用，无需在测试中显式引用
    """
    # 禁用 DeepEval 遥测
    # 设置为 "YES" 表示不发送使用数据
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

    # 设置 Python 路径
    # 确保可以正确导入 src 目录下的模块
    os.environ.setdefault("PYTHONPATH", str(_SRC_PATH))

    yield  # 测试代码在这里执行

    # 清理阶段（如果需要）
    # 当前没有需要清理的资源
    pass


@pytest.fixture
def mock_human_approval():
    """
    提供模拟人工审批函数的 fixture。
    
    在测试中，当 Agent 需要人工审批时，
    这个 fixture 会自动批准所有请求。
    
    返回的函数签名：
        (user_message: str, draft: str) -> tuple[bool, str | None]
        - user_message: 用户的原始消息
        - draft: Agent 生成的草稿回复
        - 返回值: (是否批准, 修改后的回复或 None)
    
    使用示例：
        def test_with_approval(mock_human_approval):
            result = invoke_customer_agent(msg, mock_human_approval)
            assert result['needs_human_review'] == False
    """

    def approve_or_reject(user_message: str, draft: str) -> tuple[bool, str | None]:
        """
        模拟人工审批函数 - 自动批准所有请求。
        
        Args:
            user_message: 原始用户消息
            draft: Agent 生成的草稿回复
        
        Returns:
            (True, None): 批准请求，不修改回复
        """
        # 测试时自动批准所有请求
        return True, None

    return approve_or_reject


@pytest.fixture
def strict_mock_human_approval():
    """
    提供严格模拟人工审批函数的 fixture。
    
    与 mock_human_approval 不同，这个 fixture
    会拒绝所有请求并添加修改内容。
    
    用于测试"审批拒绝"场景下 Agent 的行为。
    
    返回值：
        (False, modified_draft): 拒绝请求，返回修改后的回复
    
    使用示例：
        def test_rejection(strict_mock_human_approval):
            result = invoke_customer_agent(msg, strict_mock_human_approval)
            assert "Human Modified" in result['response']
    """

    def reject_with_modification(user_message: str, draft: str) -> tuple[bool, str]:
        """
        模拟人工审批函数 - 拒绝并修改。
        
        Args:
            user_message: 原始用户消息
            draft: Agent 生成的草稿回复
        
        Returns:
            (False, modified_draft): 拒绝请求，添加升级标注
        """
        # 在回复末尾添加人工修改标注
        modified = draft + "\n\n[Human Modified: Your request has been escalated.]"
        return False, modified

    return reject_with_modification
