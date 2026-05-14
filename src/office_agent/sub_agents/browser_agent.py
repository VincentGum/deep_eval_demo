"""
浏览器 Agent - 处理网页浏览和内容爬取任务

【功能说明】
BrowserAgent 负责：
1. URL 导航 - 访问指定网页
2. 内容爬取 - 从网页提取数据
3. 表单填写 - 填写和提交表单

【Mock 实现】
本模块使用 Mock 实现，用于演示和测试。
实际使用时替换为 Playwright/Selenium。
"""

from __future__ import annotations

import re
from typing import Any

from ..base import (
    BaseSubAgent,
    ExecutionResult,
    Task,
    AgentCapability,
)


# ============================================================================
# Mock 浏览器
# ============================================================================

class MockBrowser:
    """
    Mock 浏览器 - 模拟浏览器行为
    
    【模拟功能】
    - navigate: 导航到 URL
    - scrape: 爬取页面内容
    - fill_form: 填写表单
    
    【注意】
    实际实现应使用 Playwright 或 Selenium
    """
    
    def __init__(self):
        self.current_url: str | None = None
        self.history: list[str] = []
    
    def navigate(self, url: str) -> str:
        """
        模拟导航到 URL
        
        Args:
            url: 目标网址
        
        Returns:
            导航结果消息
        """
        self.current_url = url
        self.history.append(url)
        return f"[Browser] Navigated to {url}"
    
    def scrape(self, selector: str | None = None) -> dict[str, Any]:
        """
        模拟爬取页面内容
        
        Args:
            selector: CSS 选择器（可选）
        
        Returns:
            爬取的数据字典
        """
        if not self.current_url:
            return {"error": "No page loaded"}
        
        # 根据 URL 返回 Mock 数据
        mock_data = self._get_mock_page_data(self.current_url)
        return {
            "url": self.current_url,
            "title": f"Page - {self.current_url}",
            "content": mock_data.get("content", ""),
            "links": mock_data.get("links", []),
            "tables": mock_data.get("tables", []),
        }
    
    def fill_form(self, fields: dict[str, str]) -> str:
        """
        模拟填写表单
        
        Args:
            fields: 表单字段字典
        
        Returns:
            填写结果消息
        """
        return f"[Browser] Form filled with fields: {list(fields.keys())}"
    
    def _get_mock_page_data(self, url: str) -> dict[str, Any]:
        """
        根据 URL 返回 Mock 数据
        
        Args:
            url: 页面 URL
        
        Returns:
            Mock 页面数据
        """
        if "sales" in url.lower():
            return {
                "content": "Sales Data Report",
                "tables": [
                    {"headers": ["Product", "Q1", "Q2", "Q3", "Q4"],
                     "rows": [["Widget A", "1000", "1200", "1100", "1300"],
                             ["Widget B", "800", "900", "950", "1100"],
                             ["Widget C", "500", "600", "700", "800"]]},
                ],
                "links": ["/products", "/orders", "/reports"],
            }
        elif "news" in url.lower():
            return {
                "content": "Latest News Headlines",
                "tables": [],
                "links": ["/article/1", "/article/2", "/article/3"],
            }
        else:
            return {
                "content": f"Generic page content from {url}",
                "tables": [],
                "links": ["/page1", "/page2"],
            }


# ============================================================================
# 浏览器 Agent
# ============================================================================

class BrowserAgent(BaseSubAgent):
    """
    浏览器 Agent - 处理浏览器相关任务
    
    【能力列表】
    - BROWSER_NAVIGATE: 导航到 URL
    - BROWSER_SCRAPE: 爬取页面
    - BROWSER_FILL_FORM: 填写表单
    """
    
    def __init__(self):
        self._browser = MockBrowser()
    
    @property
    def name(self) -> str:
        return "browser_agent"
    
    @property
    def capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability.BROWSER_NAVIGATE,
            AgentCapability.BROWSER_SCRAPE,
            AgentCapability.BROWSER_FILL_FORM,
        ]
    
    def can_handle(self, task: Task) -> bool:
        """检查是否能处理任务"""
        capability = task.capability_required
        if isinstance(capability, str):
            capability = AgentCapability(capability)
        return capability in self.capabilities
    
    def execute(self, task: Task) -> ExecutionResult:
        """
        执行浏览器任务
        
        【任务路由】
        - BROWSER_NAVIGATE → _navigate()
        - BROWSER_SCRAPE → _scrape()
        - BROWSER_FILL_FORM → _fill_form()
        """
        try:
            capability = task.capability_required
            if isinstance(capability, str):
                capability = AgentCapability(capability)
            
            if capability == AgentCapability.BROWSER_NAVIGATE:
                return self._navigate(task)
            elif capability == AgentCapability.BROWSER_SCRAPE:
                return self._scrape(task)
            elif capability == AgentCapability.BROWSER_FILL_FORM:
                return self._fill_form(task)
            else:
                return ExecutionResult(
                    success=False,
                    error=f"Unknown capability: {capability}",
                )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
            )
    
    def _navigate(self, task: Task) -> ExecutionResult:
        """
        导航到 URL
        
        【输入参数】
        - url: 目标网址
        
        【特殊处理】
        - 如果 url 为空，使用默认客户网站
        - 如果 url 包含模板变量，解析为实际 URL
        """
        url = task.input_data.get("url", "")
        if not url:
            # Mock: 使用默认客户网站（模拟数据库查询结果）
            url = "https://www.example-customer.com"
            task.output = {"message": f"Navigated to customer website: {url}", "url": url}
        elif "{" in url:
            # Mock: URL 模板
            url = "https://www.example-customer.com"
            task.output = {"message": f"Resolved URL from template to: {url}", "url": url}
        
        result = self._browser.navigate(url)
        return ExecutionResult(
            success=True,
            output={
                "message": result,
                "url": url,
                "history": self._browser.history.copy(),
            },
            metadata={"action": "navigate"},
        )
    
    def _scrape(self, task: Task) -> ExecutionResult:
        """
        爬取页面内容
        
        【输入参数】
        - selector: CSS 选择器（可选）
        
        【返回】
        - url: 页面 URL
        - title: 页面标题
        - content: 页面内容
        - links: 链接列表
        - tables: 表格数据
        """
        selector = task.input_data.get("selector")
        data = self._browser.scrape(selector)
        
        if "error" in data:
            return ExecutionResult(
                success=False,
                error=data["error"],
            )
        
        return ExecutionResult(
            success=True,
            output=data,
            metadata={
                "action": "scrape",
                "url": data.get("url"),
            },
        )
    
    def _fill_form(self, task: Task) -> ExecutionResult:
        """
        填写表单
        
        【输入参数】
        - fields: 表单字段字典
        """
        fields = task.input_data.get("fields", {})
        result = self._browser.fill_form(fields)
        
        return ExecutionResult(
            success=True,
            output={"message": result, "fields": fields},
            metadata={"action": "fill_form"},
        )


# ============================================================================
# 导出
# ============================================================================

__all__ = ["BrowserAgent", "MockBrowser"]
