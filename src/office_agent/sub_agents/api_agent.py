"""
API Agent - 处理 API 调用任务

【功能说明】
ApiAgent 负责：
1. API 调用 - 发起 HTTP 请求
2. 数据获取 - 从外部 API 获取数据
3. 请求记录 - 记录 API 调用历史

【Mock 实现】
本模块使用 Mock 实现，包含模拟的 API 端点数据。
实际使用时替换为真实的 HTTP 客户端（如 httpx、requests）。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from ..base import (
    BaseSubAgent,
    ExecutionResult,
    Task,
    AgentCapability,
)


# ============================================================================
# Mock API 客户端
# ============================================================================

class MockAPIClient:
    """
    Mock API 客户端 - 模拟 API 调用
    
    【支持的端点】
    - /sales: 销售数据
    - /products: 产品目录
    - /employees: 员工数据
    - /weather: 天气信息
    
    【注意】
    实际实现应使用 httpx 或 requests
    """
    
    # Mock 销售数据
    _sales_data = {
        "Q1": {"Widget A": 1000, "Widget B": 800, "Widget C": 500},
        "Q2": {"Widget A": 1200, "Widget B": 900, "Widget C": 600},
        "Q3": {"Widget A": 1100, "Widget B": 950, "Widget C": 700},
        "Q4": {"Widget A": 1300, "Widget B": 1100, "Widget C": 800},
    }
    
    # Mock 产品目录
    _product_catalog = {
        "products": [
            {"id": "P001", "name": "Widget A", "price": 29.99, "category": "Electronics"},
            {"id": "P002", "name": "Widget B", "price": 49.99, "category": "Electronics"},
            {"id": "P003", "name": "Widget C", "price": 19.99, "category": "Accessories"},
        ]
    }
    
    # Mock 员工数据
    _employee_data = {
        "employees": [
            {"id": "E001", "name": "Alice", "department": "Sales", "salary": 75000},
            {"id": "E002", "name": "Bob", "department": "Engineering", "salary": 95000},
            {"id": "E003", "name": "Charlie", "department": "Marketing", "salary": 70000},
        ]
    }
    
    def __init__(self):
        self.call_history: list[dict] = []
    
    def call(
        self,
        endpoint: str,
        method: str = "GET",
        params: dict | None = None,
        data: dict | None = None,
    ) -> dict[str, Any]:
        """
        发起 API 调用
        
        Args:
            endpoint: API 端点
            method: HTTP 方法
            params: URL 参数
            data: 请求体数据
        
        Returns:
            API 响应数据
        """
        # 记录调用历史
        request_info = {
            "endpoint": endpoint,
            "method": method,
            "params": params or {},
            "data": data or {},
            "timestamp": datetime.now().isoformat(),
        }
        self.call_history.append(request_info)
        
        # 路由到相应的处理器
        if "/sales" in endpoint:
            return self._handle_sales_api(endpoint, params)
        elif "/products" in endpoint:
            return self._handle_products_api(endpoint, params)
        elif "/employees" in endpoint:
            return self._handle_employees_api(endpoint, params)
        elif "/weather" in endpoint:
            return self._handle_weather_api(params)
        else:
            return {
                "status": 200,
                "data": {"message": f"Mock response for {endpoint}"},
            }
    
    def _handle_sales_api(self, endpoint: str, params: dict | None) -> dict:
        """处理销售 API"""
        if "aggregate" in endpoint or "summary" in endpoint:
            # 返回聚合数据
            total_sales = {}
            for quarter, products in self._sales_data.items():
                total_sales[quarter] = sum(products.values())
            
            return {
                "status": 200,
                "data": {
                    "quarters": list(self._sales_data.keys()),
                    "totals": total_sales,
                    "grand_total": sum(total_sales.values()),
                }
            }
        return {
            "status": 200,
            "data": self._sales_data,
        }
    
    def _handle_products_api(self, endpoint: str, params: dict | None) -> dict:
        """处理产品 API"""
        if params and "category" in params:
            # 按类别过滤
            filtered = [p for p in self._product_catalog["products"] 
                       if p["category"] == params["category"]]
            return {"status": 200, "data": {"products": filtered}}
        return {"status": 200, "data": self._product_catalog}
    
    def _handle_employees_api(self, endpoint: str, params: dict | None) -> dict:
        """处理员工 API"""
        if params and "department" in params:
            # 按部门过滤
            filtered = [e for e in self._employee_data["employees"]
                       if e["department"] == params["department"]]
            return {"status": 200, "data": {"employees": filtered}}
        return {"status": 200, "data": self._employee_data}
    
    def _handle_weather_api(self, params: dict | None) -> dict:
        """处理天气 API"""
        return {
            "status": 200,
            "data": {
                "temperature": 22,
                "condition": "Sunny",
                "location": params.get("location", "Unknown") if params else "Unknown",
            }
        }


# ============================================================================
# API Agent
# ============================================================================

class ApiAgent(BaseSubAgent):
    """
    API Agent - 处理 API 调用任务
    
    【能力列表】
    - API_CALL: 调用外部 API
    """
    
    def __init__(self):
        self._client = MockAPIClient()
    
    @property
    def name(self) -> str:
        return "api_agent"
    
    @property
    def capabilities(self) -> list[AgentCapability]:
        return [AgentCapability.API_CALL]
    
    def can_handle(self, task: Task) -> bool:
        """检查是否能处理任务"""
        capability = task.capability_required
        if isinstance(capability, str):
            capability = AgentCapability(capability)
        return capability in self.capabilities
    
    def execute(self, task: Task) -> ExecutionResult:
        """
        执行 API 调用任务
        
        【输入参数】
        - endpoint: API 端点
        - method: HTTP 方法 (默认 GET)
        - params: URL 参数
        - data: 请求体数据
        """
        try:
            endpoint = task.input_data.get("endpoint", "")
            method = task.input_data.get("method", "GET")
            params = task.input_data.get("params", {})
            data = task.input_data.get("data", {})
            
            # 调用 API
            response = self._client.call(endpoint, method, params, data)
            
            return ExecutionResult(
                success=response.get("status", 200) == 200,
                output=response,
                metadata={
                    "endpoint": endpoint,
                    "method": method,
                },
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
            )


# ============================================================================
# 导出
# ============================================================================

__all__ = ["ApiAgent", "MockAPIClient"]
