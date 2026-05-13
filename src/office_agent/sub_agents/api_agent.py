"""API Agent - Handles API calls and data fetching."""

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


class MockAPIClient:
    """Mock API client for offline demo.

    Simulates various API endpoints commonly used in office scenarios.
    """

    # Mock data stores
    _sales_data = {
        "Q1": {"Widget A": 1000, "Widget B": 800, "Widget C": 500},
        "Q2": {"Widget A": 1200, "Widget B": 900, "Widget C": 600},
        "Q3": {"Widget A": 1100, "Widget B": 950, "Widget C": 700},
        "Q4": {"Widget A": 1300, "Widget B": 1100, "Widget C": 800},
    }

    _product_catalog = {
        "products": [
            {"id": "P001", "name": "Widget A", "price": 29.99, "category": "Electronics"},
            {"id": "P002", "name": "Widget B", "price": 49.99, "category": "Electronics"},
            {"id": "P003", "name": "Widget C", "price": 19.99, "category": "Accessories"},
        ]
    }

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
        """Make an API call."""
        request_info = {
            "endpoint": endpoint,
            "method": method,
            "params": params or {},
            "data": data or {},
            "timestamp": datetime.now().isoformat(),
        }
        self.call_history.append(request_info)

        # Route to appropriate handler
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
        """Handle sales API calls."""
        if "aggregate" in endpoint or "summary" in endpoint:
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
        """Handle products API calls."""
        if params and params.get("category"):
            category = params["category"]
            filtered = [p for p in self._product_catalog["products"] if p["category"] == category]
            return {"status": 200, "data": {"products": filtered}}
        return {"status": 200, "data": self._product_catalog}

    def _handle_employees_api(self, endpoint: str, params: dict | None) -> dict:
        """Handle employees API calls."""
        if params and params.get("department"):
            dept = params["department"]
            filtered = [e for e in self._employee_data["employees"] if e["department"] == dept]
            return {"status": 200, "data": {"employees": filtered}}
        return {"status": 200, "data": self._employee_data}

    def _handle_weather_api(self, params: dict | None) -> dict:
        """Handle weather API call (mock)."""
        return {
            "status": 200,
            "data": {
                "location": params.get("city", "Unknown"),
                "temperature": 22,
                "condition": "Partly Cloudy",
                "humidity": 65,
            }
        }


class ApiAgent(BaseSubAgent):
    """Agent for handling API calls.

    Capabilities:
    - Call REST APIs
    - Handle authentication
    - Process paginated responses
    """

    def __init__(self):
        self._client = MockAPIClient()

    @property
    def name(self) -> str:
        return "api_agent"

    @property
    def capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability.API_CALL,
            AgentCapability.API_AUTH,
            AgentCapability.API_PAGINATE,
        ]

    def can_handle(self, task: Task) -> bool:
        capability = task.capability_required
        if isinstance(capability, str):
            capability = AgentCapability(capability)
        return capability in self.capabilities

    def execute(self, task: Task) -> ExecutionResult:
        """Execute an API task."""
        try:
            endpoint = task.input_data.get("endpoint", "")
            method = task.input_data.get("method", "GET")
            params = task.input_data.get("params")
            data = task.input_data.get("data")

            if not endpoint:
                return ExecutionResult(
                    success=False,
                    error="No endpoint provided",
                )

            response = self._client.call(endpoint, method, params, data)

            if response.get("status") == 200:
                return ExecutionResult(
                    success=True,
                    output=response.get("data"),
                    metadata={
                        "endpoint": endpoint,
                        "method": method,
                        "call_count": len(self._client.call_history),
                    },
                )
            else:
                return ExecutionResult(
                    success=False,
                    error=f"API error: {response.get('status')}",
                    output=response,
                )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
            )
