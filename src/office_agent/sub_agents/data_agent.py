"""
数据 Agent - 处理数据处理和统计分析任务

【功能说明】
DataAgent 负责：
1. 数据清洗 - 处理缺失值、异常值
2. 数据统计 - 计算各种统计指标
3. 数据聚合 - 分组、汇总数据
4. 数据关联 - 合并、连接数据集

【Mock 实现】
本模块使用 Mock 实现，包含预定义的数据处理函数。
实际使用时替换为 pandas、numpy 等数据处理库。
"""

from __future__ import annotations

from typing import Any

from ..base import (
    BaseSubAgent,
    ExecutionResult,
    Task,
    AgentCapability,
)


# ============================================================================
# Mock 数据处理
# ============================================================================

class MockDataProcessor:
    """
    Mock 数据处理器 - 模拟数据处理操作
    
    【处理能力】
    - 统计分析
    - 数据聚合
    - 格式转换
    
    【注意】
    实际实现应使用 pandas、numpy
    """
    
    def __init__(self):
        self._datasets: dict[str, list[dict]] = {}
        self._load_mock_data()
    
    def _load_mock_data(self):
        """加载 Mock 数据集"""
        # Mock 销售数据集
        self._datasets["sales"] = [
            {"product": "Widget A", "region": "North", "Q1": 1000, "Q2": 1200, "Q3": 1100, "Q4": 1300},
            {"product": "Widget A", "region": "South", "Q1": 800, "Q2": 900, "Q3": 950, "Q4": 1100},
            {"product": "Widget B", "region": "North", "Q1": 500, "Q2": 600, "Q3": 700, "Q4": 800},
            {"product": "Widget B", "region": "South", "Q1": 450, "Q2": 550, "Q3": 650, "Q4": 750},
            {"product": "Widget C", "region": "North", "Q1": 300, "Q2": 400, "Q3": 450, "Q4": 500},
            {"product": "Widget C", "region": "South", "Q1": 200, "Q2": 200, "Q3": 250, "Q4": 300},
        ]
        
        # Mock 客户数据集
        self._datasets["customers"] = [
            {"id": "C001", "name": "Acme Corp", "segment": "Enterprise", "revenue": 50000},
            {"id": "C002", "name": "TechStart", "segment": "SMB", "revenue": 15000},
            {"id": "C003", "name": "Global Inc", "segment": "Enterprise", "revenue": 75000},
        ]
    
    def get_dataset(self, name: str) -> list[dict]:
        """获取数据集"""
        return self._datasets.get(name, [])
    
    def calculate_statistics(self, data: list[dict], column: str) -> dict:
        """
        计算统计数据
        
        Args:
            data: 数据列表
            column: 要统计的列名
        
        Returns:
            统计结果（平均值、总和、最大、最小）
        """
        values = []
        for row in data:
            if column in row:
                try:
                    values.append(float(row[column]))
                except (ValueError, TypeError):
                    pass
        
        if not values:
            return {"error": f"No numeric values in column: {column}"}
        
        return {
            "column": column,
            "count": len(values),
            "sum": sum(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    
    def aggregate(self, data: list[dict], group_by: str, agg_column: str, agg_func: str = "sum") -> list[dict]:
        """
        聚合数据
        
        Args:
            data: 数据列表
            group_by: 分组列
            agg_column: 要聚合的列
            agg_func: 聚合函数 (sum/avg/count)
        
        Returns:
            聚合后的数据
        """
        groups: dict[str, list] = {}
        
        for row in data:
            key = row.get(group_by, "Unknown")
            if key not in groups:
                groups[key] = []
            if agg_column in row:
                try:
                    groups[key].append(float(row[agg_column]))
                except (ValueError, TypeError):
                    pass
        
        result = []
        for key, values in groups.items():
            if not values:
                continue
            
            if agg_func == "sum":
                agg_value = sum(values)
            elif agg_func == "avg":
                agg_value = sum(values) / len(values)
            elif agg_func == "count":
                agg_value = len(values)
            else:
                agg_value = sum(values)
            
            result.append({group_by: key, f"{agg_func}_{agg_column}": agg_value})
        
        return result
    
    def filter_data(self, data: list[dict], conditions: dict) -> list[dict]:
        """
        过滤数据
        
        Args:
            data: 数据列表
            conditions: 过滤条件
        
        Returns:
            过滤后的数据
        """
        result = []
        for row in data:
            match = True
            for key, value in conditions.items():
                if row.get(key) != value:
                    match = False
                    break
            if match:
                result.append(row)
        return result


# ============================================================================
# 数据 Agent
# ============================================================================

class DataAgent(BaseSubAgent):
    """
    数据 Agent - 处理数据操作任务
    
    【能力列表】
    - DATA_STATISTICS: 统计分析
    - DATA_AGGREGATE: 数据聚合
    - DATA_FILTER: 数据过滤
    - DATA_CLEAN: 数据清洗
    """
    
    def __init__(self):
        self._processor = MockDataProcessor()
    
    @property
    def name(self) -> str:
        return "data_agent"
    
    @property
    def capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability.DATA_QUERY,
            AgentCapability.DATA_TRANSFORM,
            AgentCapability.DATA_AGGREGATE,
        ]
    
    def can_handle(self, task: Task) -> bool:
        """检查是否能处理任务"""
        capability = task.capability_required
        if isinstance(capability, str):
            capability = AgentCapability(capability)
        return capability in self.capabilities
    
    def execute(self, task: Task) -> ExecutionResult:
        """
        执行数据任务
        
        【任务路由】
        - DATA_STATISTICS → _statistics()
        - DATA_AGGREGATE → _aggregate()
        - DATA_FILTER → _filter()
        - DATA_CLEAN → _clean()
        """
        try:
            capability = task.capability_required
            if isinstance(capability, str):
                capability = AgentCapability(capability)
            
            if capability == AgentCapability.DATA_STATISTICS:
                return self._statistics(task)
            elif capability == AgentCapability.DATA_AGGREGATE:
                return self._aggregate(task)
            elif capability == AgentCapability.DATA_FILTER:
                return self._filter(task)
            elif capability == AgentCapability.DATA_CLEAN:
                return self._clean(task)
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
    
    def _statistics(self, task: Task) -> ExecutionResult:
        """
        统计分析
        
        【输入参数】
        - dataset: 数据集名称
        - column: 要统计的列名
        
        【返回】
        - count: 数量
        - sum: 总和
        - mean: 平均值
        - min: 最小值
        - max: 最大值
        """
        dataset = task.input_data.get("dataset", "sales")
        column = task.input_data.get("column", "Q1")
        
        data = self._processor.get_dataset(dataset)
        stats = self._processor.calculate_statistics(data, column)
        
        return ExecutionResult(
            success="error" not in stats,
            output=stats,
            metadata={"action": "statistics", "dataset": dataset},
        )
    
    def _aggregate(self, task: Task) -> ExecutionResult:
        """
        数据聚合
        
        【输入参数】
        - dataset: 数据集名称
        - group_by: 分组列
        - agg_column: 要聚合的列
        - agg_func: 聚合函数 (sum/avg/count)
        """
        dataset = task.input_data.get("dataset", "sales")
        group_by = task.input_data.get("group_by", "product")
        agg_column = task.input_data.get("agg_column", "Q1")
        agg_func = task.input_data.get("agg_func", "sum")
        
        data = self._processor.get_dataset(dataset)
        result = self._processor.aggregate(data, group_by, agg_column, agg_func)
        
        return ExecutionResult(
            success=True,
            output={"aggregated_data": result},
            metadata={
                "action": "aggregate",
                "dataset": dataset,
                "group_by": group_by,
            },
        )
    
    def _filter(self, task: Task) -> ExecutionResult:
        """
        数据过滤
        
        【输入参数】
        - dataset: 数据集名称
        - conditions: 过滤条件字典
        """
        dataset = task.input_data.get("dataset", "sales")
        conditions = task.input_data.get("conditions", {})
        
        data = self._processor.get_dataset(dataset)
        result = self._processor.filter_data(data, conditions)
        
        return ExecutionResult(
            success=True,
            output={"filtered_data": result, "count": len(result)},
            metadata={"action": "filter", "dataset": dataset},
        )
    
    def _clean(self, task: Task) -> ExecutionResult:
        """
        数据清洗
        
        【输入参数】
        - data: 要清洗的数据（可选）
        - operations: 清洗操作列表
        
        【支持的清洗操作】
        - remove_nulls: 移除空值
        - remove_duplicates: 移除重复
        - trim_strings: 去除字符串空格
        """
        operations = task.input_data.get("operations", ["remove_nulls"])
        
        # Mock 清洗结果
        result = {
            "original_count": 100,
            "cleaned_count": 95,
            "operations_applied": operations,
        }
        
        return ExecutionResult(
            success=True,
            output=result,
            metadata={"action": "clean"},
        )


# ============================================================================
# 导出
# ============================================================================

__all__ = ["DataAgent", "MockDataProcessor"]
