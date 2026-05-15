"""
可视化 Agent - 生成图表和数据可视化

【功能说明】
VisualizationAgent 负责：
1. 图表生成 - 生成各种类型的图表
2. 数据格式化 - 将数据转换为可视化格式
3. 图表配置 - 自定义图表样式和布局
4. 导出支持 - 导出为图片或 HTML

【Mock 实现】
本模块使用 Mock 实现，返回图表的配置信息。
实际使用时替换为 matplotlib、plotly、echarts 等图表库。
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
# Mock 图表生成器
# ============================================================================

class MockChartGenerator:
    """
    Mock 图表生成器 - 模拟图表生成
    
    【支持的图表类型】
    - bar: 柱状图
    - line: 折线图
    - pie: 饼图
    - scatter: 散点图
    - table: 表格
    
    【注意】
    实际实现应使用 matplotlib、plotly 或 echarts
    """
    
    def generate(self, chart_type: str, data: dict[str, Any]) -> dict[str, Any]:
        """
        生成图表
        
        Args:
            chart_type: 图表类型
            data: 图表数据
        
        Returns:
            图表配置信息
        """
        if chart_type == "bar":
            return self._generate_bar_chart(data)
        elif chart_type == "line":
            return self._generate_line_chart(data)
        elif chart_type == "pie":
            return self._generate_pie_chart(data)
        elif chart_type == "scatter":
            return self._generate_scatter_chart(data)
        elif chart_type == "table":
            return self._generate_table(data)
        else:
            return {"error": f"Unsupported chart type: {chart_type}"}
    
    def _generate_bar_chart(self, data: dict) -> dict:
        """生成柱状图配置"""
        return {
            "type": "bar",
            "title": data.get("title", "Bar Chart"),
            "x_axis": data.get("labels", ["A", "B", "C"]),
            "y_axis": data.get("values", [10, 20, 30]),
            "config": {
                "color": "#4CAF50",
                "show_values": True,
            },
            "html": f"""
            <div class="chart bar">
                <h3>{data.get("title", "Bar Chart")}</h3>
                <div class="bars">
                    {''.join(f'<div class="bar" style="height:{v * 2}px">{l}: {v}</div>' 
                            for l, v in zip(data.get("labels", []), data.get("values", [])))}
                </div>
            </div>
            """,
        }
    
    def _generate_line_chart(self, data: dict) -> dict:
        """生成折线图配置"""
        return {
            "type": "line",
            "title": data.get("title", "Line Chart"),
            "x_axis": data.get("labels", ["Jan", "Feb", "Mar", "Apr"]),
            "y_axis": data.get("values", [10, 15, 12, 18]),
            "config": {
                "color": "#2196F3",
                "show_points": True,
            },
            "html": f"""
            <div class="chart line">
                <h3>{data.get("title", "Line Chart")}</h3>
                <div class="line-graph">
                    {''.join(f'<span class="point" style="left:{i * 25}%;bottom:{v * 2}px">{v}</span>'
                            for i, v in enumerate(data.get("values", [])))}
                </div>
            </div>
            """,
        }
    
    def _generate_pie_chart(self, data: dict) -> dict:
        """生成饼图配置"""
        labels = data.get("labels", ["A", "B", "C"])
        values = data.get("values", [30, 40, 30])
        colors = ["#4CAF50", "#2196F3", "#FFC107"]
        
        return {
            "type": "pie",
            "title": data.get("title", "Pie Chart"),
            "segments": [
                {"label": l, "value": v, "percentage": v / sum(values) * 100, "color": colors[i % len(colors)]}
                for i, (l, v) in enumerate(zip(labels, values))
            ],
            "html": f"""
            <div class="chart pie">
                <h3>{data.get("title", "Pie Chart")}</h3>
                <div class="pie-chart">
                    {''.join(f'<div class="segment" style="--p:{v / sum(values)};--c:{colors[i % len(colors)]}">{l}: {v}</div>'
                            for i, (l, v) in enumerate(zip(labels, values)))}
                </div>
            </div>
            """,
        }
    
    def _generate_scatter_chart(self, data: dict) -> dict:
        """生成散点图配置"""
        return {
            "type": "scatter",
            "title": data.get("title", "Scatter Plot"),
            "points": [
                {"x": x, "y": y, "label": l}
                for x, y, l in zip(data.get("x_values", [1, 2, 3]), data.get("y_values", [10, 20, 30]), data.get("labels", ["A", "B", "C"]))
            ],
            "config": {
                "color": "#9C27B0",
            },
            "html": f"""
            <div class="chart scatter">
                <h3>{data.get("title", "Scatter Plot")}</h3>
                <div class="scatter-plot">
                    {''.join(f'<span class="point" style="left:{x * 10}%;bottom:{y}px">{l}</span>'
                            for x, y, l in zip(data.get("x_values", []), data.get("y_values", []), data.get("labels", [])))}
                </div>
            </div>
            """,
        }
    
    def _generate_table(self, data: dict) -> dict:
        """生成表格"""
        headers = data.get("headers", ["Column 1", "Column 2", "Column 3"])
        rows = data.get("rows", [["A1", "B1", "C1"], ["A2", "B2", "C2"]])
        
        return {
            "type": "table",
            "title": data.get("title", "Data Table"),
            "headers": headers,
            "rows": rows,
            "html": f"""
            <div class="chart table">
                <h3>{data.get("title", "Data Table")}</h3>
                <table>
                    <thead><tr>{''.join(f'<th>{h}</th>' for h in headers)}</tr></thead>
                    <tbody>
                        {''.join(f'<tr>{"".join(f"<td>{c}</td>" for c in row)}</tr>' for row in rows)}
                    </tbody>
                </table>
            </div>
            """,
        }


# ============================================================================
# 可视化 Agent
# ============================================================================

class VisualizationAgent(BaseSubAgent):
    """
    可视化 Agent - 生成数据可视化
    
    【能力列表】
    - CHART_CREATE: 创建图表
    - TABLE_CREATE: 创建表格
    - REPORT_GENERATE: 生成报告
    """
    
    def __init__(self):
        self._generator = MockChartGenerator()
    
    @property
    def name(self) -> str:
        return "visualization_agent"
    
    @property
    def capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability.CHART_CREATE,
            AgentCapability.TABLE_CREATE,
            AgentCapability.REPORT_GENERATE,
        ]
    
    def can_handle(self, task: Task) -> bool:
        """检查是否能处理任务"""
        capability = task.capability_required
        if isinstance(capability, str):
            capability = AgentCapability(capability)
        return capability in self.capabilities
    
    def execute(self, task: Task) -> ExecutionResult:
        """
        执行可视化任务
        
        【任务路由】
        - CHART_CREATE → _generate()
        - TABLE_CREATE → _format()
        - REPORT_GENERATE → _generate()
        """
        try:
            capability = task.capability_required
            if isinstance(capability, str):
                capability = AgentCapability(capability)
            
            if capability == AgentCapability.CHART_CREATE:
                return self._generate(task)
            elif capability == AgentCapability.TABLE_CREATE:
                return self._format(task)
            elif capability == AgentCapability.REPORT_GENERATE:
                return self._generate(task)
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
    
    def _generate(self, task: Task) -> ExecutionResult:
        """
        生成图表
        
        【输入参数】
        - chart_type: 图表类型 (bar/line/pie/scatter/table)
        - data: 图表数据
          - labels: X 轴标签或类别
          - values: 数据值
          - title: 图表标题
        
        【返回】
        - chart_config: 图表配置
        - html: HTML 表示
        """
        chart_type = task.input_data.get("chart_type", "bar")
        data = task.input_data.get("data", {})
        
        result = self._generator.generate(chart_type, data)
        
        if "error" in result:
            return ExecutionResult(
                success=False,
                error=result["error"],
            )
        
        return ExecutionResult(
            success=True,
            output=result,
            metadata={
                "action": "generate",
                "chart_type": chart_type,
            },
        )
    
    def _format(self, task: Task) -> ExecutionResult:
        """
        格式化数据为可视化准备
        
        【输入参数】
        - raw_data: 原始数据
        - target_format: 目标格式 (bar/line/pie)
        
        【返回】
        - formatted_data: 格式化后的数据
        """
        raw_data = task.input_data.get("raw_data", [])
        target_format = task.input_data.get("target_format", "bar")
        
        # Mock 格式化逻辑
        # 实际应该智能地从原始数据提取标签和值
        formatted = {
            "labels": [],
            "values": [],
        }
        
        # 尝试从字典列表中提取数据
        if isinstance(raw_data, list) and len(raw_data) > 0:
            if isinstance(raw_data[0], dict):
                # 假设第一个字典的键是标签，值是数值
                first_row = raw_data[0]
                for key, value in first_row.items():
                    if key not in ("label", "name", "id"):
                        try:
                            formatted["values"].append(float(value))
                            formatted["labels"].append(str(key))
                        except (ValueError, TypeError):
                            pass
        
        return ExecutionResult(
            success=True,
            output=formatted,
            metadata={
                "action": "format",
                "target_format": target_format,
            },
        )


# ============================================================================
# 导出
# ============================================================================

__all__ = ["VisualizationAgent", "MockChartGenerator"]
