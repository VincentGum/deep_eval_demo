"""
文档 Agent - 处理文档读写任务

【功能说明】
DocAgent 负责：
1. 文档读取 - 读取各种格式的文档
2. 文档写入 - 写入文档
3. 文档解析 - 解析文档内容
4. 格式转换 - 在不同格式间转换

【Mock 实现】
本模块使用 Mock 实现，包含预定义的文档模板。
实际使用时替换为真实的文件系统和文档处理库。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..base import (
    BaseSubAgent,
    ExecutionResult,
    Task,
    AgentCapability,
)


# ============================================================================
# Mock 文件系统
# ============================================================================

class MockFileSystem:
    """
    Mock 文件系统 - 模拟文件系统操作
    
    【预定义文件】
    - /documents/report_template.md: 报告模板
    - /documents/meeting_notes.md: 会议笔记模板
    - /documents/sales_data.json: 销售数据
    """
    
    _files: dict[str, str] = {
        "/documents/report_template.md": """# Weekly Report Template

## Summary
{summary}

## Metrics
{metrics}

## Action Items
{action_items}
""",
        "/documents/meeting_notes.md": """# Meeting Notes - {date}

## Attendees
{attendees}

## Agenda
{agenda}

## Notes
{notes}

## Next Steps
{next_steps}
""",
        "/documents/sales_data.json": json.dumps({
            "product": "Widget A",
            "units_sold": 4600,
            "revenue": 137954,
            "regions": ["North", "South", "East", "West"],
        }, indent=2),
    }
    
    def __init__(self):
        self._written_files: dict[str, str] = {}
    
    def read(self, path: str) -> str | None:
        """
        读取文件
        
        Args:
            path: 文件路径
        
        Returns:
            文件内容，如果不存在返回 None
        """
        return self._files.get(path) or self._written_files.get(path)
    
    def write(self, path: str, content: str) -> bool:
        """
        写入文件
        
        Args:
            path: 文件路径
            content: 文件内容
        
        Returns:
            是否成功
        """
        self._written_files[path] = content
        return True
    
    def list_files(self, directory: str = "/documents") -> list[str]:
        """
        列出目录下的文件
        
        Args:
            directory: 目录路径
        
        Returns:
            文件路径列表
        """
        all_files = list(self._files.keys()) + list(self._written_files.keys())
        return [f for f in all_files if f.startswith(directory)]


# ============================================================================
# 文档 Agent
# ============================================================================

class DocAgent(BaseSubAgent):
    """
    文档 Agent - 处理文档操作任务
    
    【能力列表】
    - DOC_READ: 读取文档
    - DOC_WRITE: 写入文档
    - DOC_PARSE: 解析文档
    - DOC_CONVERT: 转换文档格式
    """
    
    def __init__(self):
        self._fs = MockFileSystem()
    
    @property
    def name(self) -> str:
        return "doc_agent"
    
    @property
    def capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability.DOC_READ,
            AgentCapability.DOC_WRITE,
            AgentCapability.DOC_PARSE,
            AgentCapability.DOC_CONVERT,
        ]
    
    def can_handle(self, task: Task) -> bool:
        """检查是否能处理任务"""
        capability = task.capability_required
        if isinstance(capability, str):
            capability = AgentCapability(capability)
        return capability in self.capabilities
    
    def execute(self, task: Task) -> ExecutionResult:
        """
        执行文档任务
        
        【任务路由】
        - DOC_READ → _read()
        - DOC_WRITE → _write()
        - DOC_PARSE → _parse()
        - DOC_CONVERT → _convert()
        """
        try:
            capability = task.capability_required
            if isinstance(capability, str):
                capability = AgentCapability(capability)
            
            if capability == AgentCapability.DOC_READ:
                return self._read(task)
            elif capability == AgentCapability.DOC_WRITE:
                return self._write(task)
            elif capability == AgentCapability.DOC_PARSE:
                return self._parse(task)
            elif capability == AgentCapability.DOC_CONVERT:
                return self._convert(task)
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
    
    def _read(self, task: Task) -> ExecutionResult:
        """
        读取文档
        
        【输入参数】
        - path 或 file_path: 文件路径
        
        【返回】
        - path: 文件路径
        - content: 文件内容
        - size: 文件大小
        """
        # 兼容 path 和 file_path
        path = task.input_data.get("path") or task.input_data.get("file_path", "")
        if not path:
            return ExecutionResult(
                success=False,
                error="No file path provided",
            )
        
        content = self._fs.read(path)
        if content is None:
            return ExecutionResult(
                success=False,
                error=f"File not found: {path}",
            )
        
        return ExecutionResult(
            success=True,
            output={
                "path": path,
                "content": content,
                "size": len(content),
            },
            metadata={"action": "read"},
        )
    
    def _write(self, task: Task) -> ExecutionResult:
        """
        写入文档
        
        【输入参数】
        - path 或 file_path: 文件路径
        - content: 文件内容
        - template_data: 模板数据（可选）
        """
        # 兼容 path 和 file_path
        path = task.input_data.get("path") or task.input_data.get("file_path", "")
        content = task.input_data.get("content", "")
        template_data = task.input_data.get("template_data", {})
        
        if not path:
            return ExecutionResult(
                success=False,
                error="No file path provided",
            )
        
        # 如果提供模板数据，格式化内容
        if template_data and "{content}" in content:
            content = content.format(**template_data)
        
        success = self._fs.write(path, content)
        
        return ExecutionResult(
            success=success,
            output={
                "path": path,
                "written": success,
                "size": len(content) if success else 0,
            },
            metadata={"action": "write"},
        )
    
    def _parse(self, task: Task) -> ExecutionResult:
        """
        解析文档
        
        【输入参数】
        - content: 文档内容
        - format: 格式类型 (auto/json/markdown)
        """
        content = task.input_data.get("content", "")
        format_type = task.input_data.get("format", "auto")
        
        parsed = {}
        # 尝试解析 JSON
        if format_type in ("auto", "json") and content.strip().startswith("{"):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                pass
        
        if not parsed:
            # 返回原始内容
            parsed = {"content": content, "format": "text"}
        
        return ExecutionResult(
            success=True,
            output=parsed,
            metadata={"action": "parse"},
        )
    
    def _convert(self, task: Task) -> ExecutionResult:
        """
        转换文档格式
        
        【输入参数】
        - content: 文档内容
        - from_format: 源格式
        - to_format: 目标格式
        
        【支持转换】
        - markdown → json
        - json → markdown
        - json → html
        """
        content = task.input_data.get("content", "")
        from_format = task.input_data.get("from_format", "markdown")
        to_format = task.input_data.get("to_format", "json")
        
        converted = {"original_format": from_format, "converted_format": to_format}
        
        # Markdown → JSON
        if from_format == "markdown" and to_format == "json":
            # 简单的 Markdown 标题提取
            lines = content.split("\n")
            data = {}
            for line in lines:
                if line.startswith("#"):
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        data[parts[1].strip()] = ""
            converted["data"] = data
            converted["content"] = json.dumps(data, ensure_ascii=False)
        
        # JSON → Markdown
        elif from_format == "json" and to_format == "markdown":
            try:
                data = json.loads(content) if isinstance(content, str) else content
                lines = ["# Generated Report"]
                for key, value in data.items():
                    lines.append(f"\n## {key}")
                    lines.append(str(value))
                converted["content"] = "\n".join(lines)
            except json.JSONDecodeError:
                converted["error"] = "Invalid JSON"
        
        else:
            converted["content"] = content
        
        return ExecutionResult(
            success="error" not in converted,
            output=converted,
            metadata={"action": "convert"},
        )


# ============================================================================
# 导出
# ============================================================================

__all__ = ["DocAgent", "MockFileSystem"]
