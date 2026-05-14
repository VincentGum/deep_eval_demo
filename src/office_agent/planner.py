"""
Planner Agent - 任务规划器，负责分析用户请求并生成任务计划

【模块概述】
Planner Agent 是 Office Agent 系统的核心组件之一，负责：
1. 语义分析 - 理解用户请求的深层含义
2. 意图分类 - 识别用户的真实意图
3. 能力映射 - 将意图映射到所需的能力
4. 任务生成 - 创建结构化的任务列表

【核心组件】

1. SemanticAnalyzer (语义分析器)
   - 使用 NLP 技术分析用户消息
   - 提取关键特征：动作词、对象名词、上下文等
   - 支持中英文双语分析

2. CapabilityMapper (能力映射器)
   - 将语义特征映射到 Agent 能力
   - 使用决策规则而非简单匹配
   - 支持复杂的多能力组合

3. TaskPlanGenerator (任务计划生成器)
   - 基于语义特征生成任务序列
   - 处理任务依赖关系
   - 确定任务优先级

4. MockReasoningModel (Mock 推理模型)
   - 模拟高级推理模型（如 o1）的思考过程
   - 生成结构化的推理步骤
   - 支持调试和解释

【设计原则】
1. 语义优先：避免关键词匹配，使用语义理解
2. 结构化推理：多步骤推理过程，可解释
3. 可扩展性：新意图类型和能力的低耦合扩展

【意图类型】
- INQUIRE: 查询/询问类请求
- MANIPULATE: 操作/处理类请求
- ANALYZE: 分析/统计类请求
- REPORT: 报告生成类请求
- COMMUNICATE: 沟通/通知类请求
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .base import (
    Task, TaskPlan, TaskPriority, AgentCapability,
    create_task_id
)


# ============================================================================
# 意图类型枚举
# ============================================================================

class IntentType(str, Enum):
    """
    意图类型枚举
    
    【类型说明】
    - INQUIRE: 查询类 - 用户想要获取信息
    - MANIPULATE: 操作类 - 用户想要执行某种操作
    - ANALYZE: 分析类 - 用户想要分析数据
    - REPORT: 报告类 - 用户想要生成报告
    - COMMUNICATE: 沟通类 - 用户想要与他人沟通
    """
    INQUIRE = "inquire"           # 查询
    MANIPULATE = "manipulate"     # 操作
    ANALYZE = "analyze"          # 分析
    REPORT = "report"             # 报告
    COMMUNICATE = "communicate"   # 沟通


# ============================================================================
# 语义特征数据类
# ============================================================================

@dataclass
class SemanticFeatures:
    """
    语义特征 - 存储分析得到的语义特征
    
    【字段说明】
    - primary_intent: 主要意图
    - secondary_intent: 次要意图（如果有）
    - action_verbs: 动作动词列表
    - object_nouns: 对象名词列表
    - data_sources: 数据来源
    - output_format: 输出格式
    - has_urgency: 是否有紧急标识
    - has_time_reference: 是否有时间参考
    - has_quantity: 是否有数量指标
    - has_quality_requirement: 是否有质量要求
    - is_multi_step: 是否为多步骤任务
    - requires_aggregation: 是否需要聚合
    - requires_visualization: 是否需要可视化
    - requires_collaboration: 是否需要协作
    """
    primary_intent: IntentType = IntentType.INQUIRE
    secondary_intent: IntentType | None = None
    
    action_verbs: list[str] = field(default_factory=list)
    object_nouns: list[str] = field(default_factory=list)
    
    data_sources: list[str] = field(default_factory=list)
    output_format: str = "text"
    
    has_urgency: bool = False
    has_time_reference: bool = False
    has_quantity: bool = False
    has_quality_requirement: bool = False
    
    is_multi_step: bool = False
    requires_aggregation: bool = False
    requires_visualization: bool = False
    requires_collaboration: bool = False


# ============================================================================
# 语义分析器
# ============================================================================

class SemanticAnalyzer:
    """
    语义分析器 - 分析用户消息的语义特征
    
    【分析流程】
    1. 分词和预处理
    2. 意图分类
    3. 动作模式提取
    4. 上下文检测
    5. 数据源识别
    6. 输出格式判断
    7. 复杂度评估
    """
    
    # 动作动词集合
    class ActionVerb:
        """动作动词集合"""
        INQUIRE_VERBS = {
            'find', 'search', 'look', 'get', 'check', 'show', 'display',
            'retrieve', 'fetch', 'query', 'obtain', '查询', '查找', '获取',
            'what', 'which', 'where', 'when', 'how'
        }
        MANIPULATE_VERBS = {
            'create', 'update', 'delete', 'modify', 'change', 'set', 'add',
            'generate', 'submit', 'send', '创建', '更新', '修改', '提交'
        }
        ANALYZE_VERBS = {
            'analyze', 'calculate', 'compute', 'sum', 'average', 'count',
            'compare', 'evaluate', 'assess', '统计', '分析', '计算'
        }
        REPORT_VERBS = {
            'report', 'summarize', 'compile', 'document', 'prepare', 'generate',
            '报告', '汇总', '整理', '生成'
        }
        COMMUNICATE_VERBS = {
            'send', 'email', 'notify', 'share', 'distribute', '通知', '发送', '分享'
        }
    
    # 紧迫性模式
    URGENCY_PATTERNS = [
        r'\burgent\b', r'\basap\b', r'\bimmediately\b', r'\b紧急\b',
        r'\bright now\b', r'\bASAP\b', r'\b尽快\b'
    ]
    
    # 时间参考模式
    TIME_PATTERNS = [
        r'\bthis week\b', r'\blast week\b', r'\bnext week\b',
        r'\bthis month\b', r'\bthis quarter\b', r'\bthis year\b',
        r'\b本周\b', r'\b本月\b', r'\b本季度\b', r'\b本周\b'
    ]
    
    # 数据源模式
    DATA_SOURCE_PATTERNS = {
        'api': [r'api', r'接口', r'rest', r'http'],
        'database': [r'database', r'db', r'data source', r'数据库'],
        'web': [r'website', r'web', r'url', r'网页', r'爬取', r'scrape'],
        'file': [r'file', r'document', r'csv', r'json', r'excel', r'文件'],
    }
    
    # 输出格式模式
    OUTPUT_FORMAT_PATTERNS = {
        'markdown': [r'markdown', r'md', r'report', r'报告', r'文档'],
        'table': [r'table', r'表格', r'csv', r'excel'],
        'chart': [r'chart', r'graph', r'plot', r'图表', r'可视化'],
        'json': [r'json', r'api response', r'json格式'],
        'pdf': [r'pdf'],
    }
    
    def analyze(self, text: str) -> SemanticFeatures:
        """
        分析文本并返回语义特征
        
        Args:
            text: 用户输入的文本
        
        Returns:
            SemanticFeatures 对象，包含分析得到的特征
        """
        # 预处理：转小写，分词
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        # 初始化特征对象
        features = SemanticFeatures()
        
        # 1. 基于动作动词分类意图
        self._classify_intent(words, text_lower, features)
        
        # 2. 提取动作模式
        self._extract_action_patterns(words, features)
        
        # 3. 检测上下文标识
        self._detect_context(text, text_lower, features)
        
        # 4. 识别数据来源
        self._identify_data_sources(text_lower, features)
        
        # 5. 确定输出格式
        self._determine_output_format(text_lower, features)
        
        # 6. 评估复杂度
        self._assess_complexity(words, text, features)
        
        return features
    
    def _classify_intent(
        self,
        words: set[str],
        text: str,
        features: SemanticFeatures
    ):
        """基于动作动词分析分类意图"""
        intent_scores = {}
        
        # 查询意图
        inquiry_count = len(words & self.ActionVerb.INQUIRE_VERBS)
        intent_scores[IntentType.INQUIRE] = inquiry_count
        
        # 操作意图
        manip_count = len(words & self.ActionVerb.MANIPULATE_VERBS)
        intent_scores[IntentType.MANIPULATE] = manip_count
        
        # 分析意图
        analyze_count = len(words & self.ActionVerb.ANALYZE_VERBS)
        intent_scores[IntentType.ANALYZE] = analyze_count
        
        # 报告意图
        report_count = len(words & self.ActionVerb.REPORT_VERBS)
        intent_scores[IntentType.REPORT] = report_count
        
        # 沟通意图
        comm_count = len(words & self.ActionVerb.COMMUNICATE_VERBS)
        intent_scores[IntentType.COMMUNICATE] = comm_count
        
        # 确定主要意图（得分最高）
        if intent_scores:
            max_score = max(intent_scores.values())
            if max_score > 0:
                # 优先级：REPORT > ANALYZE > MANIPULATE > COMMUNICATE > INQUIRE
                for intent in [IntentType.REPORT, IntentType.ANALYZE,
                              IntentType.MANIPULATE, IntentType.COMMUNICATE, IntentType.INQUIRE]:
                    if intent_scores.get(intent, 0) == max_score:
                        features.primary_intent = intent
                        break
        
        # 检查多意图（如 "analyze and report"）
        if 'and' in words or '&' in text:
            sorted_intents = sorted(intent_scores.items(), key=lambda x: -x[1])
            if len(sorted_intents) >= 2 and sorted_intents[1][1] > 0:
                features.secondary_intent = sorted_intents[1][0]
    
    def _extract_action_patterns(
        self,
        words: set[str],
        features: SemanticFeatures
    ):
        """提取动作动词和对象名词"""
        all_action_verbs = (
            self.ActionVerb.INQUIRE_VERBS |
            self.ActionVerb.MANIPULATE_VERBS |
            self.ActionVerb.ANALYZE_VERBS |
            self.ActionVerb.REPORT_VERBS |
            self.ActionVerb.COMMUNICATE_VERBS
        )
        features.action_verbs = list(words & all_action_verbs)
        
        # 业务对象名词
        business_objects = {
            'order', 'customer', 'sales', 'report', 'data', 'chart',
            'document', 'email', 'invoice', 'payment', 'product', 'user',
            '订单', '客户', '销售', '报告', '数据', '文档'
        }
        features.object_nouns = list(words & business_objects)
    
    def _detect_context(
        self,
        text: str,
        text_lower: str,
        features: SemanticFeatures
    ):
        """检测上下文标识"""
        # 紧迫性
        features.has_urgency = any(
            re.search(pattern, text_lower) for pattern in self.URGENCY_PATTERNS
        )
        
        # 时间参考
        features.has_time_reference = any(
            re.search(pattern, text_lower) for pattern in self.TIME_PATTERNS
        )
        
        # 数量指标
        quantity_patterns = [r'\d+', r'\b(sum|total|count|average|max|min)\b']
        features.has_quantity = any(
            re.search(pattern, text_lower) for pattern in quantity_patterns
        )
        
        # 质量要求
        word_set = set(text_lower.split())
        quality_words = {'accuracy', 'precision', 'detailed', 'complete', 'thorough',
                        '准确', '详细', '完整'}
        features.has_quality_requirement = bool(word_set & quality_words)
    
    def _identify_data_sources(
        self,
        text: str,
        features: SemanticFeatures
    ):
        """识别数据来源"""
        for source_type, patterns in self.DATA_SOURCE_PATTERNS.items():
            if any(re.search(p, text) for p in patterns):
                features.data_sources.append(source_type)
        
        # 默认使用 API
        if not features.data_sources:
            features.data_sources.append('api')
    
    def _determine_output_format(
        self,
        text: str,
        features: SemanticFeatures
    ):
        """确定输出格式"""
        for format_type, patterns in self.OUTPUT_FORMAT_PATTERNS.items():
            if any(re.search(p, text) for p in patterns):
                features.output_format = format_type
                break
        
        # 默认格式
        if not features.output_format:
            features.output_format = 'markdown'
    
    def _assess_complexity(
        self,
        words: set[str],
        text: str,
        features: SemanticFeatures
    ):
        """评估任务复杂度"""
        # 多步骤标识
        multi_step_words = {'and', 'then', 'also', 'plus', 'after', 'before', 'first', 'next',
                           '然后', '并且', '接下来', '首先'}
        features.is_multi_step = bool(words & multi_step_words) or len(words) > 15
        
        # 聚合需求（显式）
        agg_words = {'sum', 'total', 'average', 'count', 'aggregate', 'summarize',
                    '统计', '汇总', '求和', '平均'}
        explicit_agg = bool(words & agg_words)
        
        # 聚合需求（隐式推断）
        report_types_needing_agg = {
            'sales', 'revenue', 'performance', 'summary',
            'weekly', 'monthly', 'quarterly', 'annual',
            '销售', '收入', '每周', '每月', '季度', '年度'
        }
        inferred_agg = bool(words & report_types_needing_agg) and \
                       features.primary_intent in {IntentType.REPORT, IntentType.ANALYZE}
        
        # 可视化需求（显式）
        viz_words = {'chart', 'graph', 'plot', 'visual', 'pie', 'bar', 'line',
                    '图表', '图形', '可视化'}
        explicit_viz = bool(words & viz_words)
        
        # 可视化需求（隐式推断）
        inferred_viz = features.primary_intent == IntentType.REPORT and explicit_agg
        
        features.requires_aggregation = explicit_agg or inferred_agg
        features.requires_visualization = explicit_viz or inferred_viz
        
        # 协作需求
        collab_words = {'email', 'send', 'notify', 'share', 'team', 'colleague',
                        '邮件', '发送', '通知', '团队'}
        features.requires_collaboration = bool(words & collab_words)


# ============================================================================
# 能力映射器
# ============================================================================

class CapabilityMapper:
    """
    能力映射器 - 将语义特征映射到所需能力
    
    【映射规则】
    1. 主要意图决定基础能力
    2. 辅助特征添加额外能力
    3. 数据来源影响能力选择
    """
    
    def map_to_capabilities(self, features: SemanticFeatures) -> list[str]:
        """
        将语义特征映射到能力列表
        
        Args:
            features: 语义特征
        
        Returns:
            所需能力的字符串列表
        """
        capabilities = []
        capability_set = set()
        
        # 1. 基于主要意图映射基础能力
        intent_capability_map = {
            IntentType.INQUIRE: [
                AgentCapability.API_CALL,
                AgentCapability.BROWSER_SCRAPE,
                AgentCapability.DATA_QUERY,
            ],
            IntentType.MANIPULATE: [
                AgentCapability.DOC_WRITE,
                AgentCapability.API_CALL,
            ],
            IntentType.ANALYZE: [
                AgentCapability.DATA_QUERY,
                AgentCapability.DATA_TRANSFORM,
                AgentCapability.DATA_AGGREGATE,
            ],
            IntentType.REPORT: [
                AgentCapability.DATA_AGGREGATE,
                AgentCapability.REPORT_GENERATE,
            ],
            IntentType.COMMUNICATE: [
                AgentCapability.EMAIL_SEND,
            ],
        }
        
        base_caps = intent_capability_map.get(features.primary_intent, [])
        for cap in base_caps:
            if cap.value not in capability_set:
                capabilities.append(cap.value)
                capability_set.add(cap.value)
        
        # 2. 基于辅助特征添加能力
        if features.requires_aggregation and AgentCapability.DATA_AGGREGATE.value not in capability_set:
            capabilities.append(AgentCapability.DATA_AGGREGATE.value)
            capability_set.add(AgentCapability.DATA_AGGREGATE.value)
        
        if features.requires_visualization and AgentCapability.CHART_CREATE.value not in capability_set:
            capabilities.append(AgentCapability.CHART_CREATE.value)
        
        if features.output_format == 'table' and AgentCapability.TABLE_CREATE.value not in capability_set:
            capabilities.append(AgentCapability.TABLE_CREATE.value)
        
        if features.requires_collaboration and AgentCapability.EMAIL_SEND.value not in capability_set:
            capabilities.append(AgentCapability.EMAIL_SEND.value)
        
        # 3. 添加数据源能力
        for source in features.data_sources:
            source_capability_map = {
                'api': AgentCapability.API_CALL,
                'database': AgentCapability.DATA_QUERY,
                'web': AgentCapability.BROWSER_SCRAPE,
                'file': AgentCapability.DOC_READ,
            }
            cap = source_capability_map.get(source)
            if cap and cap.value not in capability_set:
                capabilities.append(cap.value)
        
        return capabilities


# ============================================================================
# 任务计划生成器
# ============================================================================

class TaskPlanGenerator:
    """
    任务计划生成器 - 根据语义特征生成任务列表
    
    【生成流程】
    1. 确定任务序列
    2. 创建任务对象
    3. 设置依赖关系
    4. 添加聚合/报告任务
    """
    
    def __init__(self):
        self.capability_mapper = CapabilityMapper()
    
    def generate_tasks(
        self,
        features: SemanticFeatures,
        user_request: str
    ) -> list[dict[str, Any]]:
        """
        生成任务列表
        
        Args:
            features: 语义特征
            user_request: 原始用户请求
        
        Returns:
            任务定义字典列表
        """
        tasks = []
        task_counter = [1]
        
        # 1. 确定任务序列
        task_sequence = self._determine_task_sequence(features)
        
        # 2. 生成任务
        for task_def in task_sequence:
            task = self._create_task(
                task_def,
                features,
                user_request,
                task_counter
            )
            if task:
                tasks.append(task)
        
        # 3. 添加聚合任务（如果需要）
        if features.requires_aggregation and len(tasks) > 1:
            agg_task = self._create_aggregation_task(features, tasks, task_counter)
            if agg_task:
                tasks.append(agg_task)
        
        # 4. 添加报告任务（如果需要）
        if features.output_format in ['markdown', 'pdf'] or features.primary_intent == IntentType.REPORT:
            report_task = self._create_report_task(features, tasks, task_counter)
            if report_task:
                tasks.append(report_task)
        
        return tasks
    
    def _determine_task_sequence(self, features: SemanticFeatures) -> list[dict]:
        """确定任务序列"""
        sequence = []
        
        # 数据收集阶段 - 报告类也需要先收集数据
        if features.primary_intent in [IntentType.INQUIRE, IntentType.ANALYZE, IntentType.REPORT]:
            for source in features.data_sources:
                if source == 'api':
                    sequence.append({'capability': AgentCapability.API_CALL, 'phase': 'collect'})
                elif source == 'web':
                    sequence.append({'capability': AgentCapability.BROWSER_NAVIGATE, 'phase': 'collect'})
                    sequence.append({'capability': AgentCapability.BROWSER_SCRAPE, 'phase': 'collect'})
                elif source == 'database':
                    sequence.append({'capability': AgentCapability.DATA_QUERY, 'phase': 'collect'})
        
        # 处理阶段
        if features.primary_intent == IntentType.ANALYZE:
            sequence.append({'capability': AgentCapability.DATA_TRANSFORM, 'phase': 'process'})
            if features.requires_aggregation:
                sequence.append({'capability': AgentCapability.DATA_AGGREGATE, 'phase': 'process'})
        
        # 可视化阶段
        if features.requires_visualization:
            sequence.append({'capability': AgentCapability.CHART_CREATE, 'phase': 'visualize'})
        
        # 输出阶段
        if features.output_format == 'table':
            sequence.append({'capability': AgentCapability.TABLE_CREATE, 'phase': 'output'})
        elif features.primary_intent == IntentType.REPORT:
            sequence.append({'capability': AgentCapability.REPORT_GENERATE, 'phase': 'output'})
        
        # 沟通阶段
        if features.requires_collaboration:
            sequence.append({'capability': AgentCapability.EMAIL_SEND, 'phase': 'communicate'})
        
        # 默认任务
        if not sequence:
            sequence.append({'capability': AgentCapability.API_CALL, 'phase': 'default'})
        
        return sequence
    
    def _create_task(
        self,
        task_def: dict,
        features: SemanticFeatures,
        user_request: str,
        task_counter: list[int]
    ) -> dict[str, Any] | None:
        """创建单个任务"""
        capability = task_def['capability']
        
        # 任务描述模板
        descriptions = {
            AgentCapability.API_CALL: f"Call API to retrieve {', '.join(features.data_sources) if features.data_sources else 'data'}",
            AgentCapability.BROWSER_SCRAPE: "Navigate to and scrape data from web source",
            AgentCapability.DATA_QUERY: "Query data from database",
            AgentCapability.DATA_TRANSFORM: "Transform and process collected data",
            AgentCapability.DATA_AGGREGATE: "Aggregate and calculate statistics",
            AgentCapability.CHART_CREATE: f"Create {features.output_format or 'chart'} visualization",
            AgentCapability.TABLE_CREATE: "Format data into table",
            AgentCapability.REPORT_GENERATE: f"Generate {features.output_format or 'report'} report",
            AgentCapability.EMAIL_SEND: "Send notification via email",
        }
        
        task_id = f"task_{task_counter[0]:03d}"
        task_counter[0] += 1
        
        # 确定优先级
        priority = TaskPriority.MEDIUM
        if features.has_urgency:
            priority = TaskPriority.URGENT
        elif features.primary_intent == IntentType.REPORT:
            priority = TaskPriority.HIGH
        
        return {
            'id': task_id,
            'description': descriptions.get(capability, f"Execute {capability.value}"),
            'capability': capability,
            'priority': priority,
            'phase': task_def['phase'],
        }
    
    def _create_aggregation_task(
        self,
        features: SemanticFeatures,
        tasks: list[dict],
        task_counter: list[int]
    ) -> dict | None:
        """创建聚合任务"""
        task_id = f"task_{task_counter[0]:03d}"
        task_counter[0] += 1
        
        return {
            'id': task_id,
            'description': 'Aggregate data from previous tasks',
            'capability': AgentCapability.DATA_AGGREGATE,
            'priority': TaskPriority.MEDIUM,
            'depends_on': [t['id'] for t in tasks if t.get('phase') == 'collect'],
            'phase': 'aggregate',
        }
    
    def _create_report_task(
        self,
        features: SemanticFeatures,
        tasks: list[dict],
        task_counter: list[int]
    ) -> dict | None:
        """创建报告任务"""
        task_id = f"task_{task_counter[0]:03d}"
        task_counter[0] += 1
        
        return {
            'id': task_id,
            'description': f"Generate {features.output_format} report",
            'capability': AgentCapability.REPORT_GENERATE,
            'priority': TaskPriority.HIGH,
            'depends_on': [t['id'] for t in tasks],
            'phase': 'report',
        }


# ============================================================================
# Planner Agent
# ============================================================================

class PlannerAgent:
    """
    Planner Agent - 主规划器
    
    协调语义分析、能力映射和任务生成，
    输出完整的任务计划。
    """
    
    def __init__(self):
        self.analyzer = SemanticAnalyzer()
        self.capability_mapper = CapabilityMapper()
        self.task_generator = TaskPlanGenerator()
    
    def create_plan(
        self,
        user_request: str,
        context: dict[str, Any] | None = None
    ) -> tuple[TaskPlan, list[str]]:
        """
        创建任务计划
        
        Args:
            user_request: 用户请求
            context: 额外上下文
        
        Returns:
            (TaskPlan, reasoning_steps) 元组
        """
        reasoning_steps = []
        
        # 1. 语义分析
        features = self.analyzer.analyze(user_request)
        reasoning_steps.append(f"Intent: {features.primary_intent.value}")
        reasoning_steps.append(f"Data sources: {features.data_sources}")
        reasoning_steps.append(f"Output format: {features.output_format}")
        
        # 2. 能力映射
        capabilities = self.capability_mapper.map_to_capabilities(features)
        reasoning_steps.append(f"Capabilities: {capabilities}")
        
        # 3. 生成任务
        task_defs = self.task_generator.generate_tasks(features, user_request)
        
        # 4. 创建任务对象
        tasks = []
        task_id_map = {}
        
        for task_def in task_defs:
            task = Task(
                id=task_def['id'],
                description=task_def['description'],
                capability_required=task_def['capability'].value,
                priority=task_def.get('priority', TaskPriority.MEDIUM),
                depends_on=task_def.get('depends_on', []),
            )
            tasks.append(task)
            task_id_map[task_def['id']] = task
        
        # 更新依赖引用
        for task in tasks:
            task.depends_on = [
                task_id_map[dep_id].id
                for dep_id in task.depends_on
                if dep_id in task_id_map
            ]
        
        # 5. 创建计划
        plan = TaskPlan(
            id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_request=user_request,
            tasks=tasks,
            context=context or {},
        )
        
        reasoning_steps.append(f"Total tasks: {len(tasks)}")
        
        return plan, reasoning_steps
    
    def explain_plan(self, plan: TaskPlan) -> str:
        """解释任务计划"""
        lines = [
            f"Task Plan: {plan.id}",
            f"User Request: {plan.user_request}",
            "",
            "Tasks:",
        ]
        
        for task in plan.tasks:
            deps = f" (depends on: {', '.join(task.depends_on)})" if task.depends_on else ""
            lines.append(f"  {task.id}: {task.description} [{task.capability_required}]{deps}")
        
        return "\n".join(lines)


# ============================================================================
# Mock 推理模型
# ============================================================================

class MockReasoningModel:
    """
    Mock 推理模型 - 模拟高级推理模型的思考过程
    
    用于在没有真实 LLM 的情况下展示推理步骤。
    """
    
    def __init__(self):
        self.analyzer = SemanticAnalyzer()
    
    def reason(
        self,
        user_request: str
    ) -> tuple[SemanticFeatures, list[str]]:
        """
        执行推理
        
        Args:
            user_request: 用户请求
        
        Returns:
            (features, reasoning_steps) 元组
        """
        features = self.analyzer.analyze(user_request)
        
        reasoning = [
            "Step 1: Analyzing user request...",
            f"  - Primary intent: {features.primary_intent.value}",
            f"  - Data sources: {features.data_sources}",
            f"  - Output format: {features.output_format}",
            "",
            "Step 2: Assessing complexity...",
            f"  - Multi-step: {features.is_multi_step}",
            f"  - Requires aggregation: {features.requires_aggregation}",
            f"  - Requires visualization: {features.requires_visualization}",
            "",
            "Step 3: Determining capabilities...",
        ]
        
        capability_mapper = CapabilityMapper()
        capabilities = capability_mapper.map_to_capabilities(features)
        
        for cap in capabilities:
            reasoning.append(f"  - {cap}")
        
        return features, reasoning


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    "IntentType",
    "SemanticFeatures",
    "SemanticAnalyzer",
    "CapabilityMapper",
    "TaskPlanGenerator",
    "PlannerAgent",
    "MockReasoningModel",
]
