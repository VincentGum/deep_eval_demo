"""Office Agent 评测 Rubrics 和指标定义。

本文件定义了 Office Agent 的评测指标和评分标准（Rubrics）。
使用 DeepEval 原生的 G-Eval 和自定义指标。

Office Agent 的核心评测维度：
1. 任务规划质量 - Planner 是否正确分解任务
2. 子 Agent 调用合理性 - Executor 是否正确调度 Sub Agents
3. 任务执行准确性 - 各 Sub Agent 是否正确完成任务
4. 结果验证质量 - Verify Agent 是否正确评估进度
5. Human-in-the-Loop 处理 - 人工介入是否及时合理
"""

# =============================================================================
# G-Eval Rubrics: 任务规划质量评估
# =============================================================================

PLANNING_QUALITY_RUBRIC = """
评估 Office Agent 任务规划的质量，评分标准：

1. 任务分解完整性 (35%): 是否将复杂请求分解为可执行的子任务
   - 识别所有必要的操作步骤
   - 每个子任务都有明确的目标

2. 依赖关系处理 (25%): 是否正确识别任务间的依赖关系
   - 数据获取必须在数据处理之前
   - 图表生成必须在数据分析之后

3. Agent 类型匹配 (25%): 是否为每个子任务选择了正确的 Agent 类型
   - Browser Agent: 网页浏览、数据爬取
   - API Agent: 外部 API 调用
   - Data Agent: 数据处理、统计分析
   - Doc Agent: 文档读写
   - Visualization Agent: 图表生成

4. 并行化优化 (15%): 是否充分利用了任务的并行性
   - 无依赖任务应并行执行
   - 合理安排执行顺序

评分范围: 1-5 分 (1=差, 3=一般, 5=优秀)
"""

# =============================================================================
# G-Eval Rubrics: 子 Agent 调用合理性评估
# =============================================================================

AGENT_DISPATCH_RUBRIC = """
评估 Executor 对子 Agent 调度的合理性，评分标准：

1. 调用时机 (30%): 是否在合适的时机调用子 Agent
   - 依赖任务完成后才调用
   - 避免过早或过晚调用

2. 并行执行 (25%): 是否充分利用了并行执行能力
   - 无依赖任务同时执行
   - 合理使用 ThreadPoolExecutor

3. 资源共享 (25%): 是否正确管理任务间的数据传递
   - 输出作为后续任务的输入
   - 避免重复获取相同数据

4. 错误处理 (20%): 是否妥善处理子 Agent 的执行错误
   - 重试机制
   - 优雅降级

评分范围: 1-5 分
"""

# =============================================================================
# G-Eval Rubrics: 任务执行准确性评估
# =============================================================================

EXECUTION_ACCURACY_RUBRIC = """
评估子 Agent 任务执行的准确性，评分标准：

1. 功能正确性 (35%): 是否正确完成了指定任务
   - Browser Agent: 正确访问和提取网页内容
   - API Agent: 正确调用和解析 API 响应
   - Data Agent: 正确进行数据处理和计算
   - Doc Agent: 正确读写文档内容
   - Visualization Agent: 正确生成图表

2. 数据准确性 (30%): 返回的数据/内容是否准确
   - 无语法错误
   - 格式正确
   - 内容符合预期

3. 完整性 (20%): 是否返回了所有必要的信息
   - 包含状态/元数据
   - 包含错误信息（如有）

4. 效率 (15%): 是否在合理时间内完成
   - 超时处理
   - 资源使用合理

评分范围: 1-5 分
"""

# =============================================================================
# G-Eval Rubrics: 结果验证质量评估
# =============================================================================

VERIFICATION_QUALITY_RUBRIC = """
评估 Verify Agent 验证结果的质量，评分标准：

1. 进度评估准确性 (35%): 是否正确评估了任务完成进度
   - 识别已完成的任务
   - 识别进行中的任务
   - 识别未开始的任务

2. 缺失识别 (30%): 是否正确识别了缺失信息
   - 检测不完整的结果
   - 识别矛盾的数据
   - 发现逻辑错误

3. 决策正确性 (20%): 是否做出正确的验证决策
   - 通过验证的条件
   - 需要补充信息的条件
   - 需要人工介入的条件

4. 反馈质量 (15%): 提供的反馈是否有价值
   - 清晰指出问题
   - 提供改进建议

评分范围: 1-5 分
"""

# =============================================================================
# G-Eval Rubrics: Human-in-the-Loop 处理评估
# =============================================================================

HUMAN_LOOP_RUBRIC = """
评估 Human-in-the-Loop 机制的处理质量，评分标准：

1. 介入时机 (35%): 是否在合适的时机请求人工介入
   - 任务无法自动完成时
   - 需要额外信息时
   - 涉及敏感操作时

2. 信息呈现 (30%): 是否清晰地向人工呈现需要决策的信息
   - 任务上下文
   - 可用的操作选项
   - 风险提示

3. 响应处理 (20%): 是否正确处理人工的响应
   - 接受人工输入
   - 更新任务状态
   - 继续执行流程

4. 超时处理 (15%): 是否妥善处理超时情况
   - 默认行为明确
   - 用户体验友好

评分范围: 1-5 分
"""

# =============================================================================
# G-Eval Rubrics: 最终输出质量评估
# =============================================================================

OUTPUT_QUALITY_RUBRIC = """
评估 Office Agent 最终输出内容的质量，评分标准：

1. 内容完整性 (30%): 是否包含所有预期的内容
   - 报告包含所有章节
   - 数据完整
   - 图表齐全

2. 格式规范性 (25%): 输出格式是否符合要求
   - Markdown 格式正确
   - 表格格式规范
   - 图表格式正确

3. 可读性 (25%): 内容是否易于理解
   - 语言清晰
   - 结构合理
   - 重点突出

4. 准确性 (20%): 内容是否准确无误
   - 数据正确
   - 结论合理
   - 无明显错误

评分范围: 1-5 分
"""

# =============================================================================
# DeepEval 内置指标配置
# =============================================================================

# Office Agent 评测指标配置
OFFICE_AGENT_METRICS = {
    # 内置指标 (需要 LLM API)
    "answer_relevancy": {
        "class": "AnswerRelevancyMetric",
        "threshold": 0.3,
        "params": {"strict_mode": False}
    },
    "faithfulness": {
        "class": "FaithfulnessMetric",
        "threshold": 0.5,
        "params": {"strict_mode": False}
    },

    # 自定义指标 (离线模式)
    "planning_quality": {
        "class": "PlanningQualityMetric",
        "threshold": 0.5,
        "params": {}
    },
    "agent_dispatch": {
        "class": "AgentDispatchMetric",
        "threshold": 0.5,
        "params": {}
    },
    "execution_accuracy": {
        "class": "ExecutionAccuracyMetric",
        "threshold": 0.5,
        "params": {}
    },
    "verification_quality": {
        "class": "VerificationQualityMetric",
        "threshold": 0.5,
        "params": {}
    },
    "human_loop_handling": {
        "class": "HumanLoopHandlingMetric",
        "threshold": 0.5,
        "params": {}
    },
    "output_quality": {
        "class": "OutputQualityMetric",
        "threshold": 0.5,
        "params": {}
    },
}

# =============================================================================
# 测试场景分类
# =============================================================================

# Office Agent 测试场景定义
OFFICE_AGENT_SCENARIOS = {
    "weekly_sales_report": {
        "description": "生成周销售报告",
        "typical_workflow": ["api_fetch", "data_process", "visualize", "doc_write"],
        "expected_agents": ["API Agent", "Data Agent", "Visualization Agent", "Doc Agent"],
        "complexity": "medium",
    },
    "customer_research": {
        "description": "客户调研任务",
        "typical_workflow": ["web_browse", "data_process", "report_write"],
        "expected_agents": ["Browser Agent", "Data Agent", "Doc Agent"],
        "complexity": "medium",
    },
    "meeting_preparation": {
        "description": "会议准备工作",
        "typical_workflow": ["data_query", "doc_format"],
        "expected_agents": ["Data Agent", "Doc Agent"],
        "complexity": "low",
    },
}

# =============================================================================
# G-Eval 测试配置
# =============================================================================

# G-Eval 配置（需要 OpenAI API Key）
G_EVAL_CONFIG = {
    "model": "gpt-4o",
    "temperature": 0.0,
    "enable_coherence": True,
    "enable_fluency": True,
    "enable_safety": True,
    "enable_g1": True,
}

# G-Eval 指标列表（需要 OpenAI API）
G_EVAL_METRICS = [
    "planning_quality_geval",
    "agent_dispatch_geval",
    "execution_accuracy_geval",
    "verification_quality_geval",
    "human_loop_handling_geval",
    "output_quality_geval",
]
