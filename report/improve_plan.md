现在我设计完整的 **EDD（Eval Driven Development）方案**：

---

# EDD 方案设计

## 一、DeepEval 评测能力全景

### 1.1 支持的评测指标分类

| 类别 | 指标 | 说明 | 适用 Agent |
|------|------|------|------------|
| **LLM Output** | AnswerRelevancyMetric | 答案相关性 | Both |
| | FaithfulnessMetric | 忠诚度（无幻觉） | Both |
| | SummarizationMetric | 摘要质量 | Both |
| **Agent Performance** | ToolUseMetric | 工具使用正确性 | Both |
| | TaskCompletionMetric | 任务完成度 | Office Agent |
| | GoalAccuracyMetric | 目标达成准确率 | Both |
| | PlanAdherenceMetric | 计划执行遵从度 | Office Agent |
| | PlanQualityMetric | 计划质量评估 | Office Agent |
| **Context & RAG** | ContextualPrecisionMetric | 上下文精确度 | Both |
| | ContextualRecallMetric | 上下文召回率 | Both |
| | ContextualRelevancyMetric | 上下文相关性 | Both |
| **Safety & Policy** | BiasMetric | 偏见检测 | Both |
| | ToxicityMetric | 有害内容检测 | Both |
| | PIILeakageMetric | PII 泄露检测 | Both |
| | HallucinationMetric | 幻觉检测 | Both |
| **Code & Data** | JsonCorrectnessMetric | JSON 格式正确性 | Both |
| | ExactMatchMetric | 精确匹配 | Both |
| | PatternMatchMetric | 模式匹配 | Both |
| **Role Play** | RoleAdherenceMetric | 角色一致性 | Customer Agent |
| | RoleViolationMetric | 角色违规检测 | Customer Agent |
| **G-Eval** | GEval | 自定义 LLM 评估 | Both |
| | ArenaGEval | 对抗性评估 | Both |
| | ConversationalGEval | 多轮对话评估 | Both |

### 1.2 计划使用的指标

**Customer Agent（8 个指标）：**
```
✓ AnswerRelevancyMetric     - 答案相关性
✓ FaithfulnessMetric        - 忠诚度
✓ ToolUseMetric            - 工具使用
✓ RoleAdherenceMetric     - 角色一致性
✓ BiasMetric               - 偏见检测
✓ ToxicityMetric           - 有害内容
✓ HallucinationMetric      - 幻觉检测
✓ GEval (自定义)           - 响应质量评估
```

**Office Agent（10 个指标）：**
```
✓ TaskCompletionMetric     - 任务完成度
✓ GoalAccuracyMetric       - 目标达成
✓ PlanAdherenceMetric      - 计划遵从度
✓ PlanQualityMetric         - 计划质量
✓ ToolUseMetric            - 工具使用
✓ AnswerRelevancyMetric     - 答案相关性
✓ ContextualPrecisionMetric - 上下文精确度
✓ ContextualRelevancyMetric - 上下文相关性
✓ GEval (自定义)            - 任务进度评估
✓ GEval (自定义)            - 子任务质量评估
```

---

## 二、Office Agent 差距分析

### 2.1 当前实现 vs 工业级需求

| 维度 | 当前实现 | 工业级需求 | 差距等级 |
|------|----------|------------|----------|
| **Planner** | 关键词匹配 Mock | 真正的推理模型 + RAG | 🔴 高 |
| **Executor** | 简单拓扑排序 | 失败重试 + 降级 + 超时 | 🟡 中 |
| **Verify** | 关键词验证 | 多维度验证 + 置信度 | 🔴 高 |
| **Human-in-Loop** | 简单超时 | 优先级 + 反馈 + 重试 | 🟡 中 |
| **Sub Agents** | Mock 实现 | 真实 API + 错误处理 | 🔴 高 |
| **Memory** | 无 | 会话管理 + 长期记忆 | 🔴 高 |
| **Observability** | 简单打印 | 日志 + 追踪 + 监控 | 🟡 中 |
| **Multi-turn** | 单轮 | 多轮上下文 + 对话管理 | 🟡 中 |

### 2.2 详细差距清单

#### 🔴 高优先级差距

**1. Planner Agent**
- [ ] 缺乏真正的推理能力（当前仅关键词匹配）
- [ ] 无法理解复杂意图和上下文
- [ ] 计划质量无法量化评估
- [ ] 缺乏动态调整计划能力

**2. Verify Agent**
- [ ] 仅基于关键词匹配验证
- [ ] 缺乏多维度验证（格式、质量、完整性）
- [ ] 无法处理模糊/部分完成状态
- [ ] 验证过程不透明

**3. Sub Agents（真实能力）**
- [ ] Browser Agent：缺乏真实浏览器自动化
- [ ] API Agent：缺乏真实 API 调用能力
- [ ] Doc Agent：缺乏真实文档处理
- [ ] Data Agent：缺乏真实数据源
- [ ] Visualization Agent：缺乏真实图表生成

**4. Memory System**
- [ ] 无会话状态管理
- [ ] 无跨任务上下文传递
- [ ] 无长期记忆能力
- [ ] 无用户偏好学习

#### 🟡 中优先级差距

**5. Executor**
- [ ] 缺乏任务失败重试机制
- [ ] 缺乏降级策略
- [ ] 缺乏资源限制管理
- [ ] 缺乏任务优先级队列

**6. Human-in-Loop**
- [ ] 超时机制过于简单
- [ ] 缺乏输入验证
- [ ] 缺乏反馈机制
- [ ] 缺乏优先级排序

**7. Observability**
- [ ] 缺乏结构化日志
- [ ] 缺乏执行追踪
- [ ] 缺乏性能指标
- [ ] 缺乏错误分类

**8. Multi-turn Conversation**
- [ ] 仅支持单轮请求
- [ ] 无对话历史管理
- [ ] 无上下文累积
- [ ] 无对话状态机

---

## 三、EDD 开发计划

### 3.1 EDD 流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        EDD 流程                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ 1. 设计 Eval │───▶│ 2. 运行评测 │───▶│ 3. 分析结果 │      │
│  │  (Rubrics)   │    │   (deepeval) │    │  (FAILED)    │      │
│  └──────────────┘    └──────────────┘    └──────┬───────┘      │
│         ▲                                         │              │
│         │                                         ▼              │
│         │            ┌──────────────┐    ┌──────────────┐      │
│         └────────────│ 5. 优化代码  │◀───│ 4. 定位问题  │      │
│                      │   (Fix)      │    │  (Analysis)  │      │
│                      └──────────────┘    └──────────────┘      │
│                                                     │              │
│                                                     ▼              │
│                                            ┌──────────────┐      │
│                                            │  PASSED ✅   │      │
│                                            └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 分阶段实施计划

#### Phase 1: Customer Agent 评测增强（1 周）

**目标**：充分挖掘 DeepEval 能力，建立完整的评测体系

| Week | 任务 | 评测指标 | 场景数 |
|------|------|----------|--------|
| 1.1 | 设计评测 Rubrics | GEval (自定义) | 15 |
| 1.2 | 实现评测用例 | 8 个指标 | 20 |
| 1.3 | 扩充 Golden Dataset | PatternMatch | 30 |
| 1.4 | G-Eval 自动化评估 | GEval | 25 |

**评测场景设计：**
```
场景类别：
├── 订单查询（5 个）
├── 退款申请（5 个）
├── 订单取消（3 个）
├── 投诉处理（4 个）
├── 问候/闲聊（3 个）
└── 复杂多意图（5 个）
```

#### Phase 2: Office Agent 评测体系建立（1 周）

**目标**：为 Office Agent 建立评测框架

| Week | 任务 | 评测指标 | 场景数 |
|------|------|----------|--------|
| 2.1 | 设计 Office Agent Rubrics | GEval (自定义) | 20 |
| 2.2 | 实现 Office Agent 评测用例 | 10 个指标 | 15 |
| 2.3 | 扩充场景覆盖 | TaskCompletion | 20 |
| 2.4 | G-Eval 质量评估 | GEval | 15 |

**评测场景设计：**
```
场景类别：
├── 周报生成（5 个）
├── 客户调研（5 个）
├── 会议准备（4 个）
├── 数据分析（4 个）
├── 文档处理（4 个）
└── 复杂多任务（8 个）
```

#### Phase 3: Office Agent 差距填补（2 周）

**目标**：按优先级填补差距

| 优先级 | 任务 | 验收标准 | 评测指标 |
|--------|------|----------|----------|
| P0 | 实现 Verify Agent 增强 | 多维度验证 | PlanAdherence + TaskCompletion |
| P0 | 实现 Planner 增强 | 动态规划 | PlanQuality + GoalAccuracy |
| P1 | 实现 Sub Agents Mock 增强 | 真实模拟 | ToolUse + TaskCompletion |
| P1 | 实现 Memory System | 上下文传递 | ContextualRelevancy |
| P2 | 实现 Executor 增强 | 失败重试 | TaskCompletion |
| P2 | 实现 Human-in-Loop 增强 | 反馈机制 | GoalAccuracy |

**每个 P0/P1 任务都遵循 EDD 流程：**
1. 先设计评测 Rubrics
2. 运行评测（预期失败）
3. 定位问题
4. 实现优化
5. 再次评测验证

#### Phase 4: 工业级增强（可选，2 周）

| 任务 | 说明 | 评测指标 |
|------|------|----------|
| Observability | 结构化日志 + 追踪 | Custom GEval |
| Multi-turn | 对话历史管理 | ConversationalGEval |
| RAG Integration | 知识库集成 | ContextualPrecision/Recall |
| Real API Integration | 真实 API 调用 | ToolUse + TaskCompletion |

---

## 四、评测指标详细设计

### 4.1 Customer Agent 评测 Rubrics

```python
# 1. ResponseQualityGEval (G-Eval)
rubric = """
评估客服回复质量，评分标准：
1. 准确性（40%）：回复内容是否准确回答用户问题
2. 专业性（30%）：是否使用专业的客服话术
3. 完整性（20%）：是否提供完整的信息和后续步骤
4. 友好性（10%）：语气是否友好、耐心
评分范围：0-100
"""

# 2. IntentAccuracyMetric (自定义)
# 检查 agent 是否正确识别用户意图

# 3. ToolSelectionMetric (自定义)
# 检查 agent 是否选择了正确的工具
```

### 4.2 Office Agent 评测 Rubrics

```python
# 1. TaskPlanQualityGEval (G-Eval)
rubric = """
评估任务计划质量，评分标准：
1. 完整性（30%）：是否覆盖了所有子任务
2. 依赖关系（25%）：任务依赖是否正确
3. 可执行性（25%）：每个任务是否可执行
4. 优先级（20%）：任务优先级是否合理
评分范围：0-100
"""

# 2. TaskCompletionGEval (G-Eval)
rubric = """
评估任务完成度，评分标准：
1. 完成率（40%）：多少任务已完成
2. 质量（30%）：完成的任务质量如何
3. 效率（20%）：是否在合理时间内完成
4. 准确性（10%）：结果是否符合预期
评分范围：0-100
"""

# 3. PlanAdherenceMetric (内置)
# 检查实际执行是否符合计划
```

---

## 五、预期成果

### 5.1 评测覆盖

| Agent | 评测指标 | 场景数 | Golden Dataset |
|-------|----------|--------|----------------|
| Customer Agent | 12 个 | 30 个 | 100 条 |
| Office Agent | 15 个 | 30 个 | 80 条 |

### 5.2 质量目标

| 指标 | 目标 | 当前 |
|------|------|------|
| Customer Agent Pass Rate | 95% | 100% (简化) |
| Office Agent Pass Rate | 85% | N/A |
| G-Eval 平均分 | 80+ | N/A |
| 场景覆盖率 | 90% | 30% |

### 5.3 交付物

1. **完整的评测代码**（DeepEval 原生 API）
2. **Golden Dataset**（180+ 条测试用例）
3. **评测报告**（每次迭代的结果）
4. **优化建议**（基于评测结果）

---

## 六、风险和缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| G-Eval 需要 LLM API | 成本 | 使用本地模型或降低频率 |
| Office Agent 评测复杂 | 时间 | 分阶段实施 |
| Mock LLM 不够真实 | 准确性 | 设计更复杂的 Mock 场景 |
| 评测用例设计困难 | 质量 | 参考 DeepEval 官方示例 |

---

这个方案你觉得如何？如果没问题，我就按这个计划开始实施 Phase 1。
现在我有足够的信息来设计一个完整的 EDD 方案了。让我整理一下：

## DeepEval 支持的评测能力

### 1. 核心评测指标

**LLM Output Quality:**
- `AnswerRelevancyMetric` - 答案相关性
- `FaithfulnessMetric` - 忠诚度（不产生幻觉）
- `SummarizationMetric` - 摘要质量

**Agent Performance:**
- `ToolUseMetric` - 工具使用正确性
- `TaskCompletionMetric` - 任务完成度
- `GoalAccuracyMetric` - 目标达成准确率
- `PlanAdherenceMetric` - 计划执行遵从度
- `PlanQualityMetric` - 计划质量

**Context & RAG:**
- `ContextualPrecisionMetric` - 上下文精确度
- `ContextualRecallMetric` - 上下文召回率
- `ContextualRelevancyMetric` - 上下文相关性

**Safety & Policy:**
- `BiasMetric` - 偏见检测
- `ToxicityMetric` - 有害内容检测
- `PIILeakageMetric` - PII 泄露检测
- `HallucinationMetric` - 幻觉检测

**Code & Data:**
- `JsonCorrectnessMetric` - JSON 格式正确性
- `ExactMatchMetric` - 精确匹配
- `PatternMatchMetric` - 模式匹配

**Role Play:**
- `RoleAdherenceMetric` - 角色扮演一致性


- `RoleViolationMetric` - 角色违规检测

### 2. G-Eval（基于 LLM 的自动化评估）

G-Eval 是 DeepEval 最强大的特性，通过 LLM 自动化地评估 LLM 输出。它支持自定义评估标准（criteria）、明确的评估步骤（evaluation_steps）和自定义评分标准（rubric），使用思维链（CoT）方式生成评估理由，确保评分的一致性。

## Office Agent 当前实现与工业级应用的差距

### 当前实现

基于 LangGraph 的 PEV 架构，采用 Mock LLM 进行离线演示，支持并行任务执行和 Human-in-the-Loop 机制，包含 5 个 Sub Agent（Browser、API、Doc、Data、Visualization）。

### 差距分析

#### 1. **Planner Agent**
- 当前的 MockReasoningModel 过于简单，仅通过关键词匹配
- 缺乏真正的推理能力、上下文理解和动态规划
- 计划质量无法评估

#### 2. **Executor**
- 任务依赖处理采用简单的拓扑排序
- 缺乏失败重试、降级处理和资源管理机制
- 缺少任务超时控制、优先级队列和任务取消功能

#### 3. **Verify Agent**
- 当前仅基于关键词匹配进行验证
- 缺乏深度验证能力和置信度评估
- 验证过程缺乏透明度

#### 4. **Human-in-the-Loop**
- 超时机制表现简陋
- 缺少输入验证、优先级排序和反馈机制

#### 5. **Sub Agents**
- 所有 Agent 都依赖 Mock 实现
- 缺乏真实的 API 调用、错误处理和重试逻辑

#### 6. **Memory & Context**
- 没有会话管理、跨会话上下文和长期记忆存储

#### 7. **Observability**
- 缺少日志记录、追踪和监控能力

#### 8. **Evaluation**
- 还没有建立全面的评估体系

需要设计一个方案来弥补这些差距。
让我快速查看 office_agent.py 的核心逻辑，分析差距：
现在让我查看当前的 Office Agent 实现，分析与工业级应用的差距：
让我查看 G-Eval 的详细信息：
DeepEval 支持很多指标！让我查看更多细节：
让我先研究 DeepEval 支持的所有评测能力，然后设计完整方案。

用户想要一个详细的方案，涉及三个方面：

1. **充分挖掘 DeepEval 的评测能力**：
   - G-Eval（官方提供的自动化评估方法）
   - 更多评测指标（官方支持的）
   - 扩充评测数据和场景
   - 多维度评测

2. **优化 Office Agent**：
   - 理清当前实现和工业级应用的差距
   - 罗列差距
   - 梳理优化计划

3. **EDD（Eval Driven Development）模式**：
   - 先设计评测 rubrics 和 metrics
   - 覆盖更多场景
   - 逐步开发/优化
   - 迭代评测

让我先研究 DeepEval 支持哪些评测能力，然后设计一个完整的方案。
我打算实现：
1. 充分挖掘DeepEval的评测能力（它支持G-Eval），充分用到两个demo agent中，扩充更多的评测数据、场景，实现更多维度的评测（DeepEval官方提供的评测能力，能用的都用上，并告诉我支持哪些，用了哪些）
2. 优化office agent，理清当前实现和实际可落地的工业级应用的差距，罗列这些差距，并梳理优化计划，填补这些差距
3. 「1和2」的方向和规划明确后，采取类似于「TDD（test driven development）」的模式，我们称之为「EDD（eval driven development）」，意思是，先设计评测的rubrics和metrics，覆盖尽可能多的场景，然后逐步开发agent，每次迭代后，运行一次评测任务，发现并定位问题，然后进行优化迭代，直至EDD中的eval完全通过。

你先针对以上的1，2，3，设计一个方案，我过目后觉得没问题，再动手实现。