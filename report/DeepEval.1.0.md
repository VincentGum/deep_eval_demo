# Multi-Agent PEV 系统

> 基于 PEV (Plan-Execute-Verify) 架构的智能 Agent 系统，包含 Customer Agent（智能客服）和 Office Agent（AI 办公助手）

---

## 目录

1. [项目设计](#1-项目设计)
2. [架构设计](#2-架构设计)
3. [Customer Agent 实现](#3-customer-agent-实现)
4. [Office Agent 实现](#4-office-agent-实现)
5. [DeepEval 评测框架](#5-deepeval-评测框架)
6. [运行与验证](#6-运行与验证)
7. [后续可参考的设计模式](#7-后续可参考的设计模式)

---

## 1. 项目设计

### 1.1 项目目标

本项目旨在构建一个**可复用的 Multi-Agent PEV 系统**，包含两个典型场景：

| Agent | 定位 | 特点 |
|-------|------|------|
| **Customer Agent** | 智能客服 | PEV + Human-in-the-Loop，支持敏感操作人工审批 |
| **Office Agent** | AI 办公助手 | PEV + 多 Agent 协作，并行执行复杂办公任务 |

### 1.2 技术选型

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| 语言 | Python 3.10+ | 主力语言 |
| Agent 框架 | LangChain + LangGraph | PEV 状态机实现 |
| 评测框架 | DeepEval 4.0 | 离线模式运行 |
| 测试框架 | Pytest | 单元测试 |
| LLM | Mock | 自定义离线实现（无外部依赖） |

### 1.3 项目结构

```mermaid
mindmap
  root((Multi-Agent PEV))
    Customer Agent
      agent.py - PEV状态图
      mock_llm.py - 离线LLM
      policies.py - 策略验证
      tools.py - 工具定义
    Office Agent
      base.py - 共享基类
      planner.py - 任务规划
      executor.py - 并行执行
      verify.py - 进度验证
      human_loop.py - 人工介入
      sub_agents
        browser_agent.py
        api_agent.py
        doc_agent.py
        data_agent.py
        viz_agent.py
    评测
      tests/evals/
    文档
      report/
```

---

## 2. 架构设计

### 2.1 PEV 核心模式

PEV (Plan-Execute-Verify) 是一种**确定性较强**的 Agent 架构，将任务分解为三个阶段：

```mermaid
flowchart LR
    subgraph PLAN["📋 PLAN"]
        P1[分析意图]
        P2[选择工具]
        P3[生成计划]
    end

    subgraph EXECUTE["⚡ EXECUTE"]
        E1[调用工具]
        E2[收集结果]
    end

    subgraph VERIFY["✅ VERIFY"]
        V1[检查策略]
        V2{是否通过}
    end

    PLAN --> EXECUTE
    EXECUTE --> VERIFY
    V2 -->|通过| END[结束]
    V2 -->|需人工| HITL[Human-in-Loop]
    HITL -->|补充信息| EXECUTE

    style PLAN fill:#e3f2fd
    style EXECUTE fill:#fff3e0
    style VERIFY fill:#e8f5e9
    style HITL fill:#ffebee
```

### 2.2 Human-in-the-Loop 机制

当自动流程无法完成时，引入人工介入：

```mermaid
sequenceDiagram
    participant U as User
    participant V as Verify Agent
    participant H as Human Loop
    participant E as Executor

    U->>V: 任务执行结果
    V->>V: 识别缺失信息

    alt 信息缺失
        V->>H: 请求人工补充
        H-->>U: 提示用户输入
        U->>H: 提供信息
        H->>E: 返回补充信息
        E->>E: 继续执行
    else 信息完整
        V->>V: 继续验证
    end
```

### 2.3 Office Agent 多 Agent 协作

Office Agent 采用 **Planner-Executor-Verify** 架构，支持并行任务执行：

```mermaid
flowchart TB
    START([用户请求]) --> PLANNER

    subgraph PLANNER["🧠 Planner Agent"]
        P1[分析请求]
        P2[生成Task Plan]
        P3[确定依赖关系]
    end

    PLANNER --> EXEC

    subgraph EXEC["⚡ Task Executor (并行)"]
        direction LR
        SUB1[Browser Agent]
        SUB2[API Agent]
        SUB3[Data Agent]
        SUB4[Doc Agent]
        SUB5[Viz Agent]
    end

    EXEC --> VERIFY

    subgraph VERIFY["🔍 Verify Agent"]
        V1[收集结果]
        V2[对比预期]
        V3{是否完成}
    end

    VERIFY -->|全部完成| END([结束])
    VERIFY -->|缺失信息| HITL
    HITL -->|补充后| EXEC

    HITL([⏸ Human-in-Loop])
    HITL -.->|超时跳过| END

    style PLANNER fill:#e3f2fd
    style EXEC fill:#fff3e0
    style VERIFY fill:#e8f5e9
    style HITL fill:#ffebee
```

### 2.4 数据流设计

```mermaid
flowchart TB
    subgraph State["Agent State"]
        direction TB
        S1[user_message]
        S2[intent]
        S3[action_plan]
        S4[tool_results]
        S5[response]
        S6[requires_human_review]
    end

    S1 -->|输入| S2
    S2 -->|意图| S3
    S3 -->|计划| S4
    S4 -->|结果| S5
    S5 -->|检查| S6

    style State fill:#fafafa,stroke:#ddd
```

---

## 3. Customer Agent 实现

### 3.1 架构概述

Customer Agent 是**单 Agent PEV 模式**的典型实现，专注于客服场景：

```mermaid
flowchart TD
    A[用户消息] --> B[PLAN]
    B --> C{意图分析}
    C -->|订单查询| D[lookup_order]
    C -->|退款| E[create_refund_case]
    C -->|问候| F[直接回复]

    D --> G[EXECUTE]
    E --> G
    F --> J

    G --> H[VERIFY]
    H -->|安全| J[回复用户]
    H -->|需审批| I[HUMAN_REVIEW]
    I -->|批准| J
    I -->|拒绝| K[错误回复]

    style A fill:#e1f5fe
    style J fill:#c8e6c9
    style I fill:#fff3e0
```

### 3.2 状态定义

```mermaid
classDiagram
    class AgentState {
        +str user_message
        +Optional~str~ intent
        +Optional~str~ draft_response
        +List~str~ tools_to_use
        +List~ToolResult~ tool_results
        +Optional~bool~ requires_human_review
        +Optional~str~ policy_reason
        +str current_node
        +str final_response
    }

    class ToolResult {
        +str tool_name
        +Any result
        +bool success
    }

    class PolicyDecision {
        +bool requires_human_review
        +str reason
    }

    AgentState "1" o-- "0..*" ToolResult
    AgentState "1" -- "1" PolicyDecision
```

### 3.3 核心流程伪代码

```python
# 伪代码展示 PEV 流程

def customer_agent_pipeline(user_message):
    # PLAN: 分析意图，选择工具
    plan = plan_node(user_message)
    if plan.confidence < 0.5:
        return "抱歉，我没有理解您的问题"

    # EXECUTE: 调用工具
    results = execute_tools(plan.tools_to_use)

    # VERIFY: 检查策略
    policy = verify_policy(results)

    if policy.requires_human_review:
        # HUMAN_REVIEW: 等待人工审批
        approved = request_human_approval(results)
        if not approved:
            return "操作已被拒绝"
        return format_response(results)

    return format_response(results)
```

### 3.4 意图识别模式

```mermaid
flowchart LR
    subgraph Input["用户输入"]
        I1["我想查一下订单 #A100"]
        I2["我要申请退款"]
        I3["你好"]
    end

    subgraph Patterns["意图模式匹配"]
        P1{包含订单号?}
        P2{退款/取消/投诉关键词?}
        P3{hello/hi/你好?}
    end

    subgraph Output["意图输出"]
        O1[order_status]
        O2[refund/cancel/complaint]
        O3[greeting]
    end

    I1 --> P1 -->|是| O1
    I2 --> P2 -->|是| O2
    I3 --> P3 -->|是| O3
```

### 3.5 策略验证

```python
# 伪代码展示策略验证

class PolicyChecker:
    SENSITIVE_KEYWORDS = ["退款", "取消", "投诉"]

    def verify(self, intent, confidence, tool_results):
        # 1. 检查敏感词
        if contains_sensitive(intent):
            return PolicyDecision(requires_human=True, reason="涉及敏感操作")

        # 2. 检查置信度
        if confidence < 0.7:
            return PolicyDecision(requires_human=True, reason="置信度过低")

        # 3. 检查工具结果
        if tool_failed(tool_results):
            return PolicyDecision(requires_human=True, reason="工具执行失败")

        return PolicyDecision(requires_human=False, reason="通过验证")
```

---

## 4. Office Agent 实现

### 4.1 架构概述

Office Agent 是**多 Agent 协作模式**的实现，专注于复杂办公任务：

```mermaid
flowchart TB
    subgraph Input["用户复杂请求"]
        R1["帮我生成上周的销售周报"]
        R2["调研一下 XX 客户的情况"]
        R3["准备明天客户会议的资料"]
    end

    subgraph PLANNER["🧠 Planner Agent"]
        P1[解析请求]
        P2[分解任务]
        P3[生成 Task Plan]
    end

    Input --> PLANNER

    subgraph TASKS["并行任务执行"]
        T1["Browser: 抓取数据"]
        T2["API: 调用接口"]
        T3["Data: 数据统计"]
        T4["Doc: 生成文档"]
        T5["Viz: 绘制图表"]
    end

    PLANNER --> TASKS

    subgraph VERIFY["🔍 Verify Agent"]
        V1[收集结果]
        V2[对比预期]
        V3{完成?}
    end

    TASKS --> VERIFY

    VERIFY -->|缺失| HITL[⏸ 人工介入]
    HITL -->|补充| TASKS
    VERIFY -->|完成| OUTPUT[最终输出]

    style PLANNER fill:#e3f2fd
    style TASKS fill:#fff3e0
    style VERIFY fill:#e8f5e9
    style HITL fill:#ffebee
```

### 4.2 Task Plan 结构

```mermaid
classDiagram
    class TaskPlan {
        +str id
        +str original_request
        +List~Task~ tasks
        +str expected_outcome
    }

    class Task {
        +str id
        +str description
        +AgentCapability required_capability
        +Optional~List~str~ depends_on
        +Optional~Dict~ params
        +TaskStatus status
        +Optional~Any~ result
    }

    class TaskStatus {
        <<enumeration>>
        PENDING
        RUNNING
        COMPLETED
        FAILED
        BLOCKED
        SKIPPED
    }

    class AgentCapability {
        <<enumeration>>
        BROWSER
        API
        DATA
        VISUALIZATION
        DOC
    }

    TaskPlan "1" --> "1..*" Task
    Task --> TaskStatus
    Task --> AgentCapability
```

### 4.3 Planner Agent 工作流程

```python
# 伪代码展示 Planner 流程

class PlannerAgent:
    def plan(self, user_request):
        # 1. 高阶推理分析
        analysis = self.reasoning_model.analyze(user_request)

        # 2. 任务分解
        tasks = []
        for subtask in analysis.subtasks:
            task = Task(
                description=subtask.description,
                capability=subtask.required_capability,
                params=subtask.params,
                depends_on=subtask.dependencies
            )
            tasks.append(task)

        # 3. 生成预期结果
        expected = self.reasoning_model.predict_outcome(tasks)

        return TaskPlan(
            tasks=tasks,
            expected_outcome=expected
        )
```

### 4.4 Task Executor 并行执行

```mermaid
flowchart LR
    subgraph Ready["就绪任务"]
        R1[T1]
        R2[T2]
        R3[T3]
    end

    subgraph Pool["执行池"]
        W1[Worker 1]
        W2[Worker 2]
        W3[Worker 3]
    end

    subgraph Complete["已完成"]
        C1[T1 ✓]
        C2[T2 ✓]
    end

    R1 -->|依赖满足| W1
    R2 -->|依赖满足| W2
    R3 -->|等待 T1| W3

    W1 --> C1
    W2 --> C2
    W3 -.->|T1完成| R3

    style W1 fill:#c8e6c9
    style W2 fill:#c8e6c9
    style W3 fill:#fff3e0
```

### 4.5 Verify Agent 验证逻辑

```python
# 伪代码展示 Verify 流程

class VerifyAgent:
    def verify(self, plan, results):
        # 1. 收集任务结果
        completed = {t.id: t for t in results if t.status == COMPLETED}

        # 2. 检查缺失任务
        missing = [t for t in plan.tasks if t.id not in completed]

        if not missing:
            # 3. 对比预期结果
            outcome_check = self.check_outcome(plan.expected_outcome, results)
            if outcome_check.passed:
                return VerificationResult(status=COMPLETED, missing_info=None)
            else:
                return VerificationResult(status=NEEDS_HUMAN, missing_info=outcome_check.gaps)

        # 4. 识别需要人工补充的信息
        gaps = self.identify_gaps(missing, results)
        return VerificationResult(status=NEEDS_HUMAN, missing_info=gaps)
```

### 4.6 办公场景示例

#### 场景 1: 周报生成

```mermaid
flowchart LR
    A[用户请求] --> P[Planner]

    P -->|Task 1| B["API: 获取销售数据"]
    P -->|Task 2| C["Data: 统计分析"]
    P -->|Task 3| D["Viz: 生成图表"]
    P -->|Task 4| E["Doc: 撰写周报"]

    B -->|数据| C
    C -->|分析结果| D
    D -->|图表| E

    E --> V[Verify]
    V -->|完成| F[最终报告]

    style B fill:#e3f2fd
    style C fill:#fff3e0
    style D fill:#e8f5e9
    style E fill:#f3e5f5
```

---

## 5. DeepEval 评测框架

### 5.1 评测指标设计

```mermaid
flowchart TB
    subgraph Metrics["自定义评测指标"]
        M1[IntentAccuracyMetric]
        M2[ToolSelectionMetric]
        M3[HumanReviewDecisionMetric]
    end

    subgraph TestCase["测试用例"]
        T1[输入: 用户消息]
        T2[输出: Agent回复]
        T3[上下文: 预期意图/工具/审批]
    end

    subgraph Evaluation["评估流程"]
        E1[加载 Golden Data]
        E2[执行 Agent]
        E3[计算指标得分]
        E4[生成报告]
    end

    Metrics --> Evaluation
    TestCase --> Evaluation
```

### 5.2 离线评测实现

```python
# 使用 DeepEval 自定义指标

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric

class IntentAccuracyMetric(BaseMetric):
    """评估意图识别的准确性"""

    def _evaluate(self, test_case):
        expected_intent = test_case.context[0]
        actual_output = test_case.actual_output

        # 离线模式：基于关键词匹配
        if expected_intent in ["greeting", "thanks"]:
            score = 1.0 if any(kw in actual_output for kw in ["hello", "help", "thank"]) else 0.0
        else:
            score = 1.0 if "order" in actual_output.lower() else 0.0

        return score
```

### 5.3 评测命令

```bash
# 使用 deepeval 命令运行测试
deepeval test run tests/evals/test_customer_agent.py

# 或使用 pytest
pytest tests/evals/test_customer_agent.py -v
```

---

## 6. 运行与验证

### 6.1 Customer Agent 测试

```bash
# 运行单个场景
PYTHONPATH=src python3 examples/run_demo.py

# 运行 DeepEval 评测
deepeval test run tests/evals/test_customer_agent.py
```

### 6.2 Office Agent 测试

```bash
# 列出可用场景
PYTHONPATH=src python3 examples/run_office_agent.py --list

# 运行指定场景
PYTHONPATH=src python3 examples/run_office_agent.py --scenario=weekly_sales_report

# 运行所有场景
PYTHONPATH=src python3 examples/run_office_agent.py --all
```

### 6.3 测试结果

| Agent | 测试项 | 结果 |
|-------|--------|------|
| Customer Agent | 单元测试 | 7/7 通过 |
| Customer Agent | Golden Dataset | 10/10 通过 |
| Office Agent | Weekly Sales Report | 7 tasks ✓ |
| Office Agent | Customer Research | 6 tasks ✓ |
| Office Agent | Meeting Preparation | 3 tasks ✓ |

---

## 7. 后续可参考的设计模式

### 7.1 多 Agent 协作模式

```mermaid
flowchart TB
    subgraph Supervisor["Supervisor Agent"]
        S1[分析任务]
        S2[分配子任务]
        S3[汇总结果]
    end

    subgraph Workers["Worker Agents"]
        W1[Agent 1]
        W2[Agent 2]
        W3[Agent N]
    end

    S1 --> S2
    S2 --> W1
    S2 --> W2
    S2 --> W3
    W1 --> S3
    W2 --> S3
    W3 --> S3
```

### 7.2 记忆增强模式

```mermaid
flowchart LR
    subgraph Memory["记忆系统"]
        M1[短期记忆<br/>当前会话]
        M2[长期记忆<br/>持久化]
        M3[向量存储<br/>语义检索]
    end

    Agent -->|写入| M1
    M1 -->|定期同步| M2
    M2 -->|检索| M3
    M3 -->|召回| Agent

    style M1 fill:#e3f2fd
    style M2 fill:#fff3e0
    style M3 fill:#e8f5e9
```

### 7.3 工具调用增强

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| Tool Fusion | 组合多个工具调用 | 复杂查询 |
| Tool Retry | 失败自动重试 | 网络不稳定 |
| Tool Fallback | 主工具失败切换备选 | 降级处理 |
| Tool Planning | 工具调用序列规划 | 多步骤任务 |

### 7.4 扩展方向

| 方向 | 说明 |
|------|------|
| **RAG 集成** | 引入知识库检索增强回复 |
| **多模态** | 支持图像、文档等非文本输入 |
| **实时 LLM** | 替换 Mock 为真实模型（GPT-4、Claude 等） |
| **分布式执行** | 支持多 Agent 跨进程/跨机器协作 |
| **评测增强** | 引入 G-Eval 等基于模型的评测指标 |

---

## 附录

### A. 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PYTHONPATH` | Python 模块搜索路径 | `src` |
| `DEEPEVAL_TELEMETRY_OPT_OUT` | 关闭 DeepEval 遥测 | `YES` |

### B. 依赖安装

```bash
pip install langchain langgraph deepeval pytest
```

### C. 版本信息

- DeepEval: 4.0.0
- LangChain: 最新兼容版本
- LangGraph: 最新兼容版本
