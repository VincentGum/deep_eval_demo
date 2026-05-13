## 项目概述

**Multi-Agent PEV 系统** - 包含两个基于 LangChain/LangGraph 的 PEV（Plan-Execute-Verify）架构的 Agent Demo：

1. **Customer Agent** - 智能客服系统（已完成 DeepEval 评测）
2. **Office Agent** - AI 办公助手（支持 Human-in-the-Loop）

## 技术栈

- **语言**: Python 3.10+
- **框架**: LangChain, LangGraph
- **测试**: DeepEval (离线模式，仅 Customer Agent)
- **包管理**: uv + 项目级虚拟环境（系统环境用 pip）

## 目录结构

```
/workspace/projects/
├── src/
│   ├── customer_agent/           # 客服 Agent
│   │   ├── __init__.py
│   │   ├── agent.py             # LangGraph PEV 状态图
│   │   ├── mock_llm.py         # Mock LLM 实现
│   │   ├── policies.py         # 策略验证
│   │   └── tools.py            # 工具定义
│   │
│   ├── office_agent/            # 办公助手 Agent
│   │   ├── __init__.py
│   │   ├── base.py             # 共享基类和接口
│   │   ├── planner.py         # Planner Agent (任务规划)
│   │   ├── executor.py         # Task Executor (并行执行)
│   │   ├── verify.py           # Verify Agent (进度验证)
│   │   ├── human_loop.py       # Human-in-the-Loop
│   │   ├── office_agent.py     # 主入口和流程编排
│   │   ├── scenarios.py        # 办公场景定义
│   │   └── sub_agents/         # 子 Agent 集合
│   │       ├── __init__.py
│   │       ├── browser_agent.py
│   │       ├── api_agent.py
│   │       ├── doc_agent.py
│   │       ├── data_agent.py
│   │       ├── visualization_agent.py
│   │       └── registry.py
│   │
│   └── shared/                  # 共享模块（未来扩展）
│       └── __init__.py
│
├── examples/
│   ├── run_demo.py             # Customer Agent Demo
│   └── run_office_agent.py      # Office Agent Demo
│
├── tests/
│   ├── conftest.py             # Pytest 配置
│   └── evals/                   # Customer Agent 评测
│
├── report/                      # 文档
│   └── DeepEval.1.0.md
│
├── pyproject.toml
├── requirements.txt
├── README.md
└── AGENTS.md
```

## 核心模块

### Customer Agent (客服 Agent)

**PEV 架构节点:**
1. **PLAN**: 分析用户意图，决定调用哪些工具
2. **EXECUTE**: 调用 tools.py 中的工具
3. **VERIFY**: 调用 policies.verify_policy 检查敏感词和置信度
4. **HUMAN_REVIEW**: 敏感操作需人工审批

**核心函数:**
- `invoke_customer_agent(user_message, human_approval_func)` - 入口函数
- `build_customer_support_graph()` - 构建 LangGraph 状态图

### Office Agent (办公助手 Agent)

**PEV + 子 Agent 架构:**

```
User Request
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  PLANNER AGENT (High-level Reasoning Model)                │
│  - 分析用户请求                                             │
│  - 生成 Task Plan（一组 Tasks）                             │
│  - 确定任务依赖关系                                         │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  TASK EXECUTOR (Parallel Execution)                        │
│  - 并行调度多个 Sub Agents                                  │
│  - 管理任务依赖                                             │
│  - 共享数据传递                                             │
└─────────────────────────────────────────────────────────────┘
     │
     ├───► Browser Agent (浏览网页、爬取数据)
     ├───► API Agent (调用 OpenAPI)
     ├───► Data Agent (数据处理、统计分析)
     ├───► Visualization Agent (图表生成)
     ├───► Doc Agent (文档读写)
     └───► ... (可扩展更多 Agent)
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  VERIFY AGENT (Progress Monitoring)                         │
│  - 对比实际结果 vs Planner 预期                              │
│  - 判断任务是否完成                                         │
│  - 识别缺失信息                                             │
└─────────────────────────────────────────────────────────────┘
     │
     ├───► 全部完成 ──► 流程结束
     │
     └───► 缺少信息 ──► HUMAN-IN-THE-LOOP ──► 补充信息 ──► 继续执行
```

**核心特性:**
- **Planner Agent**: 使用 Mock 高级推理模型（模拟 o1）
- **并行执行**: ThreadPoolExecutor 支持多任务并发
- **Verify Agent**: 独立验证进度，对比 Planner 预期
- **Human-in-the-Loop**: 任务等待人工输入（有超时限制）
- **可扩展 Sub Agents**: 通过 Registry 动态注册

## 运行与预览

### Customer Agent

```bash
# 运行 Demo
PYTHONPATH=src python3 examples/run_demo.py

# 运行测试
PYTHONPATH=src DEEPEVAL_TELEMETRY_OPT_OUT=YES python3 -m pytest tests/evals/test_customer_agent.py -v
```

### Office Agent

```bash
# 列出可用场景
PYTHONPATH=src python3 examples/run_office_agent.py --list

# 运行特定场景
PYTHONPATH=src python3 examples/run_office_agent.py --scenario=weekly_sales_report

# 运行所有场景
PYTHONPATH=src python3 examples/run_office_agent.py --all

# 交互模式
PYTHONPATH=src python3 examples/run_office_agent.py --interactive
```

## 办公场景

| 场景 | 描述 | 涉及 Agent |
|------|------|-----------|
| Weekly Sales Report | 生成周报（API + 统计 + 图表 + 文档） | API, Data, Visualization, Doc |
| Customer Research | 客户调研（爬虫 + 数据库 + 报告） | Browser, API, Data, Doc |
| Meeting Preparation | 会议准备（查询 + 格式化 + 文档） | Data, Doc |

## 项目状态

### Customer Agent
- **完成度**: 100%
- **测试通过**: 7/7
- **Golden Dataset**: 10/10 通过，平均得分 99.17%

### Office Agent
- **完成度**: 100%
- **场景测试**: 3/3 通过
- **DeepEval**: 暂未实现（按需求）

## 用户偏好与长期约束

- 使用 `uv` 管理 Python 依赖（系统环境用 pip）
- `PYTHONPATH` 需设为 `src` 才能正确导入模块
- DeepEval 需设置 `DEEPEVAL_TELEMETRY_OPT_OUT=YES` 离线运行
- Office Agent 默认 Human-in-the-Loop 超时 60 秒

## 共享模块设计

为了支持两个 Agent 之间的能力复用，Office Agent 的 `base.py` 提供了可复用的抽象：

- `Task`, `TaskPlan`, `TaskStatus` - 任务定义
- `AgentCapability` - 能力枚举
- `BaseSubAgent` - 子 Agent 基类

未来可以将共享模块提取到 `src/shared/` 目录，供两个 Agent 共用。

## 常见问题和预防

- 缺少入口文件时项目无法运行
- 运行时必须设置 `PYTHONPATH=src`
- Office Agent 的 Human-in-the-Loop 需要用户提供输入才能继续
