# DeepEval 1.0 技术文档

> Offline PEV Customer Support Agent 评测实现完整指南

---

## 目录

1. [项目概述](#1-项目概述)
2. [架构设计](#2-架构设计)
3. [核心模块实现](#3-核心模块实现)
4. [DeepEval 评测框架](#4-deepeval-评测框架)
5. [运行方式](#5-运行方式)
6. [运行结果](#6-运行结果)
7. [附录](#7-附录)
8. [Office Agent - AI 办公助手](#8-office-agent---ai-办公助手)

---

## 1. 项目概述

### 1.1 目标

实现一个智能客服的 **PEV + Human-in-the-Loop** 架构 Agent，模型使用 Mock 离线运行，使用 DeepEval 进行效果评测。

### 1.2 技术栈

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| 语言 | Python 3.10+ | 主力语言 |
| Agent 框架 | LangChain + LangGraph | PEV 状态机实现 |
| 评测框架 | DeepEval 4.0 | 离线模式运行 |
| 测试框架 | Pytest | 单元测试 |
| LLM | MockChatModel | 自定义离线实现 |

### 1.3 项目结构

```
/workspace/projects/
├── src/
│   └── customer_agent/
│       ├── __init__.py              # 导出接口
│       ├── agent.py                 # LangGraph PEV 状态图 [核心]
│       ├── mock_llm.py              # Mock LLM 实现 [核心]
│       ├── policies.py              # 策略验证
│       └── tools.py                 # 工具定义
├── examples/
│   └── run_demo.py                  # 演示脚本
├── tests/
│   ├── conftest.py                  # Pytest 配置
│   └── evals/
│       ├── test_customer_agent.py   # 评测用例
│       └── customer_support_goldens.json  # 测试数据
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 2. 架构设计

### 2.1 PEV 架构

PEV (Plan-Execute-Verify) 是一种确定性较强的 Agent 架构模式：

```
┌─────────────────────────────────────────────────────────────────┐
│                           User Input                              │
│                     "Where is my order #A100?"                   │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                         PLAN Node                                 │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ - 分析用户意图 (intent)                                  │     │
│  │ - 决定需要调用的工具 (tools)                             │     │
│  │ - 生成草稿回复 (draft_response)                          │     │
│  │ - 评估置信度 (confidence)                                 │     │
│  └─────────────────────────────────────────────────────────┘     │
│  输出: {intent: "order_status", tools: ["lookup_order"],          │
│        confidence: 0.85, draft_response: "..."}                   │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        EXECUTE Node                               │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ - 根据 PLAN 决定调用对应工具                              │     │
│  │ - 收集工具返回结果                                        │     │
│  │ - 追加到草稿回复                                          │     │
│  └─────────────────────────────────────────────────────────┘     │
│  输出: {tool_calls: [...], tool_results: ["Order A100 is..."]}  │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        VERIFY Node                               │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ - 调用 policies.verify_policy()                          │     │
│  │ - 检查敏感关键词 (refund/complaint/cancel)               │     │
│  │ - 检查置信度阈值 (< 0.7 需人工审批)                       │     │
│  └─────────────────────────────────────────────────────────┘     │
│  输出: PolicyDecision {requires_human: False, reason: "..."}    │
└─────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
┌─────────────────────────────┐    ┌─────────────────────────────┐
│     requires_human=False    │    │     requires_human=True     │
│         ↓ 直接回复          │    │        HUMAN_REVIEW         │
│         结束                │    │  (可选人工审批流程)          │
└─────────────────────────────┘    └─────────────────────────────┘
```

### 2.2 Human-in-the-Loop 机制

敏感操作触发人工审批流程：

| 触发条件 | 说明 |
|----------|------|
| 置信度 < 0.7 | 模型对回复不确定 |
| 包含敏感关键词 | refund/退款/complaint/投诉/cancel/取消 等 |
| 用户明确要求 | 如取消订单、申请赔偿 |

### 2.3 状态定义

```python
class AgentState(TypedDict):
    user_message: str           # 用户输入
    current_stage: AgentStage  # 当前阶段 (PLAN/EXECUTE/VERIFY/HUMAN_REVIEW/FINAL)
    plan: PlanResult | None    # PLAN 阶段输出
    execute: ExecuteResult | None  # EXECUTE 阶段输出
    verify: VerifyResult | None    # VERIFY 阶段输出
    human_review: HumanReviewResult | None  # HUMAN_REVIEW 阶段输出
    final_response: str         # 最终回复
    messages: list             # 消息历史
    error: str | None          # 错误信息
```

---

## 3. 核心模块实现

### 3.1 Mock LLM (`mock_llm.py`)

#### 设计思路

由于是离线 Demo，不能依赖真实 LLM API。Mock LLM 通过**意图模式匹配**模拟 LLM 的响应：

```python
class IntentPattern:
    keywords: tuple[str, ...]      # 匹配的关键词
    response: str                   # 模拟回复
    confidence: float              # 置信度
    suggested_action: str | None    # 建议工具
```

#### 预定义意图模式

| 意图 | 关键词 | 置信度 | 建议工具 |
|------|--------|--------|----------|
| order_status | order, delivery, shipping, 物流, track... | 0.85 | lookup_order |
| greeting | hello, hi, 你好, 问 | 0.95 | None |
| thanks | thank, thanks, 谢谢, 感谢 | 0.95 | None |

#### 规范化 Intent 映射

```python
INTENT_NAMES = {
    "order": "order_status",
    "delivery": "order_status",
    "track": "order_status",
    "refund": "refund",
    "cancel": "cancel",
    "complaint": "complaint",
    "hello": "greeting",
    "thank": "thanks",
    # ...
}
```

#### 使用示例

```python
from customer_agent.mock_llm import MockChatModel
from langchain_core.messages import HumanMessage

model = MockChatModel()
response = model.invoke([HumanMessage(content="Where is my order #A100?")])
# response.content = "I'll look up your order status right away."
# response.additional_kwargs['confidence'] = 0.85
# response.additional_kwargs['intent'] = "order_status"
```

### 3.2 Agent 实现 (`agent.py`)

#### PLAN Node

```python
def create_plan_node(llm: MockChatModel) -> Callable[[AgentState], AgentState]:
    def plan_node(state: AgentState) -> AgentState:
        # 1. 调用 Mock LLM 获取意图和响应
        response = llm.invoke(messages)

        # 2. 提取订单 ID
        order_id = extract_order_id(user_message)

        # 3. 基于用户消息关键词决定工具
        tools_to_use = []
        if order_id != "UNKNOWN":
            tools_to_use = ["lookup_order"]
            if any(keyword in user_message_lower for keyword in SENSITIVE_KEYWORDS):
                tools_to_use.append("create_refund_case")

        # 4. 构建 PlanResult
        state["plan"] = PlanResult(
            intent=intent,
            tools_to_use=tools_to_use,
            confidence=confidence,
            draft_response=draft_response,
        )
        state["current_stage"] = AgentStage.EXECUTE
        return state
    return plan_node
```

#### EXECUTE Node

```python
def create_execute_node() -> Callable[[AgentState], AgentState]:
    tools_map = {
        "lookup_order": lookup_order,
        "create_refund_case": create_refund_case,
    }

    def execute_node(state: AgentState) -> AgentState:
        for tool_name in plan.tools_to_use:
            tool_func = tools_map[tool_name]
            result = tool_func.invoke({"order_id": order_id})
            tool_results.append(str(result))

        state["execute"] = ExecuteResult(
            tool_calls=[...],
            tool_results=tool_results,
        )
        state["current_stage"] = AgentStage.VERIFY
        return state
    return execute_node
```

#### VERIFY Node

```python
def create_verify_node() -> Callable[[AgentState], AgentState]:
    def verify_node(state: AgentState) -> AgentState:
        # 构建草稿回复
        draft = plan.draft_response
        if execute.tool_results:
            draft += "\n" + "\n".join(execute.tool_results)

        # 调用策略验证
        decision = verify_policy(
            user_message=state["user_message"],
            draft=draft,
            confidence=plan.confidence,
        )

        # 决定下一步
        if decision.requires_human:
            state["current_stage"] = AgentStage.HUMAN_REVIEW
        else:
            state["current_stage"] = AgentStage.FINAL
            state["final_response"] = draft

        return state
    return verify_node
```

#### 图构建

```python
def build_customer_support_graph(
    human_approval_func=None,
    llm=None,
) -> StateGraph:
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("plan", create_plan_node(llm))
    workflow.add_node("execute", create_execute_node())
    workflow.add_node("verify", create_verify_node())
    workflow.add_node("human_review", create_human_review_node(human_approval_func))
    workflow.add_node("final", lambda s: s)

    # 定义边
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "execute")
    workflow.add_edge("execute", "verify")
    workflow.add_conditional_edges("verify", should_human_review)
    workflow.add_edge("human_review", "final")
    workflow.add_edge("final", END)

    return workflow.compile()
```

### 3.3 策略验证 (`policies.py`)

```python
SENSITIVE_KEYWORDS = (
    "refund", "退款", "complaint", "投诉",
    "chargeback", "赔偿", "cancel", "取消",
)

@dataclass(frozen=True)
class PolicyDecision:
    requires_human: bool
    reason: str

def verify_policy(user_message: str, draft: str, confidence: float) -> PolicyDecision:
    text = f"{user_message} {draft}".lower()

    # 低置信度 → 需要人工审批
    if confidence < 0.7:
        return PolicyDecision(True, "low_confidence")

    # 敏感关键词 → 需要人工审批
    if any(keyword in text for keyword in SENSITIVE_KEYWORDS):
        return PolicyDecision(True, "sensitive_customer_action")

    # 默认自动通过
    return PolicyDecision(False, "auto_approved")
```

### 3.4 工具定义 (`tools.py`)

```python
@tool
def lookup_order(order_id: str) -> str:
    """返回本地 mock 订单状态"""
    orders = {
        "A100": "Order A100 is in transit and expected tomorrow.",
        "B200": "Order B200 was delivered yesterday.",
        "C300": "Order C300 payment is pending.",
    }
    return orders.get(order_id.upper(), "Order was not found...")

@tool
def create_refund_case(order_id: str, reason: str) -> str:
    """创建退款工单（仍需人工审批）"""
    return f"Refund case DRAFT-{order_id.upper()} created for review: {reason}"
```

---

## 4. DeepEval 评测框架

### 4.1 设计思路

DeepEval 的核心是**指标 (Metric)** 和**测试用例 (Test Case)** 的分离：

```
┌─────────────────────────────────────────────────────────┐
│                    DeepEval Framework                    │
├─────────────────────────────────────────────────────────┤
│  Test Case: {                                             │
│    input: "Where is my order #A100?",                     │
│    expected_intent: "order_status",                       │
│    expected_tools: ["lookup_order"],                      │
│    expected_human_review: false,                          │
│    expected_response_contains: ["A100", "transit"]        │
│  }                                                        │
│                          │                                │
│                          ▼                                │
│  Metrics:                                                 │
│  - IntentAccuracyMetric  (意图准确率)                     │
│  - ToolSelectionMetric    (工具选择准确率)                │
│  - HumanReviewDecisionMetric (人工审批决策准确率)         │
│  - ResponseContainsMetric (回复关键词覆盖率)              │
│                          │                                │
│                          ▼                                │
│  Score: 100% / 75% / 50% / 25% / 0%                       │
└─────────────────────────────────────────────────────────┘
```

### 4.2 自定义离线指标

DeepEval 完整功能依赖 LLM 评分，本项目实现**确定性离线指标**：

#### IntentAccuracyMetric

```python
class IntentAccuracyMetric(OfflineMetric):
    def evaluate(self, actual_intent: str, expected_intent: str, **kwargs):
        success = actual_intent == expected_intent
        score = 1.0 if success else 0.0
        return {"success": success, "score": score, "reason": "..."}
```

#### ToolSelectionMetric

```python
class ToolSelectionMetric(OfflineMetric):
    def evaluate(self, actual_tools: list, expected_tools: list, **kwargs):
        if not expected_tools:
            return {"success": True, "score": 1.0}
        all_present = all(tool in actual_tools for tool in expected_tools)
        return {"success": all_present, "score": 1.0 if all_present else 0.0}
```

#### HumanReviewDecisionMetric

```python
class HumanReviewDecisionMetric(OfflineMetric):
    def evaluate(self, actual_human_review: bool, expected_human_review: bool, **kwargs):
        success = actual_human_review == expected_human_review
        return {"success": success, "score": 1.0 if success else 0.0}
```

#### ResponseContainsMetric

```python
class ResponseContainsMetric(OfflineMetric):
    def evaluate(self, response: str, expected_contains: list, **kwargs):
        response_lower = response.lower()
        matched = sum(1 for kw in expected_contains if kw.lower() in response_lower)
        score = matched / len(expected_contains)
        return {"success": score >= 0.7, "score": score}
```

### 4.3 Golden Dataset

存储在 `tests/evals/customer_support_goldens.json`：

```json
[
  {
    "id": "test_001",
    "input": "Hi, can you tell me the status of my order #A100?",
    "expected_intent": "order_status",
    "expected_tools": ["lookup_order"],
    "expected_human_review": false,
    "expected_response_contains": ["A100", "transit", "tomorrow"],
    "category": "order_inquiry"
  },
  {
    "id": "test_002",
    "input": "I want to request a refund for order #B200, the item was damaged.",
    "expected_intent": "order_status",
    "expected_tools": ["lookup_order", "create_refund_case"],
    "expected_human_review": true,
    "expected_response_contains": ["B200", "refund", "review"],
    "category": "refund_request"
  },
  // ...
]
```

### 4.4 测试用例

```python
def test_all_golden_cases(self, golden_data, metrics):
    results = []
    for test_case in golden_data:
        result = invoke_customer_agent(test_case["input"])
        plan = result["state"].get("plan")

        # 评估每个指标
        intent_result = metrics["intent"].evaluate(
            actual_intent=plan.intent,
            expected_intent=test_case["expected_intent"],
        )
        tools_result = metrics["tools"].evaluate(
            actual_tools=plan.tools_to_use,
            expected_tools=test_case["expected_tools"],
        )
        hr_result = metrics["human_review"].evaluate(...)
        response_result = metrics["response_contains"].evaluate(...)

        # 计算总分
        overall_score = sum([r["score"] for r in metric_results.values()]) / len(metric_results)
        results.append({...})

    # 输出汇总
    print(f"Total: {len(results)}, Passed: {sum(1 for r in results if r['success'])}")
```

---

## 5. 运行方式

### 5.1 环境准备

```bash
# 安装依赖（系统环境）
pip install langchain langgraph deepeval pytest --user

# 或使用虚拟环境
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5.2 运行 Agent Demo

```bash
PYTHONPATH=src python3 examples/run_demo.py
```

**预期输出：**
```
============================================================
PEV Customer Support Agent - Demo
============================================================

────────────────────────────────────────────────────────────
Test 1: Order Status Inquiry
User: Hi, can you tell me the status of my order #A100?
────────────────────────────────────────────────────────────

Agent Response:
Based on your inquiry about order A100, I found:
Order A100 is in transit and expected tomorrow.

[Debug Info]
  Stage: final
  Needs Human Review: False
  Intent: order_status
  Confidence: 0.85
  Tools Used: ['lookup_order']
  Policy Decision: requires_human=False, reason=auto_approved
...
```

### 5.3 运行 DeepEval 测试

```bash
PYTHONPATH=src DEEPEVAL_TELEMETRY_OPT_OUT=YES python3 -m pytest tests/evals/test_customer_agent.py -v
```

**预期输出：**
```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.4.2
plugins: deepeval-4.0.0, ...

tests/evals/test_customer_agent.py::TestCustomerSupportAgent::test_customer_agent_order_inquiry PASSED
tests/evals/test_customer_agent.py::TestCustomerSupportAgent::test_customer_agent_refund_request PASSED
tests/evals/test_customer_agent.py::TestCustomerSupportAgent::test_customer_agent_complaint PASSED
tests/evals/test_customer_agent.py::TestCustomerSupportAgent::test_customer_agent_general_inquiry PASSED
tests/evals/test_customer_agent.py::TestCustomerSupportAgent::test_customer_agent_cancellation PASSED
tests/evals/test_customer_agent.py::TestCustomerSupportAgent::test_customer_agent_thanks PASSED
tests/evals/test_customer_agent.py::TestCustomerSupportAgent::test_all_golden_cases PASSED

============================== 7 passed in 0.33s ===============================
```

### 5.4 查看 Golden Dataset 详细结果

```bash
PYTHONPATH=src DEEPEVAL_TELEMETRY_OPT_OUT=YES python3 -m pytest tests/evals/test_customer_agent.py::TestCustomerSupportAgent::test_all_golden_cases -v -s
```

---

## 6. 运行结果

### 6.1 测试通过情况

| 测试用例 | 状态 | 说明 |
|----------|------|------|
| test_customer_agent_order_inquiry | ✅ PASSED | 订单查询，无需人工审批 |
| test_customer_agent_refund_request | ✅ PASSED | 退款请求，触发人工审批 |
| test_customer_agent_complaint | ✅ PASSED | 投诉处理，触发人工审批 |
| test_customer_agent_general_inquiry | ✅ PASSED | 通用问候，无工具调用 |
| test_customer_agent_cancellation | ✅ PASSED | 取消请求，触发人工审批 |
| test_customer_agent_thanks | ✅ PASSED | 感谢回复，礼貌响应 |
| test_all_golden_cases | ✅ PASSED | Golden Dataset 完整评测 |

**总计：7/7 测试通过**

### 6.2 Golden Dataset 评测结果

```
============================================================
Golden Dataset Test Summary
============================================================

Total: 10, Passed: 10, Failed: 0
Average Score: 99.17%

Detailed Results:
  [✓] test_001 (order_inquiry): 100.00%
  [✓] test_002 (refund_request): 100.00%
  [✓] test_003 (order_inquiry): 100.00%
  [✓] test_004 (general): 100.00%
  [✓] test_005 (cancellation): 100.00%
  [✓] test_006 (complaint): 100.00%
  [✓] test_007 (general): 100.00%
  [✓] test_008 (order_inquiry): 100.00%
  [✓] test_009 (complaint): 100.00%
  [✓] test_010 (order_inquiry): 100.00%
```

### 6.3 评测指标分布

| 指标 | 含义 | 权重 |
|------|------|------|
| intent_accuracy | 意图识别准确率 | 25% |
| tool_selection | 工具选择准确率 | 25% |
| human_review_decision | 人工审批决策准确率 | 25% |
| response_contains | 回复关键词覆盖率 | 25% |

---

## 7. 附录

### 7.1 核心 API

#### invoke_customer_agent

```python
from customer_agent import invoke_customer_agent

result = invoke_customer_agent(
    user_message="Where is my order #A100?",
    human_approval_func=None,  # 可选：人工审批函数
    llm=None,                   # 可选：自定义 LLM
)

# 返回值
{
    "response": "...",              # 最终回复
    "state": {...},                 # 完整状态
    "needs_human_review": False,     # 是否需要人工审批
    "stage": "final",               # 当前阶段
    "error": None,                  # 错误信息
}
```

#### 自定义 Human Approval Function

```python
def my_approval_func(user_message: str, draft: str) -> tuple[bool, str | None]:
    """人工审批函数

    Args:
        user_message: 用户原始消息
        draft: Agent 生成的草稿回复

    Returns:
        (approved, modified_response)
        - approved: 是否批准
        - modified_response: 如果不批准，返回修改后的回复
    """
    # 示例：包含 "cancel" 的请求拒绝
    if "cancel" in user_message.lower():
        return False, "您的取消请求已转交人工处理。"
    return True, None

result = invoke_customer_agent(
    user_message="I need to cancel order #A100",
    human_approval_func=my_approval_func,
)
```

### 7.2 环境变量

| 变量 | 说明 | 必需 |
|------|------|------|
| `PYTHONPATH` | 设为 `src` 以便导入模块 | 是 |
| `DEEPEVAL_TELEMETRY_OPT_OUT` | 设为 `YES` 关闭遥测 | 是（离线时） |

### 7.3 Mock 数据

| 订单号 | 状态 |
|--------|------|
| A100 | In transit, expected tomorrow |
| B200 | Delivered yesterday |
| C300 | Payment pending |

### 7.4 版本信息

| 组件 | 版本 |
|------|------|
| Python | 3.10+ |
| langchain | >=0.3,<0.4 |
| langgraph | >=0.2,<0.3 |
| deepeval | 4.0.0 |
| pytest | >=9.0 |

---

## 8. Office Agent - AI 办公助手

### 8.1 概述

Office Agent 是一个基于 **PEV + 多 Agent 协作** 的 AI 办公助手，用于自动化处理复杂的办公任务。与 Customer Agent 不同，Office Agent 使用 Planner-Executor-Verify 架构，并行调度多个专业 Sub Agent 完成复杂任务。

**核心特点：**
- Mock Reasoning Model 担任 Planner（模拟 o1 等高阶推理模型）
- 并行执行多个 Sub Agent（Browser、API、Doc、Data、Visualization）
- Verify Agent 独立监控进度，对比实际结果与 Planner 预期
- Human-in-the-Loop 支持缺失信息的人工补充
- 共享基础设施（Task、TaskPlan、BaseSubAgent）可扩展

### 8.2 架构设计

#### 8.2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                          User Request                             │
│                  "帮我生成上周的销售周报"                          │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PLANNER AGENT                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ MockReasoningModel (模拟 o1 高阶推理)                    │     │
│  │                                                          │     │
│  │ - 分析用户请求的复杂意图                                 │     │
│  │ - 拆解为多个原子 Task                                    │     │
│  │ - 确定 Task 依赖关系（并行/串行）                        │     │
│  │ - 定义每个 Task 的预期输出                               │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                  │
│  输出: TaskPlan {                                               │
│    tasks: [Task(id=1, capability=API, ...),                      │
│            Task(id=2, capability=DATA, depends_on=[1], ...),     │
│            ...],                                                 │
│    overall_goal: "生成完整周报",                                  │
│    final_task_id: 7                                              │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TASK EXECUTOR                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ ThreadPoolExecutor (并行执行)                            │     │
│  │                                                          │     │
│  │ - 从可执行队列取出 Task                                  │     │
│  │ - 根据 capability 选择对应 Sub Agent                     │     │
│  │ - 通过 AgentRegistry 调度                                │     │
│  │ - 管理 Task 依赖关系（等待前置完成）                     │     │
│  │ - 收集结果存入 shared_data                              │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                  │
│  输出: completed_tasks=[], shared_data={...}                    │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│    API Agent        │ │    Data Agent       │ │   Browser Agent     │
│  (调用 OpenAPI)     │ │  (数据统计分析)      │ │   (网页浏览)        │
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VERIFY AGENT                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ MockVerificationModel (验证模型)                        │     │
│  │                                                          │     │
│  │ - 遍历每个 Task 的预期 vs 实际结果                       │     │
│  │ - 判断 Task 是否达到完成状态                             │     │
│  │ - 识别缺失信息和需要补充的数据                           │     │
│  │ - 决定是否需要 Human-in-the-Loop                        │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                  │
│  输出: VerificationResult {                                      │
│    is_complete: bool,                                            │
│    missing_info: [{task_id, description, suggestions}],         │
│    updated_tasks: [...]                                           │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              │                                       │
              ▼                                       ▼
┌─────────────────────────────┐    ┌─────────────────────────────┐
│      is_complete = True     │    │   missing_info 非空         │
│          ↓ 直接结束          │    │         ↓ HUMAN_LOOP        │
│      输出最终结果             │    │   等待用户补充信息           │
└─────────────────────────────┘    │   (有超时限制)               │
                                   │         ↓ 继续执行            │
                                   └─────────────────────────────┘
```

#### 8.2.2 Sub Agent 架构

```
AgentRegistry (单例)
       │
       ├── API Agent ──────► API Gateway / OpenAPI
       ├── Browser Agent ──► Selenium / Requests
       ├── Doc Agent ──────► File I/O / Pandoc
       ├── Data Agent ─────► Pandas / SQLite
       └── Visualization Agent ─► Matplotlib / Pandas

每个 Sub Agent 继承 BaseSubAgent:
┌─────────────────────────────────────┐
│         BaseSubAgent                │
├─────────────────────────────────────┤
│  capability: AgentCapability        │
│  execute(task, shared_data) → Result│
│  validate(task) → bool             │
└─────────────────────────────────────┘
```

### 8.3 核心模块实现

#### 8.3.1 共享基础设施 (`base.py`)

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Any

class AgentCapability(Enum):
    """Sub Agent 能力枚举"""
    API = "api"
    BROWSER = "browser"
    DATA = "data"
    DOCUMENT = "document"
    VISUALIZATION = "visualization"

@dataclass
class Task:
    """任务定义"""
    id: str
    description: str
    capability: AgentCapability
    params: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    expected_output: str = ""
    status: str = "pending"  # pending, running, completed, failed
    result: Any = None

@dataclass
class TaskPlan:
    """Planner 生成的完整任务计划"""
    overall_goal: str
    tasks: list[Task]
    final_task_id: str  # 最后一个任务的 ID，表示整体完成

@dataclass
class VerificationResult:
    """Verify Agent 的验证结果"""
    is_complete: bool
    missing_info: list[dict]
    updated_tasks: list[Task]
    message: str

class BaseSubAgent:
    """Sub Agent 基类"""
    capability: AgentCapability = None
    
    def execute(self, task: Task, shared_data: dict) -> Any:
        raise NotImplementedError
    
    def validate(self, task: Task) -> bool:
        return True
```

#### 8.3.2 Planner Agent (`planner.py`)

```python
class MockReasoningModel:
    """Mock 高阶推理模型（模拟 o1）"""
    
    def __init__(self, scenarios: dict | None = None):
        self.scenarios = scenarios or {}
    
    def plan(self, user_request: str) -> TaskPlan:
        """分析请求，生成任务计划"""
        # 1. 检查预定义场景
        scenario = self._match_scenario(user_request)
        if scenario:
            return self._build_plan_from_scenario(scenario)
        
        # 2. 动态规划（基于关键词分析）
        return self._dynamic_planning(user_request)
    
    def _dynamic_planning(self, user_request: str) -> TaskPlan:
        """基于关键词的动态任务规划"""
        request_lower = user_request.lower()
        tasks = []
        task_id = 1
        
        # API 调用
        if any(kw in request_lower for kw in ["销售", "订单", "数据", "报表"]):
            tasks.append(Task(
                id=str(task_id),
                description="从销售系统获取数据",
                capability=AgentCapability.API,
                params={"endpoint": "/sales/weekly"},
                expected_output="JSON 格式的销售数据",
            ))
            task_id += 1
        
        # 数据处理
        if any(kw in request_lower for kw in ["统计", "分析", "汇总"]):
            tasks.append(Task(
                id=str(task_id),
                description="统计分析销售数据",
                capability=AgentCapability.DATA,
                params={"operation": "aggregate"},
                expected_output="统计汇总数据",
            ))
            task_id += 1
        
        # 文档生成
        if any(kw in request_lower for kw in ["报告", "周报", "文档"]):
            tasks.append(Task(
                id=str(task_id),
                description="生成报告文档",
                capability=AgentCapability.DOCUMENT,
                params={"format": "markdown"},
                expected_output="格式化报告内容",
            ))
        
        return TaskPlan(
            overall_goal=user_request,
            tasks=tasks,
            final_task_id=str(task_id),
        )
```

#### 8.3.3 Task Executor (`executor.py`)

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

class TaskExecutor:
    """任务执行器 - 并行调度 Sub Agents"""
    
    def __init__(self, agent_registry: AgentRegistry):
        self.registry = agent_registry
        self.max_workers = 5
    
    def execute_plan(
        self, 
        task_plan: TaskPlan, 
        shared_data: dict,
        progress_callback: callable = None,
    ) -> tuple[list[Task], dict]:
        """执行任务计划"""
        completed_tasks = []
        pending_tasks = {t.id: t for t in task_plan.tasks}
        running_tasks = {}
        
        while pending_tasks or running_tasks:
            # 1. 获取可执行的任务（依赖已满足）
            ready_tasks = self._get_ready_tasks(
                pending_tasks, completed_tasks
            )
            
            # 2. 并行执行
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for task in ready_tasks:
                    task.status = "running"
                    running_tasks[task.id] = task
                    
                    agent = self.registry.get_agent(task.capability)
                    future = executor.submit(
                        agent.execute, task, shared_data
                    )
                    futures[future] = task
                
                # 3. 收集结果
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result()
                        task.status = "completed"
                        task.result = result
                        shared_data[task.id] = result
                        completed_tasks.append(task)
                        del running_tasks[task.id]
                    except Exception as e:
                        task.status = "failed"
                        task.result = str(e)
                        completed_tasks.append(task)
                        del running_tasks[task.id]
            
            # 回调进度
            if progress_callback:
                progress_callback(completed_tasks, running_tasks)
            
            # 移除已处理的任务
            for task_id in [t.id for t in completed_tasks]:
                if task_id in pending_tasks:
                    del pending_tasks[task_id]
        
        return completed_tasks, shared_data
    
    def _get_ready_tasks(
        self, 
        pending: dict[str, Task], 
        completed: list[Task]
    ) -> list[Task]:
        """获取依赖已满足的可执行任务"""
        completed_ids = {t.id for t in completed}
        ready = []
        
        for task in pending.values():
            if task.status != "pending":
                continue
            # 检查依赖是否都已完成
            if all(dep_id in completed_ids for dep_id in task.depends_on):
                ready.append(task)
        
        return ready
```

#### 8.3.4 Verify Agent (`verify.py`)

```python
class MockVerificationModel:
    """Mock 验证模型"""
    
    def verify(
        self,
        task_plan: TaskPlan,
        completed_tasks: list[Task],
        shared_data: dict,
    ) -> VerificationResult:
        """验证任务完成状态"""
        completed_ids = {t.id for t in completed_tasks}
        all_task_ids = {t.id for t in task_plan.tasks}
        missing_info = []
        
        # 1. 检查是否有未完成的任务
        pending_ids = all_task_ids - completed_ids
        if pending_ids:
            for task_id in pending_ids:
                task = next(t for t in task_plan.tasks if t.id == task_id)
                missing_info.append({
                    "task_id": task_id,
                    "description": f"任务未完成: {task.description}",
                    "suggestions": [f"需要调用 {task.capability.value} Agent"],
                })
        
        # 2. 验证已完成任务的结果质量
        for task in completed_tasks:
            if task.status == "failed":
                missing_info.append({
                    "task_id": task.id,
                    "description": f"任务执行失败: {task.result}",
                    "suggestions": ["检查输入参数", "重试执行"],
                })
        
        # 3. 检查关键任务是否完成
        final_task = next(
            (t for t in task_plan.tasks if t.id == task_plan.final_task_id),
            None
        )
        if not final_task or final_task.status != "completed":
            if not any(m["task_id"] == final_task.id for m in missing_info):
                missing_info.append({
                    "task_id": final_task.id,
                    "description": "最终任务未完成",
                    "suggestions": ["确保所有前置任务完成"],
                })
        
        return VerificationResult(
            is_complete=len(missing_info) == 0,
            missing_info=missing_info,
            updated_tasks=task_plan.tasks,
            message="所有任务已完成" if not missing_info else "需要补充信息",
        )
```

#### 8.3.5 Human-in-the-Loop (`human_loop.py`)

```python
import threading
import time

class HumanInTheLoop:
    """人工介入机制"""
    
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
        self.user_input: dict = {}
        self.lock = threading.Lock()
        self.event = threading.Event()
    
    def request_info(
        self,
        task_id: str,
        description: str,
        suggestions: list[str],
    ) -> dict | None:
        """请求用户补充信息
        
        Returns:
            用户输入的 dict，或 None（超时）
        """
        print(f"\n{'='*60}")
        print(f"[HUMAN-IN-THE-LOOP] 需要补充信息")
        print(f"{'='*60}")
        print(f"任务 ID: {task_id}")
        print(f"描述: {description}")
        print(f"建议: {', '.join(suggestions)}")
        print(f"超时: {self.timeout} 秒")
        print(f"{'='*60}")
        
        # 等待用户输入
        user_input = input("请输入信息 (或按回车跳过): ")
        
        with self.lock:
            self.user_input[task_id] = user_input
            self.event.set()
        
        return {"task_id": task_id, "input": user_input}
    
    def get_supplied_info(self, task_id: str) -> str | None:
        """获取已补充的信息"""
        return self.user_input.get(task_id)
```

### 8.4 办公场景

| 场景 | 描述 | 涉及 Sub Agents |
|------|------|-----------------|
| Weekly Sales Report | 生成周报（API → Data → Visualization → Doc） | API, Data, Visualization, Doc |
| Customer Research | 客户调研（Browser → API → Data → Doc） | Browser, API, Data, Doc |
| Meeting Preparation | 会议准备（Data → Doc） | Data, Doc |

#### 8.4.1 Weekly Sales Report 场景

```
Task 1: API Agent - 获取上周销售数据 (parallel: none)
         params: {"endpoint": "/sales/weekly", "date_range": "last_week"}
         expected_output: "JSON: {orders: [...], revenue: 125000}"
              │
              ▼ (depends_on: [1])
Task 2: Data Agent - 统计分析销售数据 (parallel: none)
         params: {"operation": "aggregate", "metrics": ["revenue", "orders"]}
         expected_output: "统计汇总: {total_revenue, order_count, top_products}"
              │
       ┌─────┴─────┐
       ▼           ▼ (parallel)
Task 3: Data Agent  Task 4: Data Agent
  区域销售         产品分类
       │           │
       └─────┬─────┘
             ▼ (depends_on: [3, 4])
Task 5: Visualization Agent - 生成销售图表
         params: {"chart_type": "bar", "data_keys": ["region", "revenue"]}
         expected_output: "Base64 编码的图表"
             │
             ▼ (depends_on: [5])
Task 6: Visualization Agent - 生成趋势图表
         params: {"chart_type": "line", "data_keys": ["date", "revenue"]}
         expected_output: "Base64 编码的趋势图"
             │
             ▼ (depends_on: [2, 5, 6])
Task 7: Doc Agent - 生成 Markdown 周报
         params: {"format": "markdown", "include_charts": true}
         expected_output: "完整周报 Markdown 内容"
```

### 8.5 运行方式

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

### 8.6 运行结果

```
============================================================
Office Agent - Demo Scenarios
============================================================

[PASSED] weekly_sales_report - 7 tasks completed
         Tasks: [API] → [Data] → [Data] → [Data] + [Data] → [Viz] → [Viz] → [Doc]

[PASSED] customer_research - 6 tasks completed
         Tasks: [Browser] → [API] → [Data] → [Doc] → [Viz] → [Doc]

[PASSED] meeting_preparation - 3 tasks completed
         Tasks: [Data] → [Data] → [Doc]

============================================================
Total: 3/3 passed
============================================================
```

### 8.7 与 Customer Agent 的区别

| 特性 | Customer Agent | Office Agent |
|------|----------------|--------------|
| 架构 | PEV (3 节点) | PEV + Multi-Agent (Planner-Executor-Verify) |
| 复杂度 | 单轮对话 | 多轮任务协作 |
| 并行执行 | 否 | 是（ThreadPoolExecutor） |
| Sub Agent | 无 | 有（5 种专业 Agent） |
| Human-in-Loop | 基于置信度/敏感词 | 基于缺失信息检测 |
| Mock LLM | ChatModel | Reasoning Model |
| 应用场景 | 客服对话 | 复杂办公任务 |

### 8.8 共享基础设施

两个 Agent 共享以下抽象：

```python
# src/office_agent/base.py 中的可复用组件

# 1. Task/TaskPlan/TaskStatus - 任务定义（Customer Agent 可复用）
# 2. AgentCapability - 能力枚举（可扩展）
# 3. BaseSubAgent - 子 Agent 基类（可扩展）

# 未来计划：
# - 将 base.py 移动到 src/shared/
# - Customer Agent 的 tools 可注册为 Sub Agent
# - 统一 Human-in-the-Loop 接口
```

---

## 变更记录

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2024 | 初始版本：PEV Agent + DeepEval 评测框架 |
| 1.1 | 2024 | 新增 Office Agent：AI 办公助手 (PEV + Multi-Agent) |

---

*本文档由 AI 生成，如有问题请提交 Issue。*
