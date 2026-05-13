"""DeepEval test cases for the PEV Customer Support Agent.

Run with:
    PYTHONPATH=src DEEPEVAL_TELEMETRY_OPT_OUT=YES deepeval test run tests/evals/test_customer_agent.py

Note: DeepEval 4.0 uses 'deepeval test run' command, not pytest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from customer_agent import invoke_customer_agent


# --- Load Golden Dataset ---

GOLDEN_DATA_PATH = Path(__file__).parent / "customer_support_goldens.json"


def load_golden_data() -> list[dict[str, Any]]:
    """Load golden test data from JSON file."""
    with open(GOLDEN_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# --- Custom Metrics (Offline Mode) ---

class IntentAccuracyMetric(BaseMetric):
    """DeepEval metric: Evaluate if the agent correctly identified the user intent."""

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.error = None
        self.reason = None

    def measure(self, test_case: LLMTestCase) -> float:
        """Synchronous measure implementation."""
        return self._evaluate(test_case)

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """Asynchronous measure implementation."""
        return self._evaluate(test_case)

    def _evaluate(self, test_case: LLMTestCase) -> float:
        """Evaluate intent matching.

        Context[0] = expected intent (e.g., "order_inquiry", "refund")
        """
        expected_intent = test_case.context[0] if test_case.context else ""
        actual_output = test_case.actual_output

        # Map intent keywords to expected output patterns
        intent_keywords = {
            # Note: Golden Dataset uses "order_status" but other tests may use "order_inquiry"
            "order_status": ["订单", "order", "发货", "status", "look up", "#"],
            "order_inquiry": ["订单", "order", "发货", "status", "look up", "#"],
            "refund": ["退款", "refund", "退"],
            "cancel_order": ["取消", "cancel"],
            "complaint": ["投诉", "complaint", "不满"],
            "greeting": ["你好", "hello", "hi", "help"],
            "thanks": ["welcome", "you're welcome", "anything else"],
            "generic_question": ["customer service", "main website", "帮助", "help with"]
        }

        keywords = intent_keywords.get(expected_intent, [expected_intent])
        success = any(kw.lower() in actual_output.lower() for kw in keywords)

        if success:
            self.error = None
            self.reason = f"Intent '{expected_intent}' correctly identified"
            return 1.0
        else:
            self.error = f"Intent '{expected_intent}' not identified in output"
            self.reason = self.error
            return 0.0

    def is_successful(self) -> bool:
        """Return True if the last evaluation passed."""
        return self.error is None


class ToolSelectionMetric(BaseMetric):
    """DeepEval metric: Evaluate if the agent selected the correct tools."""

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.error = None
        self.reason = None

    def measure(self, test_case: LLMTestCase) -> float:
        """Synchronous measure implementation."""
        return self._evaluate(test_case)

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """Asynchronous measure implementation."""
        return self._evaluate(test_case)

    def _evaluate(self, test_case: LLMTestCase) -> float:
        """Evaluate tool selection.

        Context[1] = expected tools (e.g., "lookup_order" or "none")
        
        Note: Since agent output is natural language (not tool call traces),
        we check if the output contains expected information related to the tools.
        """
        expected_tools_str = test_case.context[1] if len(test_case.context) > 1 else "none"
        actual_output = test_case.actual_output.lower()

        # Parse expected tools
        if expected_tools_str.lower() == "none":
            # No tools expected - check output doesn't contain tool call traces
            tool_traces = ["look up your order", "created case", "refund case", "工单已创建"]
            if any(trace in actual_output for trace in tool_traces):
                self.error = "Unexpected tool usage when no tools expected"
                self.reason = self.error
                return 0.0
            self.error = None
            self.reason = "No tools required, agent correctly handled without tools"
            return 1.0

        expected_tools_list = [t.strip() for t in expected_tools_str.split(",")]

        # Check if output contains information related to expected tools
        # We check for actual content/results rather than tool call keywords
        # For lookup_order: check for specific order OR intent phrases
        tool_result_keywords = {
            "lookup_order": ["order #", "a100", "b200", "c300", "in transit", "delivered", 
                             "expected tomorrow", "pending", "payment", "order a100", 
                             "order b200", "order c300", "look up your order", "checking order",
                             "order status", "查询订单", "订单信息"],
            "create_refund_case": ["refund case", "draft-", "case created", "complaint case",
                                   "退款工单", "created for review", "工单已创建"]
        }

        all_present = True
        for tool in expected_tools_list:
            keywords = tool_result_keywords.get(tool, [tool])
            if not any(kw.lower() in actual_output for kw in keywords):
                all_present = False
                break

        if all_present:
            self.error = None
            self.reason = f"Correct tools selected: {expected_tools_list}"
            return 1.0
        else:
            self.error = f"Missing tools. Expected: {expected_tools_list}"
            self.reason = self.error
            return 0.0

    def is_successful(self) -> bool:
        """Return True if the last evaluation passed."""
        return self.error is None


class HumanReviewDecisionMetric(BaseMetric):
    """DeepEval metric: Evaluate if the agent correctly decided on human review."""

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.error = None
        self.reason = None

    def measure(self, test_case: LLMTestCase) -> float:
        """Synchronous measure implementation."""
        return self._evaluate(test_case)

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """Asynchronous measure implementation."""
        return self._evaluate(test_case)

    def _evaluate(self, test_case: LLMTestCase) -> float:
        """Evaluate human review decision.

        Context[2] = requires_human_review (e.g., "true" or "false")
        """
        expected_review = test_case.context[2] if len(test_case.context) > 2 else "false"
        expected_review_flag = expected_review.lower() == "true"
        actual_output = test_case.actual_output

        # Check if actual output indicates human review was triggered
        # Note: "pending" is removed as it's a generic status word, not specific to human review
        human_review_keywords = [
            "[WAITING FOR HUMAN APPROVAL]",
            "[HUMAN REVIEW REQUIRED]",
            "awaiting approval",
            "工单已创建",
            "refund case created",
            "created for review",
            "DRAFT-"
        ]
        human_review_triggered = any(kw in actual_output for kw in human_review_keywords)

        success = human_review_triggered == expected_review_flag

        if success:
            self.error = None
            action = "triggered" if human_review_triggered else "not triggered"
            self.reason = f"Human review decision correct: {action}"
            return 1.0
        else:
            self.error = f"Human review mismatch: expected {expected_review_flag}, got {human_review_triggered}"
            self.reason = self.error
            return 0.0

    def is_successful(self) -> bool:
        """Return True if the last evaluation passed."""
        return self.error is None


# --- Helper Function ---

def run_customer_agent_test(user_message: str) -> dict[str, Any]:
    """Run the customer agent and return results."""

    def mock_human_approval(user_message: str, draft_response: str) -> tuple[bool, str | None]:
        """Mock human approval function with correct signature."""
        return True, None  # Auto-approve for testing

    try:
        result = invoke_customer_agent(user_message, mock_human_approval)
        return {
            "success": True,
            "actual_output": result.get("response", ""),
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "actual_output": "",
            "error": str(e)
        }


# --- DeepEval Test Cases ---

def test_order_inquiry():
    """Test: Customer asks about order status - no human review needed."""
    user_message = "我的订单什么时候发货？"
    # Context: [expected_intent, expected_tools, requires_human_review]
    context = ["order_inquiry", "lookup_order", "false"]

    result = run_customer_agent_test(user_message)

    test_case = LLMTestCase(
        input=user_message,
        actual_output=result["actual_output"],
        context=context
    )

    metrics = [
        IntentAccuracyMetric(threshold=0.5),
        ToolSelectionMetric(threshold=0.5),
        HumanReviewDecisionMetric(threshold=0.5)
    ]

    assert_test(test_case, metrics)


def test_refund_request():
    """Test: Customer requests refund - human review required."""
    user_message = "我要退款"
    context = ["refund", "create_refund_case", "true"]

    result = run_customer_agent_test(user_message)

    test_case = LLMTestCase(
        input=user_message,
        actual_output=result["actual_output"],
        context=context
    )

    metrics = [
        IntentAccuracyMetric(threshold=0.5),
        ToolSelectionMetric(threshold=0.5),
        HumanReviewDecisionMetric(threshold=0.5)
    ]

    assert_test(test_case, metrics)


def test_order_cancellation():
    """Test: Customer requests order cancellation - human review required."""
    user_message = "我要取消订单"
    context = ["cancel_order", "create_refund_case", "true"]

    result = run_customer_agent_test(user_message)

    test_case = LLMTestCase(
        input=user_message,
        actual_output=result["actual_output"],
        context=context
    )

    metrics = [
        IntentAccuracyMetric(threshold=0.5),
        ToolSelectionMetric(threshold=0.5),
        HumanReviewDecisionMetric(threshold=0.5)
    ]

    assert_test(test_case, metrics)


def test_complaint():
    """Test: Customer files complaint - human review required."""
    user_message = "我要投诉你们的服务"
    context = ["complaint", "create_refund_case", "true"]

    result = run_customer_agent_test(user_message)

    test_case = LLMTestCase(
        input=user_message,
        actual_output=result["actual_output"],
        context=context
    )

    metrics = [
        IntentAccuracyMetric(threshold=0.5),
        ToolSelectionMetric(threshold=0.5),
        HumanReviewDecisionMetric(threshold=0.5)
    ]

    assert_test(test_case, metrics)


def test_greeting():
    """Test: Customer sends greeting - no tools, no human review."""
    user_message = "你好"
    context = ["greeting", "none", "false"]

    result = run_customer_agent_test(user_message)

    test_case = LLMTestCase(
        input=user_message,
        actual_output=result["actual_output"],
        context=context
    )

    metrics = [
        IntentAccuracyMetric(threshold=0.5),
        ToolSelectionMetric(threshold=0.5),
        HumanReviewDecisionMetric(threshold=0.5)
    ]

    assert_test(test_case, metrics)


def test_generic_question():
    """Test: Customer asks generic question - no tools needed."""
    user_message = "今天天气怎么样？"
    context = ["generic_question", "none", "false"]

    result = run_customer_agent_test(user_message)

    test_case = LLMTestCase(
        input=user_message,
        actual_output=result["actual_output"],
        context=context
    )

    metrics = [
        IntentAccuracyMetric(threshold=0.5),
        ToolSelectionMetric(threshold=0.5),
        HumanReviewDecisionMetric(threshold=0.5)
    ]

    assert_test(test_case, metrics)


def test_offline_evaluation_summary():
    """Test: Summary of all offline evaluation metrics.

    This test runs all golden data and reports summary.
    """
    golden_data = load_golden_data()

    total = len(golden_data)
    passed = 0
    failed = 0
    results = []

    for item in golden_data:
        user_message = item["input"]
        context = [
            item["expected_intent"],
            ",".join(item["expected_tools"]) if item["expected_tools"] else "none",
            str(item.get("expected_human_review", False)).lower()
        ]

        result = run_customer_agent_test(user_message)

        test_case = LLMTestCase(
            input=user_message,
            actual_output=result["actual_output"],
            context=context
        )

        metrics = [
            IntentAccuracyMetric(threshold=0.5),
            ToolSelectionMetric(threshold=0.5),
            HumanReviewDecisionMetric(threshold=0.5)
        ]

        try:
            assert_test(test_case, metrics, run_async=False)
            passed += 1
            results.append({"case": item["input"][:30], "status": "PASSED"})
        except AssertionError:
            failed += 1
            results.append({"case": item["input"][:30], "status": "FAILED"})

    print(f"\n{'='*60}")
    print(f"Offline Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"Pass Rate: {passed/total*100:.2f}%")
    print(f"{'='*60}")

    for r in results:
        status_symbol = "[PASSED]" if r["status"] == "PASSED" else "[FAILED]"
        print(f"  {status_symbol} {r['case']}...")

    assert passed == total, f"Expected all tests to pass, but {failed} failed"
