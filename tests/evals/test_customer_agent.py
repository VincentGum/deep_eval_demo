"""DeepEval test cases for the PEV Customer Support Agent.

Run with:
    PYTHONPATH=src DEEPEVAL_TELEMETRY_OPT_OUT=YES python -m deepeval test run tests/evals/test_customer_agent.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from customer_agent import invoke_customer_agent


# --- Load Golden Dataset ---

GOLDEN_DATA_PATH = Path(__file__).parent / "customer_support_goldens.json"


def load_golden_data() -> list[dict[str, Any]]:
    """Load golden test data from JSON file."""
    with open(GOLDEN_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# --- Custom Metrics for Offline Mode ---

class OfflineMetric:
    """Base class for offline evaluation metrics."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.success = False
        self.score = 0.0
        self.reason = ""

    def evaluate(self, **kwargs) -> dict[str, Any]:
        raise NotImplementedError


class IntentAccuracyMetric(OfflineMetric):
    """Evaluate if the agent correctly identified the user intent."""

    def __init__(self, threshold: float = 0.8):
        super().__init__(threshold)

    def evaluate(
        self,
        actual_intent: str,
        expected_intent: str,
        **kwargs
    ) -> dict[str, Any]:
        # Exact match for offline mode
        self.success = actual_intent == expected_intent
        self.score = 1.0 if self.success else 0.0
        self.reason = (
            f"Intent matched: {actual_intent} == {expected_intent}"
            if self.success
            else f"Intent mismatch: {actual_intent} != {expected_intent}"
        )
        return {"success": self.success, "score": self.score, "reason": self.reason}


class ToolSelectionMetric(OfflineMetric):
    """Evaluate if the agent selected the correct tools."""

    def __init__(self, threshold: float = 0.8):
        super().__init__(threshold)

    def evaluate(
        self,
        actual_tools: list[str],
        expected_tools: list[str],
        **kwargs
    ) -> dict[str, Any]:
        # Check if expected tools are a subset of actual tools
        if not expected_tools:
            # No tools expected - any response is fine
            self.success = True
            self.score = 1.0
            self.reason = "No tools required, agent correctly handled without tools"
        else:
            all_present = all(tool in actual_tools for tool in expected_tools)
            self.success = all_present
            self.score = 1.0 if all_present else 0.0
            self.reason = (
                f"Correct tools selected: {actual_tools}"
                if self.success
                else f"Missing tools. Expected: {expected_tools}, Got: {actual_tools}"
            )
        return {"success": self.success, "score": self.score, "reason": self.reason}


class HumanReviewDecisionMetric(OfflineMetric):
    """Evaluate if the agent correctly decided on human review."""

    def __init__(self, threshold: float = 0.9):
        super().__init__(threshold)

    def evaluate(
        self,
        actual_human_review: bool,
        expected_human_review: bool,
        **kwargs
    ) -> dict[str, Any]:
        self.success = actual_human_review == expected_human_review
        self.score = 1.0 if self.success else 0.0
        self.reason = (
            "Correct human review decision"
            if self.success
            else f"Incorrect human review decision. Expected: {expected_human_review}, Got: {actual_human_review}"
        )
        return {"success": self.success, "score": self.score, "reason": self.reason}


class ResponseContainsMetric(OfflineMetric):
    """Evaluate if the response contains expected keywords."""

    def __init__(self, threshold: float = 0.7):
        super().__init__(threshold)

    def evaluate(
        self,
        response: str,
        expected_contains: list[str],
        **kwargs
    ) -> dict[str, Any]:
        if not expected_contains:
            self.success = True
            self.score = 1.0
            self.reason = "No keyword requirements"
            return {"success": self.success, "score": self.score, "reason": self.reason}

        response_lower = response.lower()
        matched = sum(1 for keyword in expected_contains if keyword.lower() in response_lower)
        self.score = matched / len(expected_contains)
        self.success = self.score >= self.threshold
        self.reason = f"Found {matched}/{len(expected_contains)} expected keywords"

        return {"success": self.success, "score": self.score, "reason": self.reason}


# --- DeepEval Test Cases ---

class TestCustomerSupportAgent:
    """Test suite for PEV Customer Support Agent."""

    @pytest.fixture
    def golden_data(self) -> list[dict[str, Any]]:
        """Load golden test data."""
        return load_golden_data()

    @pytest.fixture
    def metrics(self):
        """Initialize metrics."""
        return {
            "intent": IntentAccuracyMetric(),
            "tools": ToolSelectionMetric(),
            "human_review": HumanReviewDecisionMetric(),
            "response_contains": ResponseContainsMetric(),
        }

    def test_customer_agent_order_inquiry(self):
        """Test: Order status inquiry - should use lookup_order tool."""
        result = invoke_customer_agent("Hi, can you tell me the status of my order #A100?")

        # Check response exists
        assert result["response"], "Agent should return a response"

        # Check no errors
        assert not result["error"], f"Agent should not have errors: {result['error']}"

        # Check correct tools used
        plan = result["state"].get("plan")
        if plan:
            assert "lookup_order" in plan.tools_to_use, "Should use lookup_order for order inquiry"

        # Check no human review required
        assert not result["needs_human_review"], "Order inquiry should not require human review"

    def test_customer_agent_refund_request(self):
        """Test: Refund request - should require human review."""
        result = invoke_customer_agent(
            "I want to request a refund for order #B200, the item was damaged."
        )

        # Check response exists
        assert result["response"], "Agent should return a response"

        # Check human review is required for refunds
        assert result["needs_human_review"], "Refund requests should require human review"

        # Check refund case was created
        execute = result["state"].get("execute")
        if execute:
            tool_names = [call["tool"] for call in execute.tool_calls]
            assert "create_refund_case" in tool_names, "Should create refund case"

    def test_customer_agent_complaint(self):
        """Test: Complaint - should require human review."""
        result = invoke_customer_agent(
            "I want to complain about my order #A100, it arrived late."
        )

        # Check response exists
        assert result["response"], "Agent should return a response"

        # Check human review is required for complaints
        assert result["needs_human_review"], "Complaints should require human review"

    def test_customer_agent_general_inquiry(self):
        """Test: General greeting - should not use tools."""
        result = invoke_customer_agent("Hello, I need some help.")

        # Check response exists
        assert result["response"], "Agent should return a response"

        # Check no tools used
        plan = result["state"].get("plan")
        if plan:
            assert len(plan.tools_to_use) == 0, "General inquiry should not use tools"

        # Check no human review required
        assert not result["needs_human_review"], "General inquiry should not require human review"

    def test_customer_agent_cancellation(self):
        """Test: Cancellation request - should require human review."""
        result = invoke_customer_agent("I need to cancel order #A100 immediately.")

        # Check response exists
        assert result["response"], "Agent should return a response"

        # Check human review is required for cancellations
        assert result["needs_human_review"], "Cancellation requests should require human review"

    def test_customer_agent_thanks(self):
        """Test: Thanks message - should respond politely without tools."""
        result = invoke_customer_agent("Thanks for your help!")

        # Check response exists
        assert result["response"], "Agent should return a response"

        # Check no tools used
        plan = result["state"].get("plan")
        if plan:
            assert len(plan.tools_to_use) == 0, "Thanks should not use tools"

        # Check response contains acknowledgment
        assert "welcome" in result["response"].lower() or "help" in result["response"].lower(), \
            "Response should acknowledge thanks"

    def test_all_golden_cases(self, golden_data, metrics):
        """Test all golden dataset cases."""
        results = []

        for test_case in golden_data:
            result = invoke_customer_agent(test_case["input"])
            plan = result["state"].get("plan")

            # Evaluate each metric
            metric_results = {}

            # Intent accuracy
            if plan:
                intent_result = metrics["intent"].evaluate(
                    actual_intent=plan.intent,
                    expected_intent=test_case.get("expected_intent", "unknown"),
                )
                metric_results["intent_accuracy"] = intent_result

            # Tool selection
            if plan:
                tools_result = metrics["tools"].evaluate(
                    actual_tools=plan.tools_to_use,
                    expected_tools=test_case.get("expected_tools", []),
                )
                metric_results["tool_selection"] = tools_result

            # Human review decision
            hr_result = metrics["human_review"].evaluate(
                actual_human_review=result["needs_human_review"],
                expected_human_review=test_case.get("expected_human_review", False),
            )
            metric_results["human_review_decision"] = hr_result

            # Response contains keywords
            response_result = metrics["response_contains"].evaluate(
                response=result["response"],
                expected_contains=test_case.get("expected_response_contains", []),
            )
            metric_results["response_contains"] = response_result

            # Calculate overall score
            scores = [r["score"] for r in metric_results.values()]
            overall_score = sum(scores) / len(scores) if scores else 0.0

            results.append({
                "test_id": test_case["id"],
                "category": test_case.get("category", "unknown"),
                "metrics": metric_results,
                "overall_score": overall_score,
                "success": all(r["success"] for r in metric_results.values()),
            })

        # Print summary
        print("\n" + "=" * 60)
        print("Golden Dataset Test Summary")
        print("=" * 60)

        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["success"])
        avg_score = sum(r["overall_score"] for r in results) / total_tests if total_tests else 0

        print(f"\nTotal: {total_tests}, Passed: {passed_tests}, Failed: {total_tests - passed_tests}")
        print(f"Average Score: {avg_score:.2%}")

        print("\nDetailed Results:")
        for r in results:
            status = "✓" if r["success"] else "✗"
            print(f"  [{status}] {r['test_id']} ({r['category']}): {r['overall_score']:.2%}")
            if not r["success"]:
                for metric_name, metric_result in r["metrics"].items():
                    if not metric_result["success"]:
                        print(f"      - {metric_name}: {metric_result['reason']}")

        print("=" * 60)

        # Assert all tests passed
        assert passed_tests == total_tests, f"{total_tests - passed_tests} tests failed"


# --- Standalone Run Support ---

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    # Run tests with pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
