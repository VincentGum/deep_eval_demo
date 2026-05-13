from __future__ import annotations

import json
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from customer_agent import invoke_customer_agent


DATASET_PATH = Path(__file__).with_name("customer_support_goldens.json")


class LocalCustomerSupportMetric(BaseMetric):
    def __init__(self, name: str, scorer, threshold: float = 1.0):
        self.name = name
        self.scorer = scorer
        self.threshold = threshold
        self.async_mode = False
        self.include_reason = True

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        score, reason = self.scorer(test_case)
        self.score = score
        self.reason = reason
        self.success = score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return self.name


def _metadata(test_case: LLMTestCase) -> dict:
    return test_case.metadata or {}


def _intent_score(test_case: LLMTestCase):
    metadata = _metadata(test_case)
    passed = metadata["actual_intent"] == metadata["expected_intent"]
    return (1.0 if passed else 0.0, f"intent={metadata['actual_intent']}")


def _tool_score(test_case: LLMTestCase):
    metadata = _metadata(test_case)
    passed = metadata["actual_tools"] == metadata["expected_tools"]
    return (1.0 if passed else 0.0, f"tools={metadata['actual_tools']}")


def _human_loop_score(test_case: LLMTestCase):
    metadata = _metadata(test_case)
    passed = metadata["actual_requires_human"] == metadata["requires_human"]
    return (1.0 if passed else 0.0, f"requires_human={metadata['actual_requires_human']}")


def _response_score(test_case: LLMTestCase):
    metadata = _metadata(test_case)
    output = test_case.actual_output or ""
    missing = [text for text in metadata["must_include"] if text not in output]
    return (0.0 if missing else 1.0, f"missing={missing}")


END_TO_END_METRICS = [
    LocalCustomerSupportMetric("Intent Routing", _intent_score),
    LocalCustomerSupportMetric("Tool Plan Correctness", _tool_score),
    LocalCustomerSupportMetric("Human Loop Policy", _human_loop_score),
    LocalCustomerSupportMetric("Response Contract", _response_score),
]


def _load_goldens() -> list[dict]:
    return json.loads(DATASET_PATH.read_text())


@pytest.mark.parametrize("golden", _load_goldens())
def test_customer_support_agent_offline(golden: dict) -> None:
    human_decision = None
    if golden["requires_human"]:
        human_decision = {"approved": True, "note": "Approved by offline evaluator."}

    result = invoke_customer_agent(
        golden["input"],
        human_decision=human_decision,
        thread_id=f"eval-{golden['expected_intent']}",
    )

    actual_tools = [entry["tool"] for entry in result.get("tool_trace", [])]
    test_case = LLMTestCase(
        input=golden["input"],
        actual_output=result["final_response"],
        expected_output="; ".join(golden["must_include"]),
        metadata={
            **golden,
            "actual_intent": result["intent"],
            "actual_tools": actual_tools,
            "actual_requires_human": result["requires_human"],
            "verification_reason": result["verification_reason"],
        },
    )
    assert_test(test_case=test_case, metrics=END_TO_END_METRICS, run_async=False)
