#!/usr/bin/env python3
"""Demo script for the PEV Customer Support Agent.

This script demonstrates the agent running with mock LLM and tools.
Run with: PYTHONPATH=src python examples/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from customer_agent import invoke_customer_agent


def main():
    """Run demo conversations with the customer support agent."""

    print("=" * 60)
    print("PEV Customer Support Agent - Demo")
    print("=" * 60)
    print()

    # Test cases
    test_cases = [
        {
            "name": "Order Status Inquiry",
            "message": "Hi, can you tell me the status of my order #A100?",
        },
        {
            "name": "Refund Request",
            "message": "I want to request a refund for order #B200, the item was damaged.",
        },
        {
            "name": "Delivery Question",
            "message": "Where is my order #C300?",
        },
        {
            "name": "General Greeting",
            "message": "Hello, I need some help.",
        },
        {
            "name": "Sensitive: Cancellation",
            "message": "I need to cancel order #A100 immediately.",
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─' * 60}")
        print(f"Test {i}: {test_case['name']}")
        print(f"User: {test_case['message']}")
        print(f"{'─' * 60}")

        # Invoke agent
        result = invoke_customer_agent(test_case["message"])

        # Display result
        print(f"\nAgent Response:")
        print(result["response"])

        print(f"\n[Debug Info]")
        print(f"  Intent: {result['intent']}")
        print(f"  Tools Called: {result['tools_called']}")
        print(f"  Human Review Required: {result['human_review']}")
        if result['human_review_reason']:
            print(f"  Human Review Reason: {result['human_review_reason']}")
        print(f"  Is Sensitive: {result['is_sensitive']}")
        print(f"  Confidence: {result['confidence']:.2f}")

        print()

    print("=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
