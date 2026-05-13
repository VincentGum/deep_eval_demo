from __future__ import annotations

from pprint import pprint

from customer_agent import invoke_customer_agent


def main() -> None:
    print("Auto-approved order status example:")
    pprint(invoke_customer_agent("Where is order A100?", thread_id="demo-status"))

    print("\nHuman-in-the-loop refund example:")
    first_pass = invoke_customer_agent(
        "I want a refund for order B200 because it arrived damaged.",
        thread_id="demo-refund",
    )
    pprint(first_pass)

    print("\nResuming with human approval:")
    approved = invoke_customer_agent(
        "I want a refund for order B200 because it arrived damaged.",
        human_decision={"approved": True, "note": "Refund follows policy."},
        thread_id="demo-refund-approved",
    )
    pprint(approved)


if __name__ == "__main__":
    main()
