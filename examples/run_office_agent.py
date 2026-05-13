#!/usr/bin/env python3
"""Office Agent Demo - Interactive demonstration of the PEV-based office automation agent.

This script demonstrates the Office Agent's capabilities with three pre-configured
scenarios:
1. Weekly Sales Report Generation
2. Customer Research Report
3. Meeting Preparation Pack

Usage:
    PYTHONPATH=src python3 examples/run_office_agent.py

Options:
    --scenario=<name>  Run a specific scenario (weekly_sales_report, customer_research, meeting_preparation)
    --list            List available scenarios
    --all             Run all scenarios
    --interactive     Run in interactive mode
"""

import argparse
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from office_agent import OfficeAgent, WorkflowResult
from office_agent.scenarios import (
    WEEKLY_SALES_REPORT,
    CUSTOMER_RESEARCH,
    MEETING_PREPARATION,
    list_scenarios,
    get_scenario,
    run_scenario,
    demo_all,
)


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_result(result: WorkflowResult) -> None:
    """Print the workflow result in a formatted way."""
    print("\n--- Workflow Result ---")
    print(f"Success: {result.success}")
    print(f"Execution Time: {result.execution_time_seconds:.2f}s")
    print(f"Final State: {result.final_state.value}")

    # Task summary
    print("\n--- Task Summary ---")
    for task in result.plan.tasks:
        status_icon = {
            "completed": "[OK]",
            "pending": "[..]",
            "running": "[>>]",
            "failed": "[!!]",
            "waiting_human_input": "[??]",
        }.get(task.status.value, "[--]")

        print(f"  {status_icon} {task.capability_required}: {task.description}")
        if task.result and task.result.output:
            output_preview = str(task.result.output)[:80]
            if len(str(task.result.output)) > 80:
                output_preview += "..."
            print(f"       -> {output_preview}")
        elif task.result and task.result.error:
            print(f"       -> ERROR: {task.result.error}")

    # Human inputs
    if result.human_inputs:
        print(f"\n--- Human Inputs ({len(result.human_inputs)}) ---")
        for human_input in result.human_inputs:
            print(f"  Task: {human_input.task_id}")
            print(f"  Status: {human_input.status.value}")
            if human_input.response:
                print(f"  Response: {human_input.response[:100]}...")

    # Summary
    if result.output.get("summary"):
        print(f"\n--- AI Summary ---")
        print(result.output["summary"])


def interactive_mode():
    """Run in interactive mode."""
    print_header("Office Agent - Interactive Mode")
    print("\nWelcome! I am your AI Office Assistant.")
    print("I can help you with various office tasks using my PEV workflow.")
    print("\nExample requests:")
    print("  - Generate a weekly sales report")
    print("  - Research our top customers")
    print("  - Prepare a meeting pack for my quarterly review")
    print("\nType 'exit' or 'quit' to end the session.\n")

    agent = OfficeAgent()

    while True:
        try:
            user_input = input("\nYour request: ").strip()

            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            print()
            result = agent.execute(user_input)
            print_result(result)

        except KeyboardInterrupt:
            print("\n\nSession ended.")
            break
        except Exception as e:
            print(f"\nError: {e}")


def run_specific_scenario(scenario_name: str) -> None:
    """Run a specific scenario by name."""
    scenario = get_scenario(scenario_name)

    if scenario:
        print_header(f"Running Scenario: {scenario.name}")
        print(f"\nDescription: {scenario.description}")
        print(f"\nUser Request: {scenario.user_request}")
        print()
        result = run_scenario(scenario, verbose=True)
        print_result(result)
    else:
        print(f"Unknown scenario: {scenario_name}")
        print("\nAvailable scenarios:")
        for s in list_scenarios():
            print(f"  - {s.name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Office Agent Demo - PEV-based office automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all              Run all scenarios
  %(prog)s --scenario=weekly_sales_report
  %(prog)s --interactive      Interactive mode
  %(prog)s --list            List available scenarios
        """
    )

    parser.add_argument(
        "--scenario",
        "-s",
        type=str,
        help="Run a specific scenario"
    )

    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Run all scenarios"
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode"
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available scenarios"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.list:
        print_header("Available Scenarios")
        for scenario in list_scenarios():
            print(f"\n  {scenario.name}")
            print(f"  Description: {scenario.description}")
            print(f"  Expected Tasks: {', '.join(scenario.expected_tasks[:3])}...")
        return

    if args.interactive:
        interactive_mode()
        return

    if args.all:
        print_header("Running All Scenarios")
        results = demo_all()

        # Summary table
        print("\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        print(f"  {'Scenario':<35} {'Status':<10}")
        print("-" * 70)
        for name, success in results:
            status = "PASSED" if success else "FAILED"
            print(f"  {name:<35} {status:<10}")
        print("=" * 70)

        passed = sum(1 for _, s in results if s)
        print(f"\nTotal: {passed}/{len(results)} passed")
        return

    if args.scenario:
        run_specific_scenario(args.scenario)
        return

    # Default: run all scenarios
    print("No options specified. Running all scenarios...")
    print("Use --help for more options.")
    print()
    args.all = True
    run_all()


if __name__ == "__main__":
    main()
