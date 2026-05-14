#!/usr/bin/env python3
"""
Office Agent 演示脚本 - 基于 PEV 架构的智能办公助手

本脚本演示 Office Agent 处理各种办公自动化任务的能力。

核心功能：
1. 周销售报告生成 - 自动获取数据、生成图表、输出报告
2. 客户调研报告 - 查询客户信息、爬取网站、整合报告
3. 会议准备包 - 整理会议资料、议程、参会人信息

运行方式：
    PYTHONPATH=src python3 examples/run_office_agent.py

命令行选项：
    --scenario=<name>  运行指定场景
                      可选值：weekly_sales_report, customer_research, meeting_preparation
    --list            列出所有可用场景
    --all             运行所有场景
    --interactive     交互模式（可输入任意请求）
    --verbose, -v     详细输出
"""

import argparse  # 命令行参数解析
import json  # JSON 数据处理
import sys  # 系统相关
import os  # 操作系统相关

# 将 src 目录添加到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# 导入 Office Agent 相关模块
from office_agent import OfficeAgent, WorkflowResult  # 主 Agent 和结果类型
from office_agent.scenarios import (
    WEEKLY_SALES_REPORT,  # 周报生成场景
    CUSTOMER_RESEARCH,  # 客户调研场景
    MEETING_PREPARATION,  # 会议准备场景
    list_scenarios,  # 列出所有场景
    get_scenario,  # 获取指定场景
    run_scenario,  # 运行场景
    demo_all,  # 运行所有演示
)


def print_header(text: str) -> None:
    """
    打印格式化的标题。
    
    Args:
        text: 要打印的标题文本
    """
    print("\n" + "=" * 70)
    print(f"  {text}")  # 居中打印文本
    print("=" * 70)


def print_result(result: WorkflowResult) -> None:
    """
    以格式化方式打印工作流执行结果。
    
    打印内容包括：
    - 执行状态和耗时
    - 任务执行摘要（每个任务的状态和输出）
    - 需要人工输入的任务
    - AI 生成的摘要
    
    Args:
        result: WorkflowResult 对象，包含执行结果
    """
    print("\n--- Workflow Result ---")
    print(f"Success: {result.success}")  # 是否成功
    print(f"Execution Time: {result.execution_time_seconds:.2f}s")  # 执行耗时
    print(f"Final State: {result.final_state.value}")  # 最终状态

    # 任务摘要
    print("\n--- Task Summary ---")
    for task in result.plan.tasks:
        # 状态图标映射
        status_icon = {
            "completed": "[OK]",  # 已完成
            "pending": "[..]",  # 等待中
            "running": "[>>]",  # 执行中
            "failed": "[!!]",  # 失败
            "waiting_human_input": "[??]",  # 等待人工输入
        }.get(task.status.value, "[--]")

        print(f"  {status_icon} {task.capability_required}: {task.description}")
        
        # 打印任务输出预览
        if task.result and task.result.output:
            output_preview = str(task.result.output)[:80]
            if len(str(task.result.output)) > 80:
                output_preview += "..."
            print(f"       -> {output_preview}")
        elif task.result and task.result.error:
            print(f"       -> ERROR: {task.result.error}")

    # 人工输入摘要
    if result.human_inputs:
        print(f"\n--- Human Inputs ({len(result.human_inputs)}) ---")
        for human_input in result.human_inputs:
            print(f"  Task: {human_input.task_id}")  # 任务 ID
            print(f"  Status: {human_input.status.value}")  # 状态
            if human_input.response:
                print(f"  Response: {human_input.response[:100]}...")  # 响应预览

    # AI 摘要
    if result.output.get("summary"):
        print(f"\n--- AI Summary ---")
        print(result.output["summary"])


def interactive_mode():
    """
    交互模式：允许用户输入任意请求。
    
    在交互模式下，用户可以：
    - 输入任何办公相关的请求
    - Agent 会自动分析、规划、执行
    - 输入 'exit' 或 'quit' 退出
    
    PEV 工作流：
    1. Planner 分析用户请求，生成任务计划
    2. Executor 并行执行任务
    3. Verify 验证结果
    4. 需要时触发 Human-in-the-Loop
    """
    print_header("Office Agent - Interactive Mode")
    print("\nWelcome! I am your AI Office Assistant.")
    print("I can help you with various office tasks using my PEV workflow.")
    print("\nExample requests:")
    print("  - Generate a weekly sales report")  # 生成周报
    print("  - Research our top customers")  # 调研客户
    print("  - Prepare a meeting pack for my quarterly review")  # 准备会议材料
    print("\nType 'exit' or 'quit' to end the session.\n")

    # 创建 Agent 实例
    agent = OfficeAgent()

    # 主循环
    while True:
        try:
            # 获取用户输入
            user_input = input("\nYour request: ").strip()

            # 检查退出命令
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            # 跳过空输入
            if not user_input:
                continue

            print()
            # 执行请求
            result = agent.execute(user_input)
            # 打印结果
            print_result(result)

        except KeyboardInterrupt:
            # Ctrl+C 中断
            print("\n\nSession ended.")
            break
        except Exception as e:
            # 打印错误
            print(f"\nError: {e}")


def run_specific_scenario(scenario_name: str) -> None:
    """
    运行指定的场景。
    
    根据场景名称从 scenarios 模块获取场景配置，
    然后执行该场景。
    
    Args:
        scenario_name: 场景名称（如 'weekly_sales_report'）
    """
    # 尝试获取场景
    scenario = get_scenario(scenario_name)

    if scenario:
        # 找到场景，打印信息并执行
        print_header(f"Running Scenario: {scenario.name}")
        print(f"\nDescription: {scenario.description}")
        print(f"\nUser Request: {scenario.user_request}")
        print()
        result = run_scenario(scenario, verbose=True)  # 运行场景
        print_result(result)  # 打印结果
    else:
        # 未找到场景，打印可用选项
        print(f"Unknown scenario: {scenario_name}")
        print("\nAvailable scenarios:")
        for s in list_scenarios():
            print(f"  - {s.name}")


def main():
    """
    主入口函数。
    
    解析命令行参数并执行相应的操作：
    - --list: 列出所有场景
    - --interactive: 交互模式
    - --scenario: 运行指定场景
    - --all: 运行所有场景
    - 默认: 运行所有场景
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Office Agent Demo - PEV-based office automation",  # 描述
        formatter_class=argparse.RawDescriptionHelpFormatter,  # 支持多行帮助
        epilog="""
Examples:
  %(prog)s --all              Run all scenarios
  %(prog)s --scenario=weekly_sales_report
  %(prog)s --interactive      Interactive mode
  %(prog)s --list            List available scenarios
        """
    )

    # 添加命令行参数
    parser.add_argument(
        "--scenario", "-s",  # 短选项 -s
        type=str,
        help="Run a specific scenario"  # 运行指定场景
    )

    parser.add_argument(
        "--all", "-a",  # 短选项 -a
        action="store_true",
        help="Run all scenarios"  # 运行所有场景
    )

    parser.add_argument(
        "--interactive", "-i",  # 短选项 -i
        action="store_true",
        help="Run in interactive mode"  # 交互模式
    )

    parser.add_argument(
        "--list", "-l",  # 短选项 -l
        action="store_true",
        help="List available scenarios"  # 列出场景
    )

    parser.add_argument(
        "--verbose", "-v",  # 短选项 -v
        action="store_true",
        help="Verbose output"  # 详细输出
    )

    # 解析参数
    args = parser.parse_args()

    # 处理 --list 选项
    if args.list:
        print_header("Available Scenarios")
        for scenario in list_scenarios():
            print(f"\n  {scenario.name}")
            print(f"  Description: {scenario.description}")
            print(f"  Expected Tasks: {', '.join(scenario.expected_tasks[:3])}...")
        return

    # 处理 --interactive 选项
    if args.interactive:
        interactive_mode()
        return

    # 处理 --all 选项
    if args.all:
        print_header("Running All Scenarios")
        results = demo_all()

        # 打印汇总表
        print("\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        print(f"  {'Scenario':<35} {'Status':<10}")
        print("-" * 70)
        for name, success in results:
            status = "PASSED" if success else "FAILED"
            print(f"  {name:<35} {status:<10}")
        print("=" * 70)

        # 统计通过数
        passed = sum(1 for _, s in results if s)
        print(f"\nTotal: {passed}/{len(results)} passed")
        return

    # 处理 --scenario 选项
    if args.scenario:
        run_specific_scenario(args.scenario)
        return

    # 默认：运行所有场景
    print("No options specified. Running all scenarios...")
    print("Use --help for more options.")
    print()
    args.all = True
    run_all()  # 注意：这个函数在当前代码中没有定义，应该是 demo_all()


if __name__ == "__main__":
    # 当直接运行此脚本时执行 main 函数
    main()
