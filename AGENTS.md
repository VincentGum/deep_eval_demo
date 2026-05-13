## 项目概述
Offline PEV Customer Support Agent - 基于 LangChain + LangGraph 的离线客服智能体 demo，采用 PEV（Plan-Execute-Verify）架构。

## 技术栈
- **语言**: Python 3.10+
- **框架**: LangChain, LangGraph
- **测试**: DeepEval (离线模式)
- **包管理**: uv + 项目级虚拟环境（系统环境用 pip）

## 目录结构
```
/workspace/projects/
├── src/
│   └── customer_agent/
│       ├── __init__.py          # 导出接口
│       ├── agent.py             # LangGraph PEV 状态图
│       ├── mock_llm.py          # Mock LLM 实现
│       ├── policies.py          # 策略验证
│       └── tools.py             # 工具定义
├── examples/
│   └── run_demo.py              # 演示脚本
├── tests/
│   ├── conftest.py              # Pytest 配置
│   └── evals/
│       ├── test_customer_agent.py       # 评测用例
│       └── customer_support_goldens.json # 测试数据
├── pyproject.toml               # 项目配置
├── requirements.txt             # 依赖声明
├── README.md
└── .coze                        # Coze 项目配置
```

## 关键入口 / 核心模块

### PEV 架构节点
1. **PLAN**: 分析用户意图，决定调用哪些工具
2. **EXECUTE**: 调用 tools.py 中的工具
3. **VERIFY**: 调用 policies.verify_policy 检查敏感词和置信度
4. **HUMAN_REVIEW**: 敏感操作需人工审批（可配置审批函数）

### 核心函数
- `invoke_customer_agent(user_message, human_approval_func=None)` - 入口函数
- `build_customer_support_graph()` - 构建 LangGraph 状态图

## 运行与预览

### 运行 Demo
```bash
PYTHONPATH=src python3 examples/run_demo.py
```

### 运行测试
```bash
PYTHONPATH=src DEEPEVAL_TELEMETRY_OPT_OUT=YES python3 -m pytest tests/evals/test_customer_agent.py -v
```

## 项目状态
- **完成度**: 100%
- **测试通过**: 7/7
- **Golden Dataset**: 10/10 通过，平均得分 99.17%

## 用户偏好与长期约束
- 使用 `uv` 管理 Python 依赖（系统环境用 pip）
- `PYTHONPATH` 需设为 `src` 才能正确导入模块
- DeepEval 需设置 `DEEPEVAL_TELEMETRY_OPT_OUT=YES` 离线运行

## 常见问题和预防
- 缺少入口文件时项目无法运行
- 运行时必须设置 `PYTHONPATH=src`
