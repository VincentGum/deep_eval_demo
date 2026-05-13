## 项目概述
Offline PEV Customer Support Agent - 基于 LangChain + LangGraph 的离线客服智能体 demo，采用 PEV（Plan-Execute-Verify）架构。

## 技术栈
- **语言**: Python 3.10+
- **框架**: LangChain, LangGraph
- **测试**: DeepEval (离线模式)
- **包管理**: uv + 项目级虚拟环境

## 目录结构
```
/workspace/projects/
├── src/
│   └── customer_agent/
│       ├── __init__.py
│       ├── policies.py      # 策略验证（敏感词检测、低置信度暂停）
│       └── tools.py         # 工具定义（查询订单、创建退款）
├── pyproject.toml           # 项目配置
├── requirements.txt         # 依赖声明
├── README.md
└── .coze                    # Coze 项目配置
```

## 关键入口 / 核心模块
- **入口**: `src/customer_agent/` (包)
- **依赖链**: `__init__.py` → `agent.py` (当前缺失)
- **运行命令**: `PYTHONPATH=src .venv/bin/python examples/run_demo.py`
- **评测命令**: `PYTHONPATH=src DEEPEVAL_TELEMETRY_OPT_OUT=YES .venv/bin/deepeval test run tests/evals/test_customer_agent.py`

## 项目状态
⚠️ **不完整**: 
- `src/customer_agent/agent.py` 缺失（被 `__init__.py` 引用但文件不存在）
- `examples/run_demo.py` 缺失
- `tests/` 目录缺失

## 运行与预览
- **项目类型**: backend（纯离线 demo，无 Web 界面）
- **预览**: 不可预览
- **部署**: 需要先补全缺失的入口文件

## 用户偏好与长期约束
- 使用 `uv` 管理 Python 依赖
- `PYTHONPATH` 需设为 `src` 才能正确导入模块

## 常见问题和预防
- 缺少入口文件时项目无法运行
- 运行时必须设置 `PYTHONPATH=src`
