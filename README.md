# Offline PEV Customer Support Agent

This demo contains a LangChain + LangGraph customer support agent using a PEV architecture:

- Plan: classify customer intent, choose tools, and decide whether the case is sensitive.
- Execute: call local mock tools and a mocked LLM runnable to draft the reply.
- Verify: check policy and pause for human approval when needed.

Everything runs locally. The LLM is mocked, and the DeepEval suite uses deterministic local metrics instead of model-judged metrics.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run The Demo

```bash
PYTHONPATH=src .venv/bin/python examples/run_demo.py
```

## Run DeepEval Offline

```bash
PYTHONPATH=src DEEPEVAL_TELEMETRY_OPT_OUT=YES .venv/bin/deepeval test run tests/evals/test_customer_agent.py --identifier offline-pev-customer-agent
```

The eval data lives in `tests/evals/customer_support_goldens.json`.
