"""Two execution paradigms, managed separately, sharing one contract.

local.py   LangGraph planner/executor/synthesizer/evaluator subgraph.
claude.py  Agentic Claude session with MCP tools.

Both consume AgentRequest and emit AgentResponse.
"""
