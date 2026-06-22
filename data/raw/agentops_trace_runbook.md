# AgentOps Trace Runbook

AgentOps tracing records the state transition for every workflow node in an agent run.

The workflow contains an intent router, planner, retriever, evidence verifier, writer, and citation checker.

Each trace event stores the node name, status, timing, input summary, output summary, and any error message.

The trace makes it possible to replay failures, debug missing citations, and explain why the agent produced a specific answer.
