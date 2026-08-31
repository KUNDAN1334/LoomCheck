# Loomcheck

A regression and drift harness for multi-step AI agent workflows.

An agent that crashes gets caught by observability. An agent that keeps working while getting
worse does not. Someone edits one line of a plain-English procedure, the agent stops escalating
the cases it should escalate and starts guessing instead — and the dashboards stay green, because
throughput went up and cost went down.

Loomcheck runs a fixed suite of scenarios against an agent with deterministically mocked tools,
grades each run on five independent dimensions, stores every trace, and diffs two runs so a
behaviour change can be attributed to the procedure edit that caused it.

Work in progress.
