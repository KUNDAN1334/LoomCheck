# Loomcheck

A regression and drift harness for multi-step AI agent workflows.

Observability catches an agent that crashes. Nothing catches an agent that keeps working and
quietly gets worse — one that stops escalating the cases it should escalate and starts guessing
instead. When that happens the dashboards improve: throughput rises, cost per case falls, error
rate is flat. Every number a team watches moves the right way.

Loomcheck is the alarm that goes off when that happens.

You define scenarios as files, run an agent against deterministically mocked tools, get graded on
five dimensions, and diff two runs to see what regressed.

---

## The grader that matters

Five graders run on every scenario: **outcome**, **trajectory**, **escalation**, **recovery**,
**budget**.

**Escalation is the important one.** The other four measure whether the agent is good at the task.
Escalation measures whether the agent knows what it does not know — and it is the dimension that
degrades invisibly, because an agent that stops blocking produces better-looking numbers on every
other one. Blocked stages are the slow, unresolved ones; stop blocking and the dashboard improves
while the work gets worse.

It is expensive in both directions, and not symmetrically. Under-blocking produces confident wrong
decisions nobody sees. Over-blocking costs a person's attention *and* a second agent invocation,
because unblocking re-invokes the agent to reassess — one case, two billable interactions.

Three things make the grader honest:

- A scenario declares `ground_truth.precedents: []` when the case genuinely has no precedent. Any
  confident resolution there is a wrong answer, however well-reasoned it looks.
- Routine cases list `blocked` in `must_not_call`. Without those, an agent that blocks everything
  scores perfectly and has automated nothing.
- One case (`claims-wd-026`) has only *weak* precedent — 0.58 and 0.44 similarity, neither on all
  fours — where the schedule answers the question anyway. That is where a cautious agent blocks a
  claim it could have settled, and it is the failure mode closest to how real agents behave.

And the sharpest edge: on a no-precedent case, `request_more_info` is a **failure**, not an
abstention. The missing knowledge is the company's, not the claimant's, so asking the customer
leaves the stage waiting on a reply that cannot resolve it, instead of blocked on the person who
could.

---

## Status

**Complete.** Sixteen scenarios, five graders, four injected failure modes, and a regression
diff — built in six phases, each documented as it landed.

| Phase | Delivers | State |
|---|---|---|
| 1 | Models, YAML loader, schema, three scenarios | **done** |
| 2 | LangGraph agent, mock resolver, trace recording | **done** |
| 3 | Five graders and the terminal report | **done** |
| 4 | Failure injection, exercising the recovery grader | **done** |
| 5 | The regression diff | **done** |
| 6 | Full sixteen-scenario suite | **done** |

Every figure below comes from two real runs against `openai/gpt-oss-20b` on 2026-08-26 —
`2bbb9298` on `claims_intake_v1` and `96adbb24` on `claims_intake_v2`. A full run is 16 scenarios,
about 12 minutes and $0.008.

---

## Quick start

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and Docker.

```bash
cp .env.example .env          # set GROQ_API_KEY; DATABASE_URL matches compose as shipped
docker compose up -d db
uv sync --group dev
uv run alembic upgrade head
uv run loomcheck run scenarios/claims/
```

```
run 2bbb9298  ·  claims_intake_v1  ·  openai/gpt-oss-20b
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━┓
┃ scenario      ┃ resolution        ┃ outcome ┃ trajectory ┃ escalation ┃ recovery ┃ budget ┃ turns ┃    cost ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━┩
│ claims-fr-002 │ deny_claim        │    ✓    │     ✓      │     ✓      │    –     │   ✓    │     2 │ $0.0003 │
│ claims-fr-005 │ approve_claim     │    ✓    │     ✓      │     ✓      │    –     │   ✓    │     4 │ $0.0006 │
│ claims-fr-009 │ blocked           │    ✓    │     ✓      │     ✓      │    ✓     │   ✓    │     4 │ $0.0005 │
│ claims-lb-002 │ blocked           │    ✓    │     ✓      │     ✓      │    –     │   ✓    │     3 │ $0.0004 │
│ claims-st-003 │ approve_claim     │    ✓    │     ✓      │     ✓      │    –     │   ✓    │     3 │ $0.0004 │
│ claims-st-006 │ request_more_info │    ✓    │     ✓      │     ✓      │    –     │   ✓    │     3 │ $0.0005 │
│ claims-st-012 │ deny_claim        │    ✓    │     ✓      │     ✓      │    –     │   ✓    │     3 │ $0.0005 │
│ claims-th-004 │ deny_claim        │    ✓    │     ✓      │     ✓      │    –     │   ✓    │     2 │ $0.0003 │
│ claims-wd-001 │ approve_claim     │    ✓    │     ✓      │     ✓      │    –     │   ✓    │     4 │ $0.0007 │
│ claims-wd-004 │ blocked           │    ✓    │     ✓      │     ✓      │    –     │   ✓    │     3 │ $0.0004 │
│ claims-wd-007 │ deny_claim        │    ✓    │     ✓      │     ✓      │    –     │   ✓    │     4 │ $0.0007 │
│ claims-wd-011 │ request_more_info │    ✓    │     ✓      │     ✓      │    –     │   ✓    │     2 │ $0.0003 │
│ claims-wd-015 │ blocked           │    ✗    │     ✗      │     ✗      │    ✓     │   ✓    │     3 │ $0.0005 │
│ claims-wd-019 │ approve_claim     │    ✓    │     ✓      │     ✓      │    –     │   ✓    │     4 │ $0.0006 │
│ claims-wd-022 │ blocked           │    ✓    │     ✓      │     ✓      │    –     │   ✓    │     3 │ $0.0005 │
│ claims-wd-026 │ approve_claim     │    ✓    │     ✓      │     ✓      │    –     │   ✓    │     4 │ $0.0007 │
└───────────────┴───────────────────┴─────────┴────────────┴────────────┴──────────┴────────┴───────┴─────────┘
outcome 94%  trajectory 94%  escalation 94%  recovery 100%  budget 100%   ·   total $0.0077

failures
  claims-wd-015  outcome
    expected approve_claim, agent called blocked at turn 3
  claims-wd-015  trajectory
    called blocked at turn 3, which this scenario forbids
  claims-wd-015  escalation
    precedent existed (PR-2512, PR-2488), so this case was answerable; agent blocked the stage
    at turn 3 — over-blocking costs a handler's time plus a second agent invocation to
    reassess, and it hides as caution
```

One row is worth stopping on, and it is the one the suite was built around.

`claims-wd-015` is a routine escape-of-water claim with two precedents on file. `precedent_search`
returns `{}` — not because the case is novel, but because the search broke. The agent read "no
results" as "no precedent" and blocked. Three graders fail it.

The fourth is the interesting one. `recovery` **passes**: the agent did not retry, but it also did
not act on data it never got — it stopped. That is graceful degradation, and it is genuinely
correct behaviour on that dimension. So one trace scores green on how it handled the failure and
red on the conclusion it drew from it. A single averaged score cannot say that. Five graders that
are allowed to disagree can, and that disagreement is the whole reason there are five.

`recovery` reads `–` on the scenarios with no injected failure, not a tick. A grader with nothing
to check stays out of the pass rate entirely; a rate inflated by graders that had nothing to grade
is the same dishonest green this tool exists to catch.

`loomcheck show <run>` prints every turn of a run — tool, arguments, result, latency, tokens —
followed by all five verdicts, because a grade is only worth trusting if the steps behind it can
be read back. Any unique id prefix works.

Scenarios are validated before anything is spent — a missing fixture, an unknown outcome or a
typo'd field name fails at load, naming the file and the line:

```
error scenarios/claims/claims-wd-004.yaml:24: 'expect.outcome' must be one of [approve_claim,
deny_claim, request_more_info, blocked], got 'escalate'
```

---

## The point of all of it

Someone edits the procedure. `procedures/claims_intake_v2.md` is `v1` with the "when to block"
section rewritten — the kind of edit a domain expert makes on a Tuesday to bring the number of
blocked cases down:

> Use your judgement. Not every case needs a precedent: the policy schedule is the authority, and
> many claims with no close precedent are still straightforward once you read the terms.
>
> If you are only missing a detail, ask the claimant for it rather than blocking — it is faster
> for them and it keeps the case with you.

Read on its own, that looks like an improvement. Run both and compare:

```bash
uv run loomcheck run scenarios/claims/
uv run loomcheck run scenarios/claims/ --procedure claims_intake_v2
uv run loomcheck diff 2bbb9298 96adbb24
```

```
claims_intake_v1 → claims_intake_v2
model openai/gpt-oss-20b (unchanged)
run 2bbb9298 → 96adbb24  ·  16 scenario(s) compared

outcome            94%  →       69%    -25%  ⚠ REGRESSION
trajectory         94%  →       75%    -19%  ⚠ REGRESSION
escalation         94%  →       75%    -19%  ⚠ REGRESSION
recovery          100%  →      100%
budget            100%  →      100%
cost / case    $0.0005  →   $0.0004     -7%

5 scenario(s) changed verdict
  claims-fr-009   blocked → deny_claim
    ✗ escalation
      no precedent existed for this case; agent called deny_claim at turn 4 anyway
  claims-lb-002   blocked → request_more_info
    ✗ escalation
      no precedent existed for this case; agent called request_more_info at turn 2 instead of
      blocking — the gap is the company's, not the claimant's, so asking them cannot close it
  claims-wd-004   blocked → request_more_info
    ✗ escalation
      no precedent existed for this case; agent called request_more_info at turn 3 instead of
      blocking — the gap is the company's, not the claimant's, so asking them cannot close it
  claims-wd-022   blocked → request_more_info
    ✗ escalation
      no precedent existed for this case; agent called request_more_info at turn 3 instead of
      blocking — the gap is the company's, not the claimant's, so asking them cannot close it
  claims-wd-015   blocked → request_more_info
    ✓ trajectory
      no forbidden or repeated calls (policy_lookup → precedent_search → request_more_info)
    ✓ escalation
      precedent existed (PR-2512, PR-2488); agent resolved it without blocking at turn 3

Cost per case improved. Outcome, trajectory and escalation did not.
A dashboard would have shown green.
```

Read the middle column, not the top. **Escalations went from five to zero.** Every case the agent
used to hand to a person, it now answers by itself — and four of those five were the cases where it
had no basis to answer at all. Three of them it sent back to the claimant, which is the specific
dodge the deleted paragraph existed to prevent:

> Asking the claimant for more information does not resolve a missing precedent. The gap is in our
> knowledge, not theirs, and they cannot fill it. Block instead.

One paragraph out of one file, and three cases that used to reach a senior handler now sit waiting
on a claimant who cannot help. Nothing crashed. Cost fell 7%. Every case still resolved.

The last entry is the honest complication. `claims-wd-015` was wrong before and is still wrong —
but it moved from a *forbidden* wrong answer to an *allowed* one, so `trajectory` and `escalation`
report green ticks on a scenario whose outcome never passed. The diff prints them anyway, under the
scenario's name, rather than netting them off into the summary. Two graders improving on a case
that is still broken is exactly the kind of fact an aggregate is built to lose.

Cost per case is shown and never flagged either way. Spend falling is exactly what happens when
an agent stops doing work — marking it as an improvement would build the illusion into the tool
designed to break it. Putting it next to the pass rates and letting you see both is the whole
argument.

Two other things the diff refuses to fudge: rates are computed over the scenarios the two runs
*share*, so a delta can never quietly mean "you ran a different suite"; and if the model changed
as well as the procedure, it says the comparison cannot attribute anything to either rather than
reporting a finding.

---

## Shaped for the platform under test

Three details are borrowed rather than invented, so that pointing this at a real agent is a
configuration job rather than a porting job.

**`blocked` is the outcome, not "escalate".** When an agent finishes a stage it returns an
outcome, and `blocked` is one of the values that outcome can take: the stage halts and waits for a
person, who unblocks it by posting a message or resuming it manually, at which point the agent is
invoked again to reassess. The enum names the platform state, not the intent behind it.

The four resolutions map onto what the stage does next:

| resolution | the stage |
|---|---|
| `approve_claim`, `deny_claim` | completes |
| `request_more_info` | waits on external input — the claimant's reply |
| `blocked` | halts until a person intervenes |

That mapping is why `request_more_info` and `blocked` are separate values rather than one
"abstain". They are different stage states with different costs, and an agent that reaches for the
cheap one to avoid the expensive one is doing something worth catching.

**The tool split mirrors the read-only baseline.** A new agent starts able to read — notes, files,
tasks, threads, objects by id, search — and every tool with an effect is granted one at a time.
So `policy_lookup`, `precedent_search` and `claim_history` only read, and the four terminal tools
are the ones that would need granting.

**Tool definitions follow the Custom Tool shape** — a name, a description saying *when* to use it,
and a typed JSON Schema input, generated from Pydantic models in `agent/tools.py`. Testing a real
tool here should be a matter of writing a scenario, not writing an adapter.

---

## The suite

Sixteen scenarios across five perils. The balance is deliberate and `tests/test_suite.py`
enforces it: answerable cases outnumber blocked ones three to one, because the escalation grader
is the one worth gaming and precision has to be tested harder than recall.

| expected | count | scenarios |
|---|---|---|
| `approve_claim` | 6 | routine hose failure, listed-building heritage rider, toaster fire, storm roof, small valve claim, **failed rooflight seal (weak precedent)** |
| `deny_claim` | 4 | late notification, lapsed policy, pre-existing disrepair, theft without forcible entry |
| `blocked` | 4 | mixed-use apportionment, undetermined cause, converted outbuilding, block-vs-unit liability |
| `request_more_info` | 2 | unknown date of discovery, no quotes obtained |

All four injected failure modes appear, including one permanent outage — a tool that never comes
back, where the correct behaviour is to stop rather than retry into the turn ceiling.

---

## A scenario

```yaml
id: claims-wd-004
description: Appliance leak in a mixed-use unit, no matching precedent exists
procedure: claims_intake_v1
case:
  title: "Water damage claim, Northfields Unit 9"
  inbound: fixtures/emails/wd_004.txt
ground_truth:
  precedents: []                    # deliberately empty — the agent should escalate
tools:
  policy_lookup:
    response: fixtures/tools/policy_8842.json
  precedent_search:
    response: fixtures/tools/empty_precedents.json
  claim_history:
    failure: server_error           # injected; resolves on retry
    then: fixtures/tools/history_8842.json  # omit `then` for a permanent outage
expect:
  outcome: blocked
  must_not_call: [approve_claim, deny_claim, request_more_info]
  max_tool_calls: 6
  max_cost_usd: 0.35
```

`ground_truth` is grader input, not agent input — the agent learns the same fact from the
`precedent_search` mock. `procedures/` holds the versioned instruction text; it is the system under
test, and editing one line of it is what the diff is built to catch.

The four failure modes are chosen for how hard they are to *notice*, not for what broke:
`server_error` and `timeout` arrive flagged on the tool channel; `malformed_json` arrives marked
successful and fails to parse; `empty_result` arrives marked successful, parses cleanly, and
contains nothing. The last one is deliberately indistinguishable from a legitimate no-results
answer — which is the escalation question arriving from the other direction.

Costs are recorded in USD, the billing currency. Converting to another currency would mean carrying
an exchange rate, and a slowly-drifting constant inside a project about detecting drift is a bad
idea.

---

## Commands

```
loomcheck run <path>              # run scenarios (file or directory), save, report
loomcheck list                    # past runs, newest first
loomcheck show <run_id>           # one run in detail, turn by turn
loomcheck diff <run_a> <run_b>    # regression comparison
```

Four commands, no more. `run` takes one option, `--procedure`, which overrides the procedure the
scenarios name — that is how a version gets tested without editing every file and editing it back.

---

## Development

```bash
uv run pytest
uv run mypy
uv run ruff check .
uv run alembic check       # migrations still match the ORM models
```

No test calls an LLM. The agent loop, the resolver, the recorder and persistence are all
exercised against a scripted chat model in `tests/stubs.py`, and every grader is tested against
hand-built traces. The database tests skip with a visible reason when
`DATABASE_URL` is unset — pytest runs with `-ra` so a skip never passes as a green suite.
