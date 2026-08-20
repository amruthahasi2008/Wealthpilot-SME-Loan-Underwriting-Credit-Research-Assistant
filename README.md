# WealthPilot — SME Loan Underwriting & Credit Research Assistant

Capstone project for the Applied AI Professional Certification Program.
Milestones M1–M5, built on synthetic data from the program's
`capstone-data-toolkit` (domain: `wealthpilot`) — 13 credit policy documents,
6 loan applications, and 4 mock API tables (`applicants`, `bureau_reports`,
`bank_statements`, `past_decisions`).

Run each milestone independently: `python M1.py`, `python M2.py`, etc.
(activate `venv` first, or call `./venv/bin/python M1.py` directly).

---

## M1 — Provider-agnostic client + structured intake (`M1.py`)

Parses a messy loan application (applicant narrative + attached financial
figures) into a validated `LoanApplication` / `FinancialSnapshot` Pydantic
object, via LiteLLM (provider-agnostic — currently Groq, falls back to
OpenRouter) with a **repair loop**: invalid JSON or a schema mismatch is fed
back to the model with the exact validation error, and retried.

**Result:** 6/6 applications parsed successfully, 6/6 exact match against
the generator's `declared_financials` ground truth (including one
application with a deliberately missing field, handled without crashing).

## M2 — Tool-enabled single agent (`M2.py`)

A credit-risk-analyst agent with 4 tools it must call rather than guess:
`dscr_calculator`, `emi_calculator`, `mock_credit_bureau_api`,
`mock_bank_statement_api` (the last two backed by the generator's CSV
tables). Standard function-calling loop: model requests a tool, we execute
it, feed the result back, repeat until it answers in plain text.

**Result:** given a sample applicant and loan terms, the agent correctly
called all 4 tools in a sensible order (bureau → bank statements → EMI →
DSCR) and produced a cited recommendation with real computed figures.

## M3 — Persistent memory + semantic policy index (`M3.py`)

Two memory systems, both backed by Gemini embeddings with a disk cache:
- **Applicant memory**: episodic entries seeded from `past_decisions.csv`,
  recalled by semantic similarity, filterable per applicant.
- **Policy index**: the 13 policy documents chunked and embedded for
  semantic search.

**Result:** episodic recall correctly retrieved the exact past decision for
a sampled applicant; policy search correctly surfaced the **textiles-sector
DSCR floor (1.40x)** as the top hit for a textiles-specific query, distinct
from the default 1.25x floor.

## M4 — Production RAG + retrieval evaluation (`M4.py`)

Upgrades M3's plain dense search: **BM25 + dense fused with Reciprocal Rank
Fusion**, a **cross-encoder rerank** pass, and **grounded, cited answer
generation** with a structural prompt-injection defense (policy text is
treated as untrusted data, not instructions). A golden eval set scores
precision@k / recall@k / MRR across `naive-dense` vs `hybrid+rerank`.

**Result:** retrieval metrics computed across factual/multi-hop/guardrail/
unanswerable categories. Critically, on the unanswerable question ("what's
our policy on crypto collateral?"), the system correctly stated the manual
doesn't address it **instead of fabricating a citation** — the compliance
failure mode this milestone specifically guards against.

## M5 — Orchestrated workflow with a durable human-approval checkpoint (`M5.py`)

Rebuilt to match the course's own reference skeleton: a LangGraph state
machine with an explicit control/audit field split (`missing_fields` etc.
overwrite; `audit_log` accumulates), `intake → verify_documents → backfill
(loop, capped) → risk_scoring` (DSCR against the sector-specific floor
pulled live from M4's index) `→ conditional routing`. Loans under the
₹5,000,000 signatory threshold with a clear DSCR outcome are auto-decided;
everything else **pauses the graph** via `interrupt()`, persisted to a real
**SQLite file** (not just in-memory), and waits for a human.

**Result:** a small loan auto-declined with no human involved. A
₹6,000,000 loan correctly paused. The run then **rebuilds the graph from a
completely fresh SQLite connection** — simulating a real process restart —
and proves the paused state survives it, before resuming with a simulated
human decision. 8/8 milestone checklist items pass.

## M6 — Multi-agent memo-drafting team + a custom MCP server (`M6.py`, `wealthpilot_mcp_server.py`)

A supervisor-routed team (Planner, Researcher, Writer, independent
Fact-Checker, Reviewer, Escalate) drafts and self-checks an underwriting
memo. Each agent has an **enforced write scope** (a `PermissionError` if a
node writes outside its role); the Supervisor's LLM-proposed routing is
only ever honored if it matches a deterministically-computed legal move.
The Fact-Checker independently verifies every `[POLICY-#]`/`[TOOL-#]`
citation in the draft traces to real retrieved evidence. A standalone
**FastMCP server** exposes WealthPilot's real tools (bureau, bank
statements, DSCR/EMI, policy search) sandboxed to its own workspace; the
team's Researcher is swapped to pull evidence over MCP instead of local
calls, consumed two ways (direct tool list + a stateful session).

**Result:** a fabricated citation was caught and flagged. The sandbox
rejected all path-escape attempts. Local and MCP-sourced evidence for the
same query were **byte-identical**. 11/11 milestone checklist items pass.

## M7 — Observability + reliability hardening (`M7.py`)

Wraps the M6 team with span-timing decorators (Langfuse-backed if keys are
configured, degrading gracefully to a local dashboard otherwise) and builds
a real per-agent cost/latency table. Separately, a **seeded** fault-injection
harness wraps a real tool call with reproducible failure rates, hardened in
three independently-measured steps: `tenacity` retry+backoff, a fallback
path, and a 3-state circuit breaker.

**Result:** tracing added zero behavior change (traced/untraced outputs are
identical). The dashboard showed the Supervisor's routing calls dominate
cost (~9.6s across 8 calls) while deterministic nodes are near-free. The
fault harness: baseline 57% → +retry 89.5% → +fallback 100% (never raises)
→ circuit breaker opens after exactly 3 real failures and short-circuits
the remaining 7 of 10 calls. Same seed reproduces the exact same numbers
every run. 8/8 milestone checklist items pass.

## M8 — Evaluation, guardrails, bias testing, deployment (`M8.py`)

A 20-item golden dataset scores 4 pipeline components independently (tool
use, retrieval, planning, final answer) — all deterministic, by design,
given time constraints (see `ARCHITECTURE_REVIEW.md` for what a production
version would add). **Added beyond the reference notebook, which has no
equivalent**: a bias-probe suite of 3 matched applicant pairs — identical
financials, only name/city differing — run through the real M1 parser and
decision logic. Guardrails (Pydantic schema + regex prompt-injection scan,
covering both user input *and* retrieved policy text) gate every response.
The pipeline ships as a real FastAPI service (`/health`, `/chat`).

**Result:** 20/20 golden items pass. **All 3 bias-probe pairs reached
identical decisions** — no fair-lending divergence detected. All 4
guardrail test cases behaved correctly. The FastAPI service answered real
HTTP requests end to end. 7/7 milestone checklist items pass.

---

## Architecture

```
data/wealthpilot/                  <- generated by capstone-data-toolkit
  corpus/markdown/*.md             <- 13 policy docs -> M3/M4 index
  intake/records.jsonl             <- 6 applications -> M1 input
  mock_api/*.csv                   <- 4 tables -> M2 tool backends

M1.py  parse_loan_application()  ──┐
M2.py  run_agent(), 4 tools       ─┼─► M5.py: checkpointed workflow (intake→docs→risk→route)
M4.py  retrieve(), answer()       ─┤   M6.py: multi-agent memo team, local or MCP evidence
M3.py  remember()/recall(), index ─┘   M7.py: traces + hardens the M6 team
                                        M8.py: scores, guards, and deploys the whole pipeline

wealthpilot_mcp_server.py  <- standalone MCP server (M6), exposes M2/M4's tools over stdio
workflow_checkpoints.sqlite <- M5's durable checkpoint store
```

## Reference materials

`M5.py` through `M8.py` were built to match the Applied AI program's own
reference notebooks for those milestones (generic templates covering a
document-approval workflow, a research-team multi-agent pattern, and a
campus-library toy pipeline respectively), re-derived onto WealthPilot's
actual loan-underwriting domain and data. See `ARCHITECTURE_REVIEW.md` for
the full write-up (system overview, data flow, evaluation results,
guardrails, known limitations, deployment notes) and `DEMO_SCRIPT.md` for a
presenter's script covering all 8 milestones.

## Status

M1 through M8 — the full 8-milestone rubric — are built and verified.
Every milestone's own checklist (mirroring the reference notebooks' grading
rubrics where one exists) is passing against real runs, not simulated
output. There is no M9; M8 is the final milestone in this program.
