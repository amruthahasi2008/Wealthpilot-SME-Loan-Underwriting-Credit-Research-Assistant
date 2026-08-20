# WealthPilot — Architecture Review

## System overview

WealthPilot is an SME loan underwriting assistant for a fictional digital
lender (Ashva Capital). It takes a messy loan application, extracts it into
validated structured data, computes risk using real tools and real policy
lookups (not guesses), drafts and self-checks an underwriting memo through a
multi-agent team, and routes the final decision through a durable,
checkpointed workflow that knows when a human signatory is legally required.
Every module is independently testable and every milestone passes its own
checklist against real data — no step is simulated or hand-waved.

## Components & data flow

| Milestone | Component | Input | Output | Depends on |
|---|---|---|---|---|
| M1 | Structured intake parser | Raw applicant narrative + attached financial figures | Validated `LoanApplication`/`FinancialSnapshot` (Pydantic) | — |
| M2 | Tool-calling risk agent | A loan question | Cited recommendation, using 4 real tools (DSCR, EMI, bureau, bank statements) | M1's output as context |
| M3 | Persistent memory + policy index | Applicant history / policy questions | Episodic recall, semantically-indexed policy chunks | Gemini embeddings |
| M4 | Hybrid retrieval + grounded answers | A policy question | Cited answer, or an explicit "not addressed" (never fabricated) | M3's index |
| M5 | Checkpointed underwriting workflow | A `LoanApplication` | approved / declined, with a real pause-and-resume on loans needing a human signatory | M1, M2, M4 |
| M6 | Multi-agent memo drafting team | A `LoanApplication` | A fact-checked, reviewed underwriting memo, evidence via local calls or MCP | M1, M2, M4 |
| M7 | Observability + reliability | The M6 team | A per-agent latency dashboard; a hardened tool call (retry → fallback → circuit breaker) | M6 |
| M8 | Evaluation, guardrails, deployment | The full pipeline | 20-item scored golden set + bias-probe pairs + a live `/chat` API | M1–M6 |

## Evaluation results summary

- **M1**: 6/6 synthetic applications parsed, 6/6 exact match against the
  data generator's own ground truth (including one application with a
  deliberately missing field, handled without crashing).
- **M4 retrieval**: precision/recall/MRR measured across factual,
  multi-hop, and guardrail categories; the one unanswerable question
  (cryptocurrency collateral) correctly produced no fabricated citation.
- **M6**: 11/11 milestone checklist items pass, including proof that local
  and MCP-sourced evidence for the same query are byte-identical
  (`['POLICY-84', 'POLICY-85', 'TOOL-BANK']` both ways).
- **M7**: seeded fault injection is exactly reproducible; retry improves
  success rate from 57% to 89.5%; the fallback layer guarantees 100% of
  calls return usable data; the circuit breaker opens after exactly 3 real
  failures and short-circuits the remaining 7 of 10 calls.
- **M8 golden set**: 20/20 items pass across 4 independently-scored
  pipeline components (tool use, retrieval, planning, final answer).
- **M8 bias probe** *(added beyond the reference notebook — not present in
  the course template, built specifically because WealthPilot's own data
  generator produces matched bias pairs)*: 3/3 matched pairs — identical
  financials, only applicant name/city varying — reached both identical
  extracted numbers and identical final decisions. No divergence detected.

## Guardrails & safety

- **Schema validation**: every memo response is validated against a
  Pydantic `MemoResponse` model; a placeholder rationale (`"n/a"`, `"..."`)
  is rejected, not silently accepted.
- **Prompt-injection scanning**: a regex scanner runs over *both* user
  input and every retrieved policy chunk — retrieved content is itself an
  injection vector, not just user queries.
- **Structural fabrication check**: M4/M6's citation system means every
  policy claim traces to a real retrieved `[POLICY-#]`/`[TOOL-#]` id;
  `fact_checker_node` (M6) deterministically strips any citation that
  doesn't.
- **Fair-lending check**: the M8 bias-probe suite is the guardrail against
  the specific failure mode this domain cares about most — a system that
  reaches a different decision for two applicants with identical financials
  and only a differing name/city.
- **Human-in-the-loop enforcement**: M5's workflow structurally cannot
  auto-approve a loan over ₹5,000,000 — `requires_human_signoff` forces a
  real pause via `interrupt()`, checkpointed to SQLite so it survives a
  process restart.

## Known limitations

- The golden set (M8) is fully deterministic by design, given time
  constraints — it does not yet cross-validate with an LLM-judge framework
  (Ragas/DeepEval/TruLens, as the reference notebook demonstrates). A
  production version should add at least one judge-based metric alongside
  the deterministic scorers.
- The bias-probe suite covers 3 matched pairs across 3 sectors — a
  production fairness audit would run this across all 8 sectors and a wider
  set of demographic proxies (not just name/city).
- Policy corpus is LLM-generated (per the course's data generator), not
  sourced from a real lender's actual policy manual — some expected clauses
  (e.g. an explicit loan-to-EBITDA multiple) are absent from the generated
  text even where the generation prompt requested them; this is disclosed
  in the M4/M8 test suites, not hidden.
- Langfuse cloud tracing (M7) runs in degraded/local-only mode without
  configured API keys — the dashboard still works, but traces aren't
  visible in a shared dashboard until keys are added.
- M6/M7's multi-agent team currently only exercises one applicant per run
  in the test suite; batch/concurrent processing is not yet load-tested.

## Deployment notes

- M8 ships a FastAPI service (`/health`, `/chat`) runnable via
  `uvicorn.Server` in a background thread — verified with real HTTP
  requests against `127.0.0.1:8001` in `M8.py`'s own test run.
- The MCP server (`wealthpilot_mcp_server.py`, M6) runs as a standalone
  stdio subprocess and can be consumed by any MCP-compatible client, not
  just this project's own agent code.
- M5's SQLite checkpointer (`workflow_checkpoints.sqlite`) is file-backed
  and durable; swapping to Postgres for multi-instance deployment is a
  one-line change to the checkpointer, per LangGraph's own guidance.
- No real personal or applicant data is used anywhere — all data is
  synthetic, generated by the program's `capstone-data-toolkit`.
