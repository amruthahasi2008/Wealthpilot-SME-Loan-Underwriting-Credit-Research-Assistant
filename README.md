# WealthPilot — SME Loan Underwriting & Credit Research Assistant

An AI assistant that takes a small-business loan application written in plain
text and turns it into a policy-grounded underwriting recommendation. Every
number is computed by code, every policy claim cites a real document, and
large or borderline loans stop and wait for a human.

Capstone project for the Applied AI Professional Certification Program
(IIT Hyderabad). Built on synthetic data only: 13 credit-policy documents,
6 loan applications, and 4 mock API tables for a fictional lender, Ashva
Capital.

## How it works

```
loan application (free text)
      │
      ▼
M1  structured intake ──► validated LoanApplication (Pydantic)
      │
      ▼
M2  tool-calling agent ──► DSCR, EMI, credit bureau, bank statements
      │
      ▼
M3  memory + policy index ─► past decisions (Supermemory), 89 policy chunks
M4  hybrid retrieval ─────► cited answers from the policy corpus (Qdrant)
      │
      ▼
M5  LangGraph workflow ───► auto-decide, or pause for a human signatory
M6  multi-agent team ─────► fact-checked underwriting memo, tools over MCP
      │
      ▼
M7  tracing + fault tolerance        M8  evaluation, guardrails, API
```

## Tech stack

| Area | Tools |
|---|---|
| LLM access | LiteLLM (Groq `gpt-oss-120b`, OpenRouter fallback), Gemini for embeddings and judging |
| Structured output | Pydantic v2, JSON mode, validation-error repair loop |
| Agents and tools | LangChain (`tool()`, `ChatLiteLLM.bind_tools`), LangGraph |
| Memory | Supermemory (applicant history) |
| Retrieval | Qdrant Cloud, `gemini-embedding-001`, BM25 (`rank_bm25`), Reciprocal Rank Fusion, cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) |
| Workflow | LangGraph `StateGraph`, `interrupt()`, `SqliteSaver` checkpoints |
| Tool protocol | Model Context Protocol (FastMCP server, `langchain-mcp-adapters` client) |
| Reliability | `tenacity` retries, fallback path, circuit breaker, seeded fault injection |
| Observability and evaluation | Langfuse, Ragas, DeepEval, TruLens |
| Serving | FastAPI, uvicorn, ngrok |

## Milestones

### M1 — Structured intake (`M1.py`)

Parses a messy application (applicant narrative plus attached form figures)
into a validated `LoanApplication` object. If the model returns invalid JSON
or the wrong shape, the exact validation error is sent back to the model and
it retries (up to 2 repairs). Rate-limit errors back off and retry.

**Result:** 6/6 applications parsed, 6/6 exact match against the data
generator's ground truth, including one application with a deliberately
missing field.

### M2 — Tool-calling agent (`M2.py`)

A credit-analyst agent with four tools it must call instead of guessing:
`dscr_calculator`, `emi_calculator`, `mock_credit_bureau_api`,
`mock_bank_statement_api`. The tools are plain Python functions wrapped with
LangChain's `tool()` and bound to a `ChatLiteLLM` model. The loop runs each
requested tool, returns the result as a `ToolMessage`, and repeats until the
model answers in plain text.

**Result:** for a sample applicant the agent called all four tools (bureau,
bank statements, EMI, DSCR) and produced a recommendation using the computed
figures.

### M3 — Memory and policy index (`M3.py`)

- **Applicant memory:** past loan decisions stored in Supermemory and
  recalled by meaning, filterable per applicant.
- **Policy index:** the 13 policy documents split into 89 chunks and embedded
  with Gemini embeddings (cached on disk).

**Result:** recall returned the correct past decision for a sampled
applicant; a textiles question surfaced the textiles-specific DSCR floor
(1.40x) rather than the default 1.25x.

### M4 — Hybrid retrieval with cited answers (`M4.py`)

Dense search on a Qdrant Cloud collection and BM25 keyword search are fused
with Reciprocal Rank Fusion, then a cross-encoder reranks the top candidates.
Answers must cite the chunks they use. Retrieved policy text is treated as
untrusted data, never as instructions.

A small golden set (5 scored questions plus 1 unanswerable) compares plain
dense search with hybrid + rerank, and both runs are logged to Langfuse.

| Variant | precision@5 | recall@5 | MRR |
|---|---|---|---|
| Dense only | 0.56 | 0.67 | 1.00 |
| Hybrid + rerank | 0.60 | 0.71 | 0.87 |

**Result:** hybrid + rerank was slightly better on precision and recall and
slightly worse on MRR (one question ranked its best chunk third). Five
questions is too few to call a winner. On the unanswerable question ("what is
our policy on cryptocurrency collateral?") the system said the manual does
not cover it instead of inventing a citation.

### M5 — Workflow with a human-approval checkpoint (`M5.py`)

A LangGraph state machine: `intake → verify_documents → backfill (capped
loop) → risk_scoring → auto_decision or human_gate → finalize_human`. Risk
scoring compares DSCR with the sector-specific floor retrieved from the
policy index. Loans above INR 5,000,000, borderline cases, and applications
with unresolved documents pause at `interrupt()` and wait for a human. State
is checkpointed to a SQLite file.

**Result:** a small loan was auto-decided with no human involved. The largest
loan paused; the graph was then rebuilt on a fresh SQLite connection
(simulating a restart), the paused state was still there, and the run resumed
with a human decision. 8/8 checklist items pass.

### M6 — Multi-agent team and MCP server (`M6.py`, `wealthpilot_mcp_server.py`)

A supervisor routes work between five specialists (Planner, Researcher,
Writer, Fact-Checker, Reviewer) with an escalation path to a human.

- Each agent may only write its own part of the shared state; anything else
  raises a `PermissionError`.
- The supervisor's LLM-proposed route is only used if it matches the route a
  plain Python function computed.
- The Fact-Checker is deterministic: every `[POLICY-#]` / `[TOOL-#]` citation
  in the memo must match evidence that was actually retrieved.

A standalone FastMCP server exposes the underwriting tools and a
citation-rules resource over stdio, with file access sandboxed to its own
workspace. The Researcher can pull evidence over MCP instead of local calls,
and the server is consumed three ways: a direct tool list, a stateful
session, and a generic ReAct agent.

**Result:** a fabricated citation was caught, a reviewer trying to overwrite
the draft was blocked, all path-escape attempts on the sandbox were rejected,
and local and MCP runs returned the same evidence. 11/11 checklist items
pass.

### M7 — Observability and reliability (`M7.py`)

Wraps each agent with timing spans (sent to Langfuse when keys are set,
always recorded locally) and builds a per-agent latency table. Separately, a
seeded fault-injection harness makes a real tool call fail at a fixed,
repeatable rate (200 calls, seed 42) and measures three hardening steps.

| Step | Success rate |
|---|---|
| Baseline | 57% |
| + retry with backoff | 89.5% |
| + fallback | 100% (no call raises; 21 of 200 served by the fallback) |

**Result:** tracing did not change the team's output. The circuit breaker
opened after exactly 3 failures and short-circuited the remaining 7 of 10
calls. The same seed reproduces the same numbers. 8/8 checklist items pass.

### M8 — Evaluation, guardrails, bias testing, deployment (`M8.py`)

- **Golden set:** 20 deterministic checks, 5 each for tool use, retrieval,
  planning, and final answer, so a failure shows which part broke.
- **Bias probe:** 3 matched pairs with identical financials and only the
  applicant name and city changed.
- **LLM judges:** Ragas, DeepEval, and TruLens score one sample question as a
  cross-check on the deterministic result, logged to Langfuse.
- **Guardrails:** a Pydantic schema for every response and a prompt-injection
  scan over both user input and retrieved policy text.
- **Service:** a FastAPI app (`/health`, `/chat`), also reachable through an
  ngrok tunnel.

**Result:** 20/20 golden items pass, all 3 bias pairs reached identical
decisions, 4/4 guardrail cases behaved correctly, and the API answered real
HTTP requests locally and over the tunnel. 9/9 checklist items pass.

## Repository contents

| File | Purpose |
|---|---|
| `M1.py` – `M8.py` | One file per milestone; later files import earlier ones |
| `wealthpilot_mcp_server.py` | Standalone MCP server used by M6 |
| `workflow_checkpoints.sqlite` | M5's checkpoint store |
| `ARCHITECTURE_REVIEW.md` | Longer design write-up |

## Running it

The data folder and API keys are not included in this repository.

1. Install the packages:

   ```bash
   pip install litellm python-dotenv pydantic langchain langchain-core langchain-litellm langchain-google-genai langchain-mcp-adapters langgraph langgraph-checkpoint-sqlite supermemory qdrant-client rank-bm25 sentence-transformers numpy pandas fastmcp tenacity langfuse ragas deepeval trulens-providers-litellm openai fastapi uvicorn pyngrok requests certifi
   ```

2. Place the generated dataset at `data/wealthpilot/` (`corpus/markdown/`,
   `intake/records.jsonl`, `mock_api/*.csv`).

3. Create a `.env` file with:

   - `GROQ_API_KEY` (or `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`)
   - `GEMINI_API_KEY`
   - `SUPERMEMORY_API_KEY`
   - `QDRANT_URL`, `QDRANT_API_KEY`
   - `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (optional, for tracing)
   - `NGROK_AUTHTOKEN` (needed for M8's public-tunnel check)

4. Run any milestone on its own:

   ```bash
   python M1.py
   ```

## Limitations

- All data is synthetic; nothing here has been tested on real loan documents.
- The evaluation sets are small: 5 scored retrieval questions, 20 golden
  checks, 3 bias pairs, and 1 question scored by the LLM judges.
- The restart test reopens the checkpoint database inside one process rather
  than killing and relaunching the program.
- The multi-agent team is exercised on one applicant per run; it has not been
  load-tested.
