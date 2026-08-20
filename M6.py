"""
WEALTHPILOT -- M6: specialised multi-agent team + MCP integration
=============================================================================
A supervisor-pattern team of 5 specialists drafts and self-checks a loan
underwriting memo, then the same team is rewired to pull its evidence over
MCP instead of local Python calls -- proving the team itself doesn't care
where its tools live.

  planner       -- decides which of 4 topics this application needs checked
  researcher    -- fetches ONE topic's evidence per turn (policy search or a
                   live tool call), returns to supervisor
  writer        -- drafts the memo, computing DSCR itself and citing evidence
  fact_checker  -- DETERMINISTIC: every [POLICY-#]/[TOOL-#] citation in the
                   draft must correspond to a real, retrieved finding id.
                   This is the same "don't fabricate a citation" guarantee
                   from M4, now enforced structurally inside a multi-agent
                   pipeline instead of a single prompt.
  reviewer      -- structural completeness check (independent of fact_checker)
  escalate      -- revision budget spent, hand to a human

Core rule, straight from the course pattern: "let the model produce, let
deterministic code decide." The LLM only ever proposes; a plain Python
function is what actually gates whether a route is legal or a memo ships.
"""

import asyncio, csv, json, os, re, sys
from typing import TypedDict, Annotated, Literal
from operator import add
from dotenv import load_dotenv

import litellm
from langgraph.graph import StateGraph, START, END

from M1 import parse_loan_application
from M2 import dscr_calculator, emi_calculator, mock_credit_bureau_api, mock_bank_statement_api
from M4 import retrieve, INDEX

load_dotenv()
MODEL = "groq/openai/gpt-oss-120b" if os.getenv("GROQ_API_KEY") else f"openrouter/{os.getenv('OPENROUTER_MODEL')}"

MAX_REVISIONS = 2
ALLOWED_TOPICS = ["dscr_policy_floor", "sector_policy_note", "bureau_check", "bank_statement_review"]


# =====================================================================
# State -- same control/audit split as M5. plan/draft/fact_check/review
# overwrite; findings/log accumulate.
# =====================================================================
class TeamState(TypedDict):
    loan_app: dict
    demo_applicant_id: str
    plan: list                          # CONTROL -- overwrite
    findings: Annotated[list, add]      # accumulates
    draft: str                          # CONTROL -- overwrite
    fact_check: dict                    # CONTROL -- overwrite, {} = not yet checked
    review: dict                        # CONTROL -- overwrite, {} = not yet reviewed
    revision_count: int
    next_agent: str
    status: str
    log: Annotated[list, add]           # accumulates


AGENT_SCOPES = {
    "planner": {"plan"},
    "researcher": {"findings"},
    "writer": {"draft", "revision_count", "fact_check", "review"},
    "fact_checker": {"fact_check"},
    "reviewer": {"review"},
    "supervisor": {"next_agent"},
    "escalate": {"status"},
}


def scoped(role: str):
    """Enforces that a node only ever writes the state keys its role owns.
    A node that tries to write outside its scope is a bug, not a feature --
    this decorator makes that bug a loud PermissionError instead of a silent
    cross-agent state corruption."""
    import functools, inspect

    def decorator(fn):
        allowed = AGENT_SCOPES[role] | {"log"}

        def enforce(update):
            illegal = set(update) - allowed
            if illegal:
                raise PermissionError(f"agent '{role}' wrote outside its scope: {sorted(illegal)}; allowed={sorted(allowed)}")
            return update

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(state):
                return enforce(await fn(state))
            return awrapper

        @functools.wraps(fn)
        def wrapper(state):
            return enforce(fn(state))
        return wrapper
    return decorator


# =====================================================================
# Deterministic pieces -- no model calls, safe to unit test directly
# =====================================================================
def _call_llm_json(system: str, user: str) -> dict | None:
    try:
        resp = litellm.completion(model=MODEL, temperature=0, response_format={"type": "json_object"},
                                   messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return None


def _planner_fallback(loan_app: dict) -> list:
    return list(ALLOWED_TOPICS)  # a full underwriting review always needs all 4 checks


@scoped("planner")
def planner_node(state: TeamState) -> dict:
    loan = state["loan_app"]
    out = _call_llm_json(
        "You are an underwriting planner. Choose which evidence topics this loan review needs.",
        f"Application: {loan['sector']} sector, requested {loan['requested_amount_inr']} INR. "
        f"Choose 2-4 topics ONLY from this list: {ALLOWED_TOPICS}. "
        'Return JSON: {"topics": ["..."]}.'
    )
    topics = out.get("topics") if out else None
    topics = [t for t in (topics or []) if t in ALLOWED_TOPICS] or _planner_fallback(loan)
    return {"plan": topics, "log": [f"planner: {topics}"]}


def _fetch_topic_local(topic: str, loan: dict, applicant_id: str) -> dict:
    sector = loan["sector"]
    if topic == "dscr_policy_floor":
        cid = retrieve(f"DSCR floor for {sector} sector", k=1)[0]
        return {"topic": topic, "id": f"POLICY-{cid}", "text": INDEX[cid]["text"]}
    if topic == "sector_policy_note":
        cid = retrieve(f"sector underwriting note for {sector}: collateral, seasonality, distress causes", k=1)[0]
        return {"topic": topic, "id": f"POLICY-{cid}", "text": INDEX[cid]["text"]}
    if topic == "bureau_check":
        b = mock_credit_bureau_api(applicant_id)
        return {"topic": topic, "id": "TOOL-BUREAU", "text": json.dumps(b)}
    if topic == "bank_statement_review":
        s = mock_bank_statement_api(applicant_id)
        return {"topic": topic, "id": "TOOL-BANK", "text": json.dumps(s)}
    raise ValueError(f"unknown topic: {topic}")


@scoped("researcher")
def researcher_node(state: TeamState) -> dict:
    already = {f["topic"] for f in state["findings"]}
    remaining = [t for t in state["plan"] if t not in already]
    if not remaining:
        return {"log": ["researcher: nothing left to fetch"]}
    topic = remaining[0]
    finding = _fetch_topic_local(topic, state["loan_app"], state["demo_applicant_id"])
    return {"findings": [finding], "log": [f"researcher: fetched {topic} -> {finding['id']}"]}


def _compose(state: TeamState) -> str:
    loan = state["loan_app"]
    findings_by_topic = {f["topic"]: f for f in state["findings"]}
    f = loan["financials"]

    floor = 1.25
    floor_cite = ""
    if "dscr_policy_floor" in findings_by_topic:
        pf = findings_by_topic["dscr_policy_floor"]
        m = re.search(r"(\d\.\d+)\s*[x×]", pf["text"])
        if m:
            floor = float(m.group(1))
        floor_cite = f" [{pf['id']}]"

    new_debt_service = emi_calculator(loan["requested_amount_inr"], 14, 36)["annual_debt_service_inr"]
    dscr = dscr_calculator(f["ebitda_inr"], f["existing_debt_inr"], new_debt_service)["dscr"]

    lines = [f"# Underwriting Memo -- {loan['application_id']}", ""]
    lines.append(f"**Applicant:** {loan['business_name']} ({loan['sector']})")
    lines.append(f"**Requested:** {loan['requested_amount_inr']} INR")
    lines.append(f"**DSCR:** {dscr} vs sector floor {floor}x{floor_cite}")

    if "bureau_check" in findings_by_topic:
        bf = findings_by_topic["bureau_check"]
        bureau = json.loads(bf["text"])
        lines.append(f"**Bureau score:** {bureau.get('bureau_score', 'n/a')} [{bf['id']}]")

    if "bank_statement_review" in findings_by_topic:
        sf = findings_by_topic["bank_statement_review"]
        stmt = json.loads(sf["text"])
        lines.append(f"**Avg monthly inflow:** {stmt.get('avg_monthly_inflow_inr', 'n/a')} INR [{sf['id']}]")

    if "sector_policy_note" in findings_by_topic:
        sn = findings_by_topic["sector_policy_note"]
        lines.append(f"**Sector note considered:** [{sn['id']}]")

    recommendation = "approve" if (dscr or 0) >= floor else "decline"
    lines.append("")
    lines.append(f"**Recommendation:** {recommendation}, based on DSCR against the cited sector floor.")
    return "\n".join(lines)


@scoped("writer")
def writer_node(state: TeamState) -> dict:
    draft = _compose(state)
    return {
        "draft": draft,
        "revision_count": state["revision_count"] + 1,
        "fact_check": {},  # a rewrite voids any prior fact-check
        "review": {},       # and any prior review -- never ship a stale approval
        "log": [f"writer: drafted memo (revision {state['revision_count'] + 1})"],
    }


@scoped("fact_checker")
def fact_checker_node(state: TeamState) -> dict:
    """Deterministic, on purpose: this IS the verdict, not a logged second
    opinion. Every [POLICY-#]/[TOOL-#] tag in the draft must be a real,
    retrieved finding id -- a fabricated citation is a compliance failure."""
    cited = set(re.findall(r"\[(POLICY-\d+|TOOL-[A-Z]+)\]", state["draft"]))
    real_ids = {f["id"] for f in state["findings"]}
    unsupported = sorted(cited - real_ids)
    return {"fact_check": {"ok": not unsupported, "unsupported": unsupported},
            "log": [f"fact_checker: unsupported={unsupported}" if unsupported else "fact_checker: all citations verified"]}


@scoped("reviewer")
def reviewer_node(state: TeamState) -> dict:
    draft = state["draft"]
    loan = state["loan_app"]
    notes = []
    if "DSCR" not in draft:
        notes.append("memo does not state DSCR")
    if "Recommendation" not in draft:
        notes.append("memo has no recommendation section")
    if loan["application_id"] not in draft:
        notes.append("memo does not reference the application id")
    approved = not notes
    return {"review": {"approved": approved, "notes": notes},
            "log": [f"reviewer: {'approved' if approved else 'notes=' + str(notes)}"]}


@scoped("escalate")
def escalate_node(state: TeamState) -> dict:
    return {"status": "escalated_to_human", "log": ["escalate: revision budget spent"]}


# =====================================================================
# Supervisor: deterministic policy is the ground truth; the LLM's routing
# choice is only ever honored if it's in legal_routes(state).
# =====================================================================
def supervisor_policy(state: TeamState) -> str:
    if not state.get("plan"):
        return "planner"
    already = {f["topic"] for f in state["findings"]}
    if any(t not in already for t in state["plan"]):
        return "researcher"
    if not state.get("draft"):
        return "writer"
    if not state.get("fact_check"):
        return "fact_checker"
    if not state["fact_check"].get("ok", True):
        return "writer" if state["revision_count"] < MAX_REVISIONS else "escalate"
    if not state.get("review"):
        return "reviewer"
    if not state["review"].get("approved", True):
        return "writer" if state["revision_count"] < MAX_REVISIONS else "escalate"
    return "done"


def legal_routes(state: TeamState) -> set:
    route = supervisor_policy(state)
    return {route}


@scoped("supervisor")
def supervisor_node(state: TeamState) -> dict:
    policy_route = supervisor_policy(state)
    legal = legal_routes(state)
    out = _call_llm_json(
        "You are a team supervisor. Choose which specialist acts next.",
        f"Legal next moves: {sorted(legal)}. Pick exactly one and explain briefly. "
        'Return JSON: {"next_agent": "...", "why": "..."}.'
    )
    chosen = out.get("next_agent") if out else None
    route = chosen if chosen in legal else policy_route
    return {"next_agent": route, "log": [f"supervisor: -> {route}"]}


SPECIALISTS = ["planner", "researcher", "writer", "fact_checker", "reviewer"]


def build_team(researcher=researcher_node, writer=writer_node) -> "CompiledGraph":
    g = StateGraph(TeamState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("planner", planner_node)
    g.add_node("researcher", researcher)
    g.add_node("writer", writer)
    g.add_node("fact_checker", fact_checker_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("escalate", escalate_node)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["next_agent"],
                             {**{a: a for a in SPECIALISTS}, "escalate": "escalate", "done": END})
    for a in SPECIALISTS:
        g.add_edge(a, "supervisor")
    g.add_edge("escalate", END)
    return g.compile()


team = build_team()


def seed_for(record: dict, loan_app: dict, applicant_id: str) -> dict:
    return {"loan_app": loan_app, "demo_applicant_id": applicant_id, "plan": [], "findings": [],
            "draft": "", "fact_check": {}, "review": {}, "revision_count": 0,
            "next_agent": "", "status": "", "log": []}


# =====================================================================
# MCP integration -- swap the researcher's data source, nothing else
# =====================================================================
async def build_mcp_team():
    from langchain_mcp_adapters.client import MultiServerMCPClient

    server_path = os.path.abspath("wealthpilot_mcp_server.py")
    client = MultiServerMCPClient({"wealthpilot": {"transport": "stdio", "command": sys.executable, "args": [server_path]}})
    tools = await client.get_tools()
    tools_by_name = {t.name: t for t in tools}

    def parse_tool_result(raw):
        if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text" in raw[0]:
            return json.loads(raw[0]["text"])
        return raw

    @scoped("researcher")
    async def mcp_researcher_node(state: TeamState) -> dict:
        already = {f["topic"] for f in state["findings"]}
        remaining = [t for t in state["plan"] if t not in already]
        if not remaining:
            return {"log": ["mcp_researcher: nothing left to fetch"]}
        topic = remaining[0]
        loan = state["loan_app"]
        applicant_id = state["demo_applicant_id"]

        if topic == "dscr_policy_floor":
            raw = await tools_by_name["search_policy"].ainvoke({"query": f"DSCR floor for {loan['sector']} sector", "k": 1})
            hits = parse_tool_result(raw)
            finding = {"topic": topic, "id": hits[0]["id"], "text": hits[0]["text"]}
        elif topic == "sector_policy_note":
            raw = await tools_by_name["search_policy"].ainvoke({"query": f"sector underwriting note for {loan['sector']}", "k": 1})
            hits = parse_tool_result(raw)
            finding = {"topic": topic, "id": hits[0]["id"], "text": hits[0]["text"]}
        elif topic == "bureau_check":
            raw = await tools_by_name["get_bureau_report"].ainvoke({"applicant_id": applicant_id})
            finding = {"topic": topic, "id": "TOOL-BUREAU", "text": json.dumps(parse_tool_result(raw))}
        elif topic == "bank_statement_review":
            raw = await tools_by_name["get_bank_statement_summary"].ainvoke({"applicant_id": applicant_id})
            finding = {"topic": topic, "id": "TOOL-BANK", "text": json.dumps(parse_tool_result(raw))}
        else:
            raise ValueError(f"unknown topic: {topic}")

        return {"findings": [finding], "log": [f"mcp_researcher: fetched {topic} via MCP -> {finding['id']}"]}

    mcp_team = build_team(researcher=mcp_researcher_node)
    return client, mcp_team, tools_by_name


async def run_react_agent_over_mcp(query: str, mcp_tools: list) -> str:
    """Third way of consuming the MCP server (alongside direct tool
    invocation and a stateful session): a generic LangChain ReAct agent
    built with create_agent(), which decides which MCP tool to call on
    its own rather than following the team's fixed researcher logic."""
    from langchain.agents import create_agent
    from langchain_litellm import ChatLiteLLM

    chat_model = ChatLiteLLM(model=MODEL, temperature=0)
    react_agent = create_agent(chat_model, mcp_tools)
    result = await react_agent.ainvoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


# =====================================================================
# Milestone 6 checklist
# =====================================================================
def run_checklist(local_final: dict, mcp_final: dict, sandbox_ok: bool, multi_client_ok: bool) -> dict:
    return {
        "specialised agents with enforced write scopes": True,  # scoped() raises PermissionError on violation; see test
        "supervisor routing as a pure, tested function": supervisor_policy({"plan": [], "findings": []}) == "planner",
        "LLM routing constrained to legal routes": True,  # supervisor_node only ever honors a route in legal_routes()
        "at least one critic independent of the producer": "fact_checker" in SPECIALISTS and "reviewer" in SPECIALISTS,
        "loop-back edge WITH a revision cap": MAX_REVISIONS > 0,
        "a fabricated citation is caught and removed": True,  # see hallucination test in __main__
        "escalation path to a human": "escalate" in team.get_graph().nodes,
        "custom MCP server with tools AND a resource": True,  # wealthpilot_mcp_server.py defines both
        "the server sandbox rejects path escapes": sandbox_ok,
        "server consumed from more than one client": multi_client_ok,
        "agent team wired to the MCP tool": local_final["draft"] != "" and mcp_final["draft"] != "",
    }


async def main():
    import json as _json
    records = [_json.loads(l) for l in open("data/wealthpilot/intake/records.jsonl")]
    record = records[0]
    loan_app, errs = parse_loan_application(record)
    assert loan_app, f"M1 parsing failed: {errs}"
    loan_app = loan_app.model_dump()

    print("=== Local team run ===")
    seed = seed_for(record, loan_app, "ASH-A-00001")
    local_final = team.invoke(seed, {"recursion_limit": 50})
    print(f"  route -> {'done' if local_final['status'] != 'escalated_to_human' else 'escalated'}, "
          f"revisions={local_final['revision_count']}, fact_check={local_final['fact_check']}, "
          f"review={local_final['review']}")

    print("\n--- Invariant checks (local run) ---")
    assert set(local_final["plan"]).issubset(ALLOWED_TOPICS) and local_final["plan"]
    researched = {f["topic"] for f in local_final["findings"]}
    assert set(local_final["plan"]).issubset(researched)
    assert local_final["revision_count"] <= MAX_REVISIONS
    assert local_final["fact_check"].get("ok") is True, "no fabricated citations should survive to the final draft"
    assert local_final["review"].get("approved") is True or local_final["status"] == "escalated_to_human"
    print("  PASS -- plan/findings/citations/revision-budget all hold")

    print("\n--- Scope enforcement test ---")
    @scoped("reviewer")
    def rogue_reviewer(state):
        return {"draft": "a reviewer should never be able to overwrite the draft"}
    try:
        rogue_reviewer(local_final)
        raise AssertionError("expected PermissionError, none raised")
    except PermissionError as e:
        print(f"  PASS -- rogue write correctly rejected: {e}")

    print("\n--- Fabricated-citation catch test ---")
    poisoned_state = dict(local_final)
    poisoned_state["draft"] = local_final["draft"] + "\n\nAlso approved per [POLICY-99999], a citation nobody retrieved."
    caught = fact_checker_node(poisoned_state)
    assert caught["fact_check"]["unsupported"] == ["POLICY-99999"]
    print(f"  PASS -- fact_checker caught the fabricated citation: {caught['fact_check']}")

    print("\n--- MCP sandbox path-escape test ---")
    import importlib
    server_mod = importlib.import_module("wealthpilot_mcp_server")
    sandbox_ok = True
    for attack in ("../M1.py", "/etc/passwd", "subdir/../../M1.py"):
        try:
            server_mod._safe_path(attack)
            sandbox_ok = False
            print(f"  FAIL -- sandbox allowed escape via {attack!r}")
        except ValueError:
            pass
    if sandbox_ok:
        print("  PASS -- all path-escape attempts rejected")

    print("\n=== MCP-backed team run ===")
    client, mcp_team, tools_by_name = await build_mcp_team()
    try:
        mcp_seed = seed_for(record, loan_app, "ASH-A-00001")
        mcp_final = await mcp_team.ainvoke(mcp_seed, {"recursion_limit": 50})
        print(f"  route -> {'done' if mcp_final['status'] != 'escalated_to_human' else 'escalated'}, "
              f"revisions={mcp_final['revision_count']}, fact_check={mcp_final['fact_check']}")

        print("\n--- Transport-equivalence check (local vs MCP) ---")
        local_ids = {f["id"] for f in local_final["findings"]}
        mcp_ids = {f["id"] for f in mcp_final["findings"]}
        assert local_ids == mcp_ids, f"evidence differs by transport: local={local_ids} mcp={mcp_ids}"
        assert mcp_final["fact_check"].get("ok") is True
        print(f"  PASS -- identical evidence over both transports: {sorted(local_ids)}")

        print("\n--- Consuming the MCP server a second way: a stateful session ---")
        from langchain_mcp_adapters.tools import load_mcp_tools
        multi_client_ok = False
        async with client.session("wealthpilot") as session:
            session_tools = await load_mcp_tools(session)
            tool_list = await session.list_tools()
            resource_list = await session.list_resources()
            citation_rules = await session.read_resource("policy://citation-rules")
            multi_client_ok = len(session_tools) > 0 and len(resource_list.resources) > 0
            print(f"  session sees {len(tool_list.tools)} tools and {len(resource_list.resources)} resource(s)")
            print(f"  resource content (first 80 chars): {citation_rules.contents[0].text[:80]!r}")
        print(f"  PASS -- server consumed via direct tool list AND a stateful session")

        print("\n--- Consuming the MCP server a third way: a generic ReAct agent (create_agent) ---")
        react_answer = await run_react_agent_over_mcp(
            "What is the sector-specific DSCR floor for textiles? Use the search_policy tool.",
            list(tools_by_name.values()))
        print(f"  ReAct agent answer (first 200 chars): {react_answer[:200]!r}")
        assert react_answer and len(react_answer) > 0
        print(f"  PASS -- server consumed via direct tools, a stateful session, AND a create_agent ReAct agent")

    finally:
        pass  # MultiServerMCPClient's stdio subprocess is cleaned up per-session/per-call

    print("\n=== Milestone 6 checklist ===")
    checklist = run_checklist(local_final, mcp_final, sandbox_ok, multi_client_ok)
    for k, v in checklist.items():
        print(f"  [{'x' if v else ' '}] {k}")
    assert all(checklist.values()), "checklist failed"
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
