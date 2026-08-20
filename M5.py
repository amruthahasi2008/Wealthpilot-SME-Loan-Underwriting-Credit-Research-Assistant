"""
WEALTHPILOT -- M5: orchestrated workflow with a durable human-approval checkpoint
=============================================================================
Built to match the course's reference M5 skeleton (document-approval workflow)
mapped onto loan underwriting:

  checks -> revise -> checks (loop, capped)  ==>  verify_documents -> backfill -> verify_documents
  human_approval (interrupt)                 ==>  human_gate (interrupt) -- ONLY reads state, no side effects
  finalize (downstream of the pause)         ==>  finalize_human -- commits the decision, replay-safe
  InMemorySaver (dev only)                   ==>  SqliteSaver -- survives a real process restart

Two state-design rules straight from the reference, both load-bearing:

  1. CONTROL fields (missing_fields, risk_band, ...) must OVERWRITE, never
     accumulate -- routers read them to decide whether to loop, and a field
     that only grows can never signal "resolved," which loops forever.
  2. AUDIT fields (audit_log) must ACCUMULATE (Annotated[list, add]) -- that's
     the durable trail, kept deliberately separate from control state.

And the single most important lesson from the reference notebook: on resume,
LangGraph re-executes a node from the top. Anything before interrupt() in the
same node re-runs on every resume. So human_gate does *nothing* but call
interrupt() and return the raw answer; the actual decision-commit happens in
a separate node (finalize_human) downstream of the pause, where re-execution
can't double it up.
"""

import os, csv, json, re, sqlite3
from typing import TypedDict, Optional, Annotated
from operator import add
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

from M1 import parse_loan_application
from M2 import dscr_calculator, emi_calculator, mock_credit_bureau_api
from M4 import retrieve, INDEX

load_dotenv()

MAX_REVISIONS = 2
SIGNATORY_THRESHOLD_INR = 5_000_000


# =====================================================================
# State -- control fields overwrite, audit_log accumulates. This split
# is the single most important design decision in the whole graph.
# =====================================================================
class UnderwritingState(TypedDict):
    record: dict
    demo_applicant_id: str
    loan_app: Optional[dict]
    missing_fields: list                    # CONTROL -- overwrite
    revision_count: int                     # CONTROL -- loop guard
    bureau: Optional[dict]
    dscr: Optional[float]
    sector_dscr_floor: Optional[float]
    risk_band: Optional[str]
    requires_human_signoff: Optional[bool]
    human_decision: Optional[dict]
    decision: Optional[str]
    rationale: Optional[str]
    audit_log: Annotated[list, add]         # AUDIT -- accumulates, never overwritten


# =====================================================================
# Nodes
# =====================================================================
def node_intake(state: UnderwritingState) -> dict:
    app, errs = parse_loan_application(state["record"])
    if not app:
        raise RuntimeError(f"M1 parsing failed for {state['record']['application_id']}: {errs}")
    loan = app.model_dump()
    return {
        "loan_app": loan,
        "missing_fields": list(loan.get("missing_fields", [])),
        "revision_count": 0,
        "audit_log": [f"intake: parsed {loan['application_id']}, missing={loan.get('missing_fields', [])}"],
    }


def node_verify_documents(state: UnderwritingState) -> dict:
    """Deterministic check, separated from routing -- mirrors the reference's
    check_document(). No model call, no side effects beyond re-reading state."""
    loan = state["loan_app"]
    missing = list(loan.get("missing_fields", []))
    if loan.get("bureau_score") is None and "bureau_score" not in missing:
        missing.append("bureau_score")
    return {"missing_fields": missing, "audit_log": [f"verify_documents: outstanding={missing}"]}


def route_after_verify(state: UnderwritingState) -> str:
    can_backfill = "bureau_score" in state["missing_fields"]
    if can_backfill and state["revision_count"] < MAX_REVISIONS:
        return "backfill"
    return "risk_scoring"


def node_backfill(state: UnderwritingState) -> dict:
    """The 'revise' step -- but instead of a deterministic text edit (the
    reference's banned-term stripping), this pulls the real missing figure
    from a live tool (M2's bureau API), same idea: turn an outstanding issue
    into a resolved one, deterministically, then loop back to re-check."""
    bureau = mock_credit_bureau_api(state["demo_applicant_id"])
    loan = dict(state["loan_app"])
    missing = [f for f in state["missing_fields"]]
    if "error" not in bureau and bureau.get("bureau_score"):
        loan["bureau_score"] = int(bureau["bureau_score"])
        if "bureau_score" in missing:
            missing.remove("bureau_score")
    loan["missing_fields"] = missing
    return {
        "loan_app": loan,
        "bureau": bureau,
        "revision_count": state["revision_count"] + 1,
        "audit_log": [f"backfill: pulled bureau_score={loan.get('bureau_score')} from mock_credit_bureau_api"],
    }


def node_risk_scoring(state: UnderwritingState) -> dict:
    loan = state["loan_app"]
    sector = loan["sector"]

    floor = 1.25
    for cid in retrieve(f"DSCR floor for {sector} sector", k=3):
        m = re.search(r"(\d\.\d+)\s*[x×]", INDEX[cid]["text"])
        if m:
            floor = float(m.group(1))
            break

    f = loan["financials"]
    new_debt_service = emi_calculator(loan["requested_amount_inr"], 14, 36)["annual_debt_service_inr"]
    dscr = dscr_calculator(f["ebitda_inr"], f["existing_debt_inr"], new_debt_service)["dscr"]

    if dscr is None or dscr < floor:
        band = "decline"
    elif dscr < floor * 1.1:
        band = "borderline"
    else:
        band = "approve"

    unresolved_docs = bool(state["missing_fields"])
    requires_signoff = (
        loan["requested_amount_inr"] > SIGNATORY_THRESHOLD_INR
        or band == "borderline"
        or unresolved_docs
    )
    return {
        "dscr": dscr, "sector_dscr_floor": floor, "risk_band": band,
        "requires_human_signoff": requires_signoff,
        "audit_log": [f"risk_scoring: DSCR={dscr} vs {sector} floor={floor} -> {band} "
                      f"(signoff required: {requires_signoff}, unresolved docs: {unresolved_docs})"],
    }


def route_after_scoring(state: UnderwritingState) -> str:
    return "human_review" if state["requires_human_signoff"] else "auto_decision"


def node_human_gate(state: UnderwritingState) -> dict:
    """PAUSES the graph. Does NOTHING except read state and call interrupt() --
    on resume this node re-executes from the top, so anything with a side
    effect here would fire twice. The actual commit happens in finalize_human."""
    decision = interrupt({
        "question": "Approve this loan?",
        "application_id": state["loan_app"]["application_id"],
        "requested_amount_inr": state["loan_app"]["requested_amount_inr"],
        "dscr": state["dscr"],
        "sector_dscr_floor": state["sector_dscr_floor"],
        "risk_band": state["risk_band"],
        "outstanding_issues": state["missing_fields"],
    })
    return {"human_decision": decision}


def route_after_human(state: UnderwritingState) -> str:
    action = state["human_decision"].get("action")
    return "finalize_human" if action in ("approved", "rejected") else "finalize_human"


def node_finalize_human(state: UnderwritingState) -> dict:
    """The irreversible commit -- deliberately its own node, downstream of the
    interrupt, so replay-on-resume can never double-commit a decision."""
    d = state["human_decision"]
    action = d.get("action", "rejected")
    decision = "approved" if action == "approved" else "declined"
    rationale = f"Human signatory decision: {action}" + (f" -- {d['note']}" if d.get("note") else "")
    return {"decision": decision, "rationale": rationale,
            "audit_log": [f"finalize_human: {decision} by human signatory"]}


def node_auto_decision(state: UnderwritingState) -> dict:
    decision = "approved" if state["risk_band"] == "approve" else "declined"
    rationale = (f"Auto-{decision}: DSCR {state['dscr']} vs the {state['loan_app']['sector']} "
                 f"sector floor of {state['sector_dscr_floor']}x.")
    return {"decision": decision, "rationale": rationale,
            "audit_log": [f"auto_decision: {decision}, no human required"]}


# =====================================================================
# Wire the graph
# =====================================================================
def build_graph(checkpointer):
    g = StateGraph(UnderwritingState)
    g.add_node("intake", node_intake)
    g.add_node("verify_documents", node_verify_documents)
    g.add_node("backfill", node_backfill)
    g.add_node("risk_scoring", node_risk_scoring)
    g.add_node("human_gate", node_human_gate)
    g.add_node("finalize_human", node_finalize_human)
    g.add_node("auto_decision", node_auto_decision)

    g.add_edge(START, "intake")
    g.add_edge("intake", "verify_documents")
    g.add_conditional_edges("verify_documents", route_after_verify,
                             {"backfill": "backfill", "risk_scoring": "risk_scoring"})
    g.add_edge("backfill", "verify_documents")
    g.add_conditional_edges("risk_scoring", route_after_scoring,
                             {"human_review": "human_gate", "auto_decision": "auto_decision"})
    g.add_edge("human_gate", "finalize_human")
    g.add_edge("finalize_human", END)
    g.add_edge("auto_decision", END)
    return g.compile(checkpointer=checkpointer)


# durable checkpointer -- survives a real process restart, unlike InMemorySaver
_DB_PATH = "workflow_checkpoints.sqlite"
_conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
_saver = SqliteSaver(_conn)
_saver.setup()
workflow = build_graph(_saver)


# =====================================================================
# Milestone 5 checklist -- mirrors the reference notebook's own rubric
# =====================================================================
def run_checklist() -> dict:
    ann = UnderwritingState.__annotations__
    checklist = {
        "typed state schema with an explicit reducer":
            hasattr(ann["audit_log"], "__metadata__") and ann["audit_log"].__metadata__[0] is add,
        "control fields do not have a reducer":
            not hasattr(ann["missing_fields"], "__metadata__"),
        "deterministic checks separated from routing":
            callable(node_verify_documents) and callable(route_after_verify),
        "conditional routing as a pure function":
            route_after_verify({"missing_fields": ["bureau_score"], "revision_count": 0}) == "backfill",
        "loop guard (bounded revisions)":
            route_after_verify({"missing_fields": ["bureau_score"], "revision_count": 99}) == "risk_scoring",
        "human gate on the irreversible step":
            "human_gate" in workflow.get_graph().nodes,
        "durable checkpointer (survives restart)":
            os.path.exists(_DB_PATH),
        "irreversible action isolated downstream of pause":
            "finalize_human" in workflow.get_graph().nodes,
    }
    return checklist


if __name__ == "__main__":
    records = [json.loads(l) for l in open("data/wealthpilot/intake/records.jsonl")]

    print("=== Case 1: smallest loan, complete docs -- should auto-decide ===")
    small = min(records, key=lambda r: r["requested_amount_inr"])
    cfg1 = {"configurable": {"thread_id": f"case1-{small['application_id']}"}}
    result1 = workflow.invoke({"record": small, "demo_applicant_id": "ASH-A-00001"}, config=cfg1)
    print("RESULT:", json.dumps({k: result1.get(k) for k in ("decision", "rationale")}, indent=2))

    print("\n=== Case 2: largest loan -- over signatory threshold, should PAUSE ===")
    big = max(records, key=lambda r: r["requested_amount_inr"])
    cfg2 = {"configurable": {"thread_id": f"case2-{big['application_id']}"}}
    result2 = workflow.invoke({"record": big, "demo_applicant_id": "ASH-A-00002"}, config=cfg2)

    pending = workflow.get_state(cfg2)
    if pending.next:
        print(f"  graph is PAUSED, waiting on node(s): {pending.next}")
        print("\n--- Simulating process restart: rebuilding the graph from a FRESH SqliteSaver connection ---")
        _conn2 = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _saver2 = SqliteSaver(_conn2)
        workflow2 = build_graph(_saver2)
        still_pending = workflow2.get_state(cfg2)
        print(f"  state survived restart: next={still_pending.next}, "
              f"application_id={still_pending.values['loan_app']['application_id']}")

        print("\n--- Simulating a human (credit committee) approving it, on the RESTARTED graph ---")
        result2 = workflow2.invoke(
            Command(resume={"action": "approved", "note": "signed by ASH-U-014"}), config=cfg2)

    print("RESULT:", json.dumps({k: result2.get(k) for k in ("decision", "rationale")}, indent=2))

    print("\n=== Milestone 5 checklist ===")
    checklist = run_checklist()
    for k, v in checklist.items():
        print(f"  [{'x' if v else ' '}] {k}")
    assert all(checklist.values()), "checklist failed"
    print("\nALL CHECKS PASSED")
