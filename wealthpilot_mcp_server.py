"""
WEALTHPILOT MCP SERVER -- exposes real underwriting tools + a policy resource
over the Model Context Protocol (stdio transport), so any MCP-compatible
client (not just this project's own code) can use them.

Sandboxing rule, same as the course reference: everything reachable through
a file-path argument is resolved, checked to be inside WORKSPACE, and
rejected (raise, not silently clamp) if it isn't -- resolve() -> is_relative_to()
-> reject symlinks, because some filesystems resolve a symlink AFTER the
containment check, which would let a symlink hop outside the sandbox.

The strongest guardrail is a capability never exposed: there is no delete
tool and no arbitrary-path write tool here, on purpose.
"""

import csv
import json
import re
import sys
from pathlib import Path

from fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE = PROJECT_ROOT / "mcp_workspace"
WORKSPACE.mkdir(exist_ok=True)
DATA_DIR = PROJECT_ROOT / "data" / "wealthpilot"
MAX_READ_BYTES = 20_000

mcp = FastMCP("wealthpilot-underwriting")


def _safe_path(relative: str, root: Path = WORKSPACE) -> Path:
    """Resolve, then insist the result is still inside root, then refuse symlinks."""
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes the sandbox: {relative}")
    if candidate.is_symlink():
        raise ValueError(f"symlinks are not permitted: {relative}")
    return candidate


# =====================================================================
# Sandboxed file tools (workspace only -- no access to the wider project)
# =====================================================================
@mcp.tool()
def list_workspace_files(subdir: str = ".") -> list[str]:
    """List files under the MCP workspace sandbox. subdir must stay inside the sandbox."""
    target = _safe_path(subdir)
    if not target.is_dir():
        return []
    return sorted(str(p.relative_to(WORKSPACE)) for p in target.rglob("*") if p.is_file())


@mcp.tool()
def read_workspace_file(path: str) -> str:
    """Read a file from the MCP workspace sandbox, truncated to 20,000 bytes."""
    target = _safe_path(path)
    if not target.is_file():
        raise ValueError(f"no such file in workspace: {path}")
    data = target.read_bytes()[:MAX_READ_BYTES]
    return data.decode("utf-8", errors="replace")


@mcp.tool()
def write_note(name: str, content: str) -> str:
    """Write a short note into the MCP workspace. Name may not contain path separators."""
    if any(c in name for c in ("/", "\\")) or name in ("", ".", ".."):
        raise ValueError(f"invalid note name: {name!r}")
    target = WORKSPACE / name
    target.write_text(content)
    return f"wrote {len(content)} chars to {name}"


# =====================================================================
# Real WealthPilot underwriting tools -- same math/data as M2, exposed
# over MCP so any client (not just this project's own agent code) can use them.
# =====================================================================
_BUREAU = {row["applicant_id"]: row for row in csv.DictReader(open(DATA_DIR / "mock_api" / "bureau_reports.csv"))}

_STATEMENTS_BY_APPLICANT: dict[str, list[dict]] = {}
for _row in csv.DictReader(open(DATA_DIR / "mock_api" / "bank_statements.csv")):
    _STATEMENTS_BY_APPLICANT.setdefault(_row["applicant_id"], []).append(_row)


@mcp.tool()
def get_bureau_report(applicant_id: str) -> dict:
    """Fetch an applicant's credit bureau report (score, delinquencies, trade lines)."""
    row = _BUREAU.get(applicant_id)
    if not row:
        raise ValueError(f"no bureau record for {applicant_id}")
    return row


@mcp.tool()
def get_bank_statement_summary(applicant_id: str) -> dict:
    """Fetch an applicant's aggregated bank statement summary (inflow, outflow, balance, bounced instruments)."""
    rows = _STATEMENTS_BY_APPLICANT.get(applicant_id)
    if not rows:
        raise ValueError(f"no bank statements for {applicant_id}")
    n = len(rows)
    return {
        "months_of_history": n,
        "avg_monthly_inflow_inr": round(sum(float(r["inflow_inr"]) for r in rows) / n, 2),
        "avg_monthly_outflow_inr": round(sum(float(r["outflow_inr"]) for r in rows) / n, 2),
        "avg_closing_balance_inr": round(sum(float(r["closing_balance_inr"]) for r in rows) / n, 2),
        "total_bounced_instruments": sum(int(r["bounced_instruments"]) for r in rows),
    }


@mcp.tool()
def compute_dscr(ebitda_inr: float, existing_annual_debt_service_inr: float, new_annual_debt_service_inr: float) -> dict:
    """Compute the Debt-Service Coverage Ratio and whether it clears the 1.25x floor."""
    total = existing_annual_debt_service_inr + new_annual_debt_service_inr
    if total <= 0:
        return {"dscr": None, "error": "total debt service must be > 0"}
    dscr = round(ebitda_inr / total, 2)
    return {"dscr": dscr, "meets_1.25x_floor": dscr >= 1.25}


@mcp.tool()
def compute_emi(principal_inr: float, annual_rate_pct: float, tenor_months: int) -> dict:
    """Compute the monthly EMI and annual debt service for a proposed loan."""
    r = annual_rate_pct / 12 / 100
    emi = principal_inr / tenor_months if r == 0 else principal_inr * r * (1 + r) ** tenor_months / ((1 + r) ** tenor_months - 1)
    return {"monthly_emi_inr": round(emi, 2), "annual_debt_service_inr": round(emi * 12, 2)}


# Policy search -- lazily builds M3/M4's embedding index the first time it's
# called, so the server starts instantly and only pays the embedding cost if
# a client actually asks a policy question.
_POLICY_INDEX = None


def _policy_index():
    global _POLICY_INDEX
    if _POLICY_INDEX is None:
        sys.path.insert(0, str(PROJECT_ROOT))
        from M3 import build_policy_index
        _POLICY_INDEX = build_policy_index()
        for i, c in enumerate(_POLICY_INDEX):
            c.setdefault("cid", i)
    return _POLICY_INDEX


@mcp.tool()
def search_policy(query: str, k: int = 3) -> list[dict]:
    """Semantic search over the 13 Ashva Capital credit policy documents. Returns cited, sourced hits."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from M3 import embed, cosine
    index = _policy_index()
    q = embed(query)
    scored = sorted(index, key=lambda c: cosine(q, c["embedding"]), reverse=True)
    return [{"id": f"POLICY-{c['cid']}", "source": c["source"], "text": c["text"]} for c in scored[:k]]


@mcp.resource("policy://citation-rules")
def citation_rules() -> str:
    """The citation policy every WealthPilot agent output must follow."""
    return (
        "1. Every policy claim in an underwriting memo must cite a real [POLICY-<id>] "
        "or [TOOL-<NAME>] tag returned by search_policy or a tool call.\n"
        "2. A citation tag that does not correspond to a real retrieved id is a "
        "fabricated citation -- a compliance failure, not a stylistic error.\n"
        "3. If the policy corpus does not address a question, say so explicitly; "
        "never infer a plausible-sounding policy."
    )


if __name__ == "__main__":
    # NEVER print to stdout here -- it is the JSON-RPC channel for stdio transport.
    mcp.run(transport="stdio", show_banner=False)
