"""
WEALTHPILOT -- M2: tool-enabled single agent
A single LLM agent that answers underwriting questions by calling real tools
instead of guessing: a DSCR calculator, an EMI/interest calculator, and two
"mock API" lookups backed by the CSVs the data generator produced.

Concept: the model doesn't compute anything itself. It reads the user's
question, decides which tool(s) it needs, we run the actual Python function,
feed the result back, and the model repeats until it has enough to answer.
That loop is what "agent" means here -- not a new LLM feature, just a loop
around ordinary function calling.

Matches the course lab's structured-tool-calling pattern (Lab A): tools are
plain Python functions wrapped with LangChain's `tool()`, the model is a
`ChatLiteLLM` with `.bind_tools()`, and the loop reads `ai_msg.tool_calls` /
appends `ToolMessage` objects -- rather than hand-writing JSON tool schemas
and calling `litellm.completion(tools=...)` directly. The four underlying
functions stay plain, undecorated Python functions (M5/M6/M7/M8 import and
call them directly), and are wrapped separately for this agent loop.
"""

import os, csv, json, time
from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

load_dotenv()
MODEL = "groq/openai/gpt-oss-120b" if os.getenv("GROQ_API_KEY") else f"openrouter/{os.getenv('OPENROUTER_MODEL')}"

DATA_DIR = "data/wealthpilot/mock_api"


# =====================================================================
# Load the mock tables once, index by applicant_id for fast lookup
# =====================================================================
def load_csv(name):
    with open(f"{DATA_DIR}/{name}.csv") as f:
        return list(csv.DictReader(f))

BUREAU = {row["applicant_id"]: row for row in load_csv("bureau_reports")}

STATEMENTS_BY_APPLICANT = {}
for row in load_csv("bank_statements"):
    STATEMENTS_BY_APPLICANT.setdefault(row["applicant_id"], []).append(row)


# =====================================================================
# The 4 tools. Each is a plain Python function -- the LLM never runs
# code, it only ever gets to pick one of these and supply arguments.
# =====================================================================
def dscr_calculator(ebitda_inr: float, existing_annual_debt_service_inr: float, new_annual_debt_service_inr: float) -> dict:
    """Debt-Service Coverage Ratio = EBITDA / total annual debt service."""
    total_debt_service = existing_annual_debt_service_inr + new_annual_debt_service_inr
    if total_debt_service <= 0:
        return {"dscr": None, "error": "total debt service must be > 0"}
    dscr = round(ebitda_inr / total_debt_service, 2)
    return {"dscr": dscr, "meets_1.25x_floor": dscr >= 1.25}


def emi_calculator(principal_inr: float, annual_rate_pct: float, tenor_months: int) -> dict:
    """Standard reducing-balance EMI formula."""
    r = annual_rate_pct / 12 / 100
    if r == 0:
        emi = principal_inr / tenor_months
    else:
        emi = principal_inr * r * (1 + r) ** tenor_months / ((1 + r) ** tenor_months - 1)
    return {
        "monthly_emi_inr": round(emi, 2),
        "annual_debt_service_inr": round(emi * 12, 2),
        "total_repayment_inr": round(emi * tenor_months, 2),
    }


def mock_credit_bureau_api(applicant_id: str) -> dict:
    """Look up an applicant's bureau report."""
    row = BUREAU.get(applicant_id)
    if not row:
        return {"error": f"no bureau record for {applicant_id}"}
    return row


def mock_bank_statement_api(applicant_id: str) -> dict:
    """Aggregate an applicant's bank statement history into a summary."""
    rows = STATEMENTS_BY_APPLICANT.get(applicant_id)
    if not rows:
        return {"error": f"no bank statements for {applicant_id}"}
    n = len(rows)
    return {
        "months_of_history": n,
        "avg_monthly_inflow_inr": round(sum(float(r["inflow_inr"]) for r in rows) / n, 2),
        "avg_monthly_outflow_inr": round(sum(float(r["outflow_inr"]) for r in rows) / n, 2),
        "avg_closing_balance_inr": round(sum(float(r["closing_balance_inr"]) for r in rows) / n, 2),
        "total_bounced_instruments": sum(int(r["bounced_instruments"]) for r in rows),
    }


TOOL_FUNCTIONS = {
    "dscr_calculator": dscr_calculator,
    "emi_calculator": emi_calculator,
    "mock_credit_bureau_api": mock_credit_bureau_api,
    "mock_bank_statement_api": mock_bank_statement_api,
}

# =====================================================================
# LangChain-wrapped tools -- `tool()` reads each function's type hints and
# docstring to build its JSON schema automatically (no hand-written schema
# dict, unlike a raw litellm.completion(tools=[...]) call). Wrapping the
# plain functions here (rather than decorating them at definition) keeps
# dscr_calculator etc. callable as ordinary Python functions elsewhere.
# =====================================================================
dscr_calculator_tool = tool(dscr_calculator)
emi_calculator_tool = tool(emi_calculator)
mock_credit_bureau_api_tool = tool(mock_credit_bureau_api)
mock_bank_statement_api_tool = tool(mock_bank_statement_api)

TOOLS = [dscr_calculator_tool, emi_calculator_tool, mock_credit_bureau_api_tool, mock_bank_statement_api_tool]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """You are a credit risk analyst at Ashva Capital, a digital SME lender.
Given a loan question, use the available tools to gather real numbers -- bureau
report, bank statement summary, DSCR, EMI -- before giving a risk assessment.
Never guess a number a tool can give you. State your reasoning and cite the
actual figures you retrieved. You only recommend; you never approve loans yourself."""

llm = ChatLiteLLM(model=MODEL, max_retries=5)


# =====================================================================
# The agent loop: call the model, execute any tool calls it requests,
# feed results back, repeat until it answers in plain text. This is
# LangChain's bind_tools()/tool_calls/ToolMessage idiom, not a manual
# JSON tool-schema + litellm.completion(tools=...) loop.
# =====================================================================
def run_agent(user_question: str, max_turns: int = 6) -> str:
    model_with_tools = llm.bind_tools(TOOLS)
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_question)]

    for _ in range(max_turns):
        # Groq's generation occasionally corrupts a tool name with a stray
        # zero-width character, which the provider's own validation then
        # rejects -- a transient model glitch, not a logic bug, so retry.
        for attempt in range(3):
            try:
                ai_msg = model_with_tools.invoke(messages)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  (tool-call generation glitched, retrying: {e})")
                time.sleep(1.5)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            return ai_msg.content

        for tc in ai_msg.tool_calls:
            print(f"  [tool call] {tc['name']}({tc['args']})")
            result = TOOLS_BY_NAME[tc["name"]].invoke(tc["args"])
            print(f"  [tool result] {result}")
            messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tc["id"]))

    return "Agent did not converge to a final answer within max_turns."


if __name__ == "__main__":
    # In the real pipeline, EBITDA/existing debt come from M1's parsed
    # LoanApplication -- here we pass them in directly to demo M2 standalone.
    question = (
        "Applicant ASH-A-00001 is requesting a new loan of 2000000 INR at "
        "14% annual interest over 36 months. Their EBITDA is 16713441 INR "
        "and their existing annual debt service is 3411470 INR. Check their "
        "bureau report and bank statement history, compute DSCR including the "
        "new loan's debt service, and give a recommendation."
    )
    print(f"QUESTION: {question}\n")
    answer = run_agent(question)
    print(f"\n=== FINAL ANSWER ===\n{answer}")
