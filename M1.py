import os, json, time
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
import litellm

load_dotenv()
# Prefer Groq if you've added a key (faster, less congested); OpenRouter free tier also works, just slower.
MODEL = "groq/openai/gpt-oss-120b" if os.getenv("GROQ_API_KEY") else f"openrouter/{os.getenv('OPENROUTER_MODEL')}"


def build_raw_text(record: dict) -> str:
    """Simulate the real messy intake: applicant's cover-letter narrative PLUS
    an attached financial statement -- exactly what raw_narrative alone lacks.
    Some records deliberately omit a financial line (that's the point of the
    dataset's 'missing_fields' ground truth), so use .get() everywhere."""
    f = record["declared_financials"]

    def g(key):
        return f.get(key, "not provided")

    return f"""{record['raw_narrative']}

--- Attached application form ---
Business name: {record['business_name']}
Sector: {record['sector']}
Requested loan amount (INR): {record['requested_amount_inr']}
Annual revenue (INR): {g('annual_revenue_inr')}
EBITDA (INR): {g('ebitda_inr')}
Existing debt (INR): {g('existing_debt_inr')}
Current assets (INR): {g('current_assets_inr')}
Current liabilities (INR): {g('current_liabilities_inr')}
Months operating: {g('months_operating')}
Bureau score: {record['bureau_score'] if record['bureau_score'] is not None else 'not provided'}
"""


def call_llm(**kwargs):
    """litellm.completion wrapped with backoff -- OpenRouter's free-tier models
    sit on a shared pool and 429 under normal, non-abusive load."""
    for attempt in range(5):
        try:
            return litellm.completion(model=MODEL, **kwargs)
        except litellm.exceptions.RateLimitError:
            wait = 2 ** attempt
            print(f"  (rate limited, waiting {wait}s...)")
            time.sleep(wait)
    return litellm.completion(model=MODEL, **kwargs)


class FinancialSnapshot(BaseModel):
    annual_revenue_inr: float
    ebitda_inr: float
    existing_debt_inr: float
    current_assets_inr: float
    current_liabilities_inr: float
    months_operating: int


class LoanApplication(BaseModel):
    application_id: str
    business_name: str
    sector: str
    requested_amount_inr: float
    financials: FinancialSnapshot
    bureau_score: int | None = None
    missing_fields: list[str] = []


EXTRACTION_PROMPT = """Extract loan application data from the applicant's narrative below into JSON matching exactly this shape:
{{
  "application_id": "...",
  "business_name": "...",
  "sector": "...",
  "requested_amount_inr": number,
  "financials": {{
    "annual_revenue_inr": number,
    "ebitda_inr": number,
    "existing_debt_inr": number,
    "current_assets_inr": number,
    "current_liabilities_inr": number,
    "months_operating": integer
  }},
  "bureau_score": number or null,
  "missing_fields": [list any of the above fields the narrative does not mention]
}}
Only use information actually present in the text below; if a numeric field isn't mentioned, put 0 and list it in missing_fields.
Return ONLY valid JSON, no commentary.

Applicant narrative:
{narrative}
"""


def extract_raw(narrative: str) -> str:
    resp = call_llm(
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(narrative=narrative)}],
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def parse_loan_application(record: dict, max_retries: int = 2):
    narrative = build_raw_text(record)
    raw = extract_raw(narrative)
    errors = []
    for attempt in range(max_retries + 1):
        try:
            data = json.loads(raw)
            data["application_id"] = record["application_id"]  # always trust the source record, not the LLM
            # application_id is never actually IN the text we feed the model (see build_raw_text),
            # so the model correctly notices it's absent and flags it -- but it's supplied by the
            # system above, never truly missing. Filter it out of the model's own report.
            data["missing_fields"] = [f for f in data.get("missing_fields", []) if f != "application_id"]
            app = LoanApplication(**data)
            return app, errors
        except (json.JSONDecodeError, ValidationError) as e:
            errors.append(str(e))
            if attempt == max_retries:
                return None, errors
            repair_prompt = f"""The following JSON failed validation with error: {e}

Original narrative:
{narrative}

Previous JSON attempt:
{raw}

Return corrected JSON with the exact same shape as before. Only valid JSON, no commentary."""
            resp = call_llm(
                messages=[{"role": "user", "content": repair_prompt}],
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
    return None, errors


if __name__ == "__main__":
    records = [json.loads(l) for l in open("data/wealthpilot/intake/records.jsonl")]
    results = []
    for r in records:
        app, errs = parse_loan_application(r)
        results.append((r, app, errs))
        status = "OK" if app else "FAILED"
        print(f"{r['application_id']}: {status}" + (f" ({errs[-1][:80]})" if errs and not app else ""))
        time.sleep(3)

    parsed_ok = sum(1 for _, app, _ in results if app)
    print(f"\nParsed {parsed_ok}/{len(results)} successfully.")

    print("\n--- accuracy vs declared_financials (free M1 metric) ---")
    for r, app, _ in results:
        if not app:
            continue
        declared = r["declared_financials"]
        extracted = app.financials.model_dump()
        mismatches = [k for k in declared if declared[k] != extracted.get(k)]
        print(f"{r['application_id']}: {'exact match' if not mismatches else f'mismatch on {mismatches}'}")