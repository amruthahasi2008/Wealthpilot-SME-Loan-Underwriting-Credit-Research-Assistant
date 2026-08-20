"""
WEALTHPILOT -- M3: persistent memory + semantic policy index
=============================================================================
Two separate kinds of "memory" a real underwriting system needs:

  A. Persistent memory of applicant interactions -- so the agent remembers
     past decisions across sessions, not just within one conversation. Uses
     Supermemory (a hosted memory API), matching the course lab exactly:
     remember()/recall() wrap mem.add()/search.memories(), namespaced by a
     container_tag, with metadata["type"] tagging each memory's CoALA kind
     (episodic/semantic/procedural).
  B. A semantic index over the credit policy corpus -- so the agent can
     look up what policy actually says instead of guessing. This is a
     separate concern from Supermemory's user-fact memory (the lab's own
     "semantic memory" means stable facts about a person, not a document
     corpus) -- kept here since M4's retrieval pipeline depends on it, but
     built on a local Gemini-embedding cache rather than Supermemory.
"""

import os, csv, json, glob, re, hashlib, time
from dotenv import load_dotenv
import numpy as np
import litellm
import supermemory

load_dotenv()
EMBED_MODEL = "gemini/gemini-embedding-001"
CACHE_DIR = ".embcache"
os.makedirs(CACHE_DIR, exist_ok=True)


def embed(text: str) -> list:
    """Embed one string, disk-cached by content hash so re-runs cost nothing."""
    key = hashlib.sha1((EMBED_MODEL + "::" + text).encode()).hexdigest()
    path = f"{CACHE_DIR}/{key}.json"
    if os.path.exists(path):
        return json.load(open(path))
    vec = litellm.embedding(model=EMBED_MODEL, input=[text]).data[0]["embedding"]
    json.dump(vec, open(path, "w"))
    return vec


def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# =====================================================================
# Part A -- persistent applicant memory, via Supermemory (matches the lab)
# =====================================================================
mem = supermemory.Supermemory(api_key=os.getenv("SUPERMEMORY_API_KEY"))
USER_ID = "wealthpilot-ashva-capital"  # Supermemory container_tag -- this project's memory namespace


def remember(text: str, kind: str, **extra):
    """Write a memory into Supermemory. `kind` is the CoALA type (episodic/
    semantic/procedural); `**extra` (e.g. applicant_id=...) is stored as
    additional metadata for later filtering."""
    return mem.add(content=text, container_tag=USER_ID, metadata={"type": kind, **extra})


def recall(query: str, client=None, k: int = 3, applicant_id: str = None):
    """Recall by semantic similarity. Tries the extracted-memories layer
    first (readable `.memory` text); falls back to the raw-documents layer
    (`.chunks`) only if that's empty -- same two-layer lookup as the lab.
    `applicant_id` is a WealthPilot-specific post-filter on top of the
    lab's base signature, since a real underwriter needs per-applicant
    recall, not just a global namespace search."""
    client = client or mem
    out = []
    for r in client.search.memories(q=query, container_tag=USER_ID, limit=k).results:
        if getattr(r, "memory", None):
            meta = r.metadata or {}
            if applicant_id and meta.get("applicant_id") != applicant_id:
                continue
            out.append((meta.get("type", "?"), r.memory, float(r.similarity or 0.0)))
    if not out:
        for r in client.search.documents(q=query, container_tags=[USER_ID], limit=k).results:
            meta = r.metadata or {}
            if applicant_id and meta.get("applicant_id") != applicant_id:
                continue
            text = " ".join(c.content for c in (r.chunks or []) if c and c.content).strip()
            if text:
                out.append((meta.get("type", "?"), text, float(r.score or 0.0)))
    return out


def seed_memory_from_past_decisions(limit=30):
    # Supermemory persists across runs -- unlike the earlier local-file store,
    # re-running this with no guard would duplicate the seed set every time.
    if mem.search.memories(q="past loan decision", container_tag=USER_ID, limit=1).results:
        print("  memory already seeded, skipping (Supermemory persists across runs)")
        return
    with open("data/wealthpilot/mock_api/past_decisions.csv") as f:
        rows = list(csv.DictReader(f))[:limit]
    for r in rows:
        text = (f"On {r['decided_on']}, applicant {r['applicant_id']} was {r['outcome']} "
                f"for a loan (internal grade {r['internal_grade']}, approved amount "
                f"{r['approved_amount_inr']} INR, reason code {r['reason_code']}).")
        remember(text, "episodic", applicant_id=r["applicant_id"], date=r["decided_on"])

    # Supermemory indexes asynchronously -- poll until the first write is
    # searchable (up to ~60s) rather than guessing a fixed delay.
    print("  waiting for Supermemory to index the seeded memories...")
    for _ in range(20):
        if mem.search.memories(q=rows[0]["applicant_id"], container_tag=USER_ID, limit=1).results:
            break
        time.sleep(3)
    print(f"  seeded {len(rows)} episodic memories from past_decisions.csv")


# =====================================================================
# Part B -- semantic index over the credit policy corpus
# =====================================================================
def load_and_chunk_corpus():
    chunks = []
    for path in sorted(glob.glob("data/wealthpilot/corpus/markdown/*.md")):
        text = open(path).read()
        source = os.path.basename(path)
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        buf = ""
        for p in paras:
            buf += p + "\n\n"
            if len(buf) > 500:
                chunks.append({"source": source, "text": buf.strip()})
                buf = ""
        if buf.strip():
            chunks.append({"source": source, "text": buf.strip()})
    return chunks


_INDEX_CACHE = "data/wealthpilot/policy_index.json"


def build_policy_index():
    if os.path.exists(_INDEX_CACHE):
        return json.load(open(_INDEX_CACHE))
    chunks = load_and_chunk_corpus()
    print(f"  embedding {len(chunks)} policy chunks (cached after first run)...")
    for c in chunks:
        c["embedding"] = embed(c["text"])
    json.dump(chunks, open(_INDEX_CACHE, "w"))
    return chunks


def dense_search(query: str, k=3):
    index = build_policy_index()
    q = embed(query)
    scored = sorted(index, key=lambda c: cosine(q, c["embedding"]), reverse=True)
    return [(c["source"], c["text"]) for c in scored[:k]]


if __name__ == "__main__":
    print("=== Part A: persistent applicant memory ===")
    seed_memory_from_past_decisions()

    sample_id = list(csv.DictReader(open("data/wealthpilot/mock_api/past_decisions.csv")))[0]["applicant_id"]
    print(f"\nEpisodic recall for {sample_id}:")
    for kind, text, score in recall("what was the outcome of this applicant's past loan application?", applicant_id=sample_id):
        print(f"  {score:.3f}  [{kind}] {text}")

    print("\n=== Part B: semantic index over policy corpus ===")
    index = build_policy_index()
    print(f"  indexed {len(index)} chunks from {len(set(c['source'] for c in index))} documents")

    print("\nPolicy search: 'what DSCR floor applies to the textiles sector?'")
    for source, text in dense_search("what DSCR floor applies to the textiles sector?"):
        print(f"  [{source}] {text[:150]}...")

    print("\nPolicy search: 'what approval is required for a loan above 50 lakh rupees?'")
    for source, text in dense_search("what approval is required for a loan above 50 lakh rupees?"):
        print(f"  [{source}] {text[:150]}...")
