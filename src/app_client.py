import os
import time
import json
import requests
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# ============================================================
# 1. SETUP & CONFIGURATION
# ============================================================

load_dotenv()

# Use 'instant' for fast queries, 'versatile' for deep logic
MODEL_FAST = "llama-3.1-8b-instant"
MODEL_SMART = "llama-3.3-70b-versatile"

PATHWAY_URL = "http://127.0.0.1:8000/v1/retrieve"
GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))

INPUT_CSV = "train.csv"
OUTPUT_CSV = "results.csv"

TOP_K = 3
SLEEP_TIME = 0.5

# ============================================================
# 2. PATHWAY RETRIEVAL (ROBUST)
# ============================================================

def get_evidence(book_name, query):
    """
    Fetches chunks from Pathway. Returns empty string on failure.
    """
    try:
        r = requests.post(
            PATHWAY_URL,
            json={
                "query": query,
                "k": TOP_K,
                "filters": {"book_name": book_name}
            },
            timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            if not data: return ""
            # Join with a clear separator so the LLM knows they are fragments
            return "\n[TEXT FRAGMENT]\n".join([x["text"] for x in data])
        return ""
    except Exception as e:
        print(f"⚠️ Retrieval Error: {e}")
        return ""

# ============================================================
# 3. AGENT 0 — CONSTRAINT EXTRACTOR
# ============================================================

def agent_constraint_extractor(char, claim):
    """
    Breaks the backstory into atomic facts (Time, Location, Status).
    """
    system_prompt = """
    You are a Narrative Constraint Extractor.
    Task: Convert a character backstory claim into explicit constraints.
    
    Focus on:
    1. Physical Status (Alive/Dead, Imprisoned, Disabled).
    2. Location/Time (Was he in Paris in 1815?).
    3. Key Relationships (Did he know Character X?).
    
    Output JSON:
    { "constraints": [ { "type": "Physical/Temporal", "description": "..." } ] }
    """
    user_prompt = f"Character: {char}\nBackstory Claim: '{claim}'"

    try:
        r = GROQ_CLIENT.chat.completions.create(
            model=MODEL_FAST,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        return json.loads(r.choices[0].message.content)
    except:
        # Fallback if JSON fails
        return {"constraints": [{"type": "General", "description": claim}]}

# ============================================================
# 4. AGENT 1 — ADVERSARIAL SEARCH (PROSECUTOR & DEFENDER)
# ============================================================

def agent_generate_queries(constraints):
    """
    Generates two queries: One to prove it, one to disprove it.
    """
    system_prompt = """
    You are a Legal Research Strategist.
    Input: A list of narrative constraints.
    Task: Generate TWO search queries.
    
    1. Prosecutor Query: Look for contradictions (e.g., if constraint says 'Alive', search for 'Death', 'Funeral').
    2. Defender Query: Look for support (e.g., search for the specific event mentioned).
    
    Output: Return ONLY the two queries separated by '|||'.
    """
    
    user_prompt = f"Constraints:\n{json.dumps(constraints)}"

    try:
        r = GROQ_CLIENT.chat.completions.create(
            model=MODEL_FAST,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.3
        )
        content = r.choices[0].message.content.strip()
        if "|||" in content:
            return content.split("|||")
        return [content, content] # Fallback
    except:
        return ["character history", "character contradictions"]

# ============================================================
# 6. AGENT 2 — THE JUDGE (WITH CONTEXT LIMITS)
# ============================================================

def agent_judge(claim, constraints, evidence_pro, evidence_def):
    """
    Decides Consistency based on evidence.
    """
    # CRITICAL: Prevent Token Overflow by hard truncation
    combined_evidence = f"PROSECUTION:\n{evidence_pro}\n\nDEFENSE:\n{evidence_def}"
    if len(combined_evidence) > 4500:
        combined_evidence = combined_evidence[:4500] + "\n...[TRUNCATED]"

    system_prompt = """
    You are a Narrative Consistency Judge.
    
    DECISION LOGIC:
    1. **Direct Contradiction (Score 0):** - If the text explicitly contradicts a Physical or Temporal constraint (e.g., Claim says "In Paris", Text says "In Prison"), Verdict is INCONSISTENT.
    
    2. **Argument from Silence (Score 0 vs 1):**
       - If a MAJOR CANON EVENT (Death, Marriage, War) is claimed but completely missing from text -> Verdict INCONSISTENT (0).
       - If a MINOR PRIVATE DETAIL (Thought, Feeling) is claimed and missing -> Verdict CONSISTENT (1).
       
    3. **Consistency (Score 1):**
       - If evidence supports the claim OR allows it to happen off-screen.
       
    Output JSON: { "prediction": 0 or 1, "reasoning": "short explanation" }
    """

    user_prompt = f"""
    CLAIM: {claim}
    CONSTRAINTS: {json.dumps(constraints)}
    EVIDENCE FROM BOOK:
    {combined_evidence}
    """

    try:
        r = GROQ_CLIENT.chat.completions.create(
            model=MODEL_SMART,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(r.choices[0].message.content)
    except:
        return {"prediction": 1, "reasoning": "Error in processing, defaulting to consistent."}

# ============================================================
# 7. AGENT 3 — RATIONALE WRITER
# ============================================================

def agent_rationale(claim, evidence, decision):
    # Map integer decision to string for the prompt context
    verdict_str = "CONSISTENT" if decision == 1 else "INCONSISTENT"
    
    if len(evidence) > 4000: evidence = evidence[:4000]

    system_prompt = f"""
    You are an Evidence Rationale Writer. 
    The Judge has ruled: {verdict_str}.
    
    Task: Write a single sentence explanation.
    - If INCONSISTENT: Quote the specific text that contradicts the claim.
    - If CONSISTENT: Mention that text supports it or does not refute it.
    
    Output JSON: {{ "rationale": "..." }}
    """

    user_prompt = f"CLAIM: {claim}\nEVIDENCE: {evidence}"

    try:
        r = GROQ_CLIENT.chat.completions.create(
            model=MODEL_SMART,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        return json.loads(r.choices[0].message.content)
    except:
        return {"rationale": "Rationale generation failed."}

# ============================================================
# 8. MAIN PIPELINE
# ============================================================

def run_pipeline():
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_CSV} not found.")
        return

    results = []
    print(f"🚀 Running Track-A Prosecutor/Defender Pipeline on {len(df)} rows")

    for _, row in df.iterrows():
        row_id = row["id"]
        book = row["book_name"]
        char = row["char"]
        claim = row["content"]

        print(f"\n▶ Processing ID {row_id}...")

        # 1. Extract Constraints
        constraints = agent_constraint_extractor(char, claim)

        # 2. Generate Adversarial Queries
        queries = agent_generate_queries(constraints)
        q_pro = queries[0]
        q_def = queries[1] if len(queries) > 1 else queries[0]

        # 3. Retrieve Evidence
        ev_pro = get_evidence(book, q_pro)
        ev_def = get_evidence(book, q_def)

        # 4. Judge
        judge_out = agent_judge(claim, constraints, ev_pro, ev_def)
        pred_int = judge_out.get("prediction", 1) # Default to 1 (Consistent)

        # 5. Rationale
        combined_ev = ev_pro + "\n" + ev_def
        rationale_out = agent_rationale(claim, combined_ev, pred_int)
        rationale_text = rationale_out.get("rationale", judge_out.get("reasoning", "No rationale."))

        # 6. Formatting for Submission (Strings, not Integers)
        label_str = "consistent" if pred_int == 1 else "contradict"
        
        print(f"   ⚖️  Verdict: {label_str.upper()}")
        results.append([row_id, label_str, rationale_text])
        
        time.sleep(SLEEP_TIME)

    # Save
    pd.DataFrame(results, columns=["id", "label", "rationale"]).to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Finished. Saved to {OUTPUT_CSV}")

# ============================================================
# 9. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_pipeline()