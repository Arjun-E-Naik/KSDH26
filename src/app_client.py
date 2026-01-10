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

MODEL_0 = "qwen2.5-7b-instruct"
MODEL_1 = "llama-3.1-8b-instant"
MODEL_2 = "llama-3.3-70b-versatile"

PATHWAY_URL = "http://127.0.0.1:8000/v1/retrieve"
GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))

INPUT_CSV = "test.csv"
OUTPUT_CSV = "results.csv"

TOP_K = 3
SLEEP_TIME = 0.5

# ============================================================
# 2. PATHWAY RETRIEVAL
# ============================================================

def get_evidence(book_name, query):
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
            if not data:
                return ""
            return "\n[TEXT FRAGMENT]\n".join([x["text"] for x in data])
        return ""
    except Exception as e:
        print(f"⚠️ Retrieval Error: {e}")
        return ""

# ============================================================
# 3. AGENT 0 — CONSTRAINT EXTRACTOR
# ============================================================

def agent_constraint_extractor(char, claim, caption=""):
    system_prompt = """
    You are a Narrative Constraint Extractor.
    Task: Convert a character backstory claim into explicit constraints.
    Analyse the caption and try to understand context and situation (if not None).
    Always includes character name and  in description and give better picture of the scenario.
    If multiple character names given (e.g., "Alice/Bob"), treat they are same character and one charcter has multiple names.
    description includes character name(s) and that can helps like character present in scenario.

    Focus on:
    1. Physical Status (Alive/Dead, Imprisoned, Disabled).
    2. Location/Time.
    3. Key Relationships.
    description includes all relevant details that can help judge the claim .
    Output JSON:
    { "constraints": [ { "type": "Physical/Temporal", "description": "..." } ] }
    """

    user_prompt = f"Character: {char}\nBackstory Claim: '{claim}'\nCaption: {caption}"

    try:
        r = GROQ_CLIENT.chat.completions.create(
            model=MODEL_0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        return json.loads(r.choices[0].message.content)
    except:
        merged_context = caption + " " + claim
        return {"constraints": [{"type": "General", "description": merged_context.strip()}]}

# ============================================================
# 4. AGENT 1 — QUERY GENERATOR
# ============================================================

def agent_generate_queries(char,constraints ,claim):
    system_prompt = """
    You are a Legal Research Strategist.

    Generate TWO search queries.
    The queries should contain character names and focus on finding evidence related to the constraints provided.
    If multiple character names given (e.g., "Alice/Bob"), treat they are same character and one charcter has multiple names.
    Based on scenario the character name is different (e.g., "Bruce Wayne/Batman"), include all names in the query to improve retrieval.

    Here ,Defender query contains claim's important words with character name(s) to find supportive evidence.
    1. Prosecutor Query: This would be contradict to defender query,including character name(s) and constraint details to find contradictory evidence.
    2. Defender Query: look for support

    Output: Prosecutor ||| Defender
    """

    user_prompt = f"Understand the Constraints:\n{json.dumps(constraints)} with character: {char} and claim: '{claim}'\nGenerate the two queries as per above instructions."

    try:
        r = GROQ_CLIENT.chat.completions.create(
            model=MODEL_1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        content = r.choices[0].message.content.strip()
        if "|||" in content:
            return [q.strip() for q in content.split("|||")]
        return [content, content]
    except:
        return ["character history", "character contradictions"]

# ============================================================
# 5. AGENT 2 — JUDGE
# ============================================================

def agent_judge(claim, constraints, evidence_pro, evidence_def):
    combined_evidence = f"PROSECUTION:\n{evidence_pro}\n\nDEFENSE:\n{evidence_def}"
    if len(combined_evidence) > 3500:
        combined_evidence = combined_evidence[:3500] + "\n...[TRUNCATED]"

    system_prompt = """
    You are a Narrative Consistency Judge.
    If multiple character names given (e.g., "Alice/Bob"), treat they are same character and one charcter has multiple names.
    Based on scenario the character name is different (e.g., "Bruce Wayne/Batman"), include all names in the query to improve retrieval.
    Caefully analyse and give verdict.
    
    DECISION LOGIC:
    
    1. **Direct Contradiction (Score 0):** - If the text explicitly contradicts a Physical or Temporal constraint (e.g., Claim says "In Paris", Text says "In Prison"), Verdict is INCONSISTENT.
    
    2. **Argument from Silence (Score 0 vs 1):**
       - If a MAJOR CANON EVENT (Death, Marriage, War) is claimed but completely missing from text -> Verdict INCONSISTENT (0).
       - If a MINOR PRIVATE DETAIL (Thought, Feeling) is claimed and missing -> Verdict CONSISTENT (1).
       
    3. **Consistency (Score 1):**
       - If evidence supports the claim OR allows it to happen off-screen.
    4. Analyse the evidence_pro and evidence_def separately with claim, if any of them matches or preending to give same context as claim, then will be consistent, if not matches then contradicted.
    5.Aware aboutLocation/Time like if claim taking about contry, but evidence talking about city in that contry,but incidents should matches then it is consistent(eg, Rome in Italy). 
    Output JSON: { "prediction": 0 or 1, "rationale": "Explain with quoted supporting evidence for the decision within 2 -3 lines" }
    """

    user_prompt = f"""
    CLAIM: {claim}
    CONSTRAINTS: {json.dumps(constraints)}
    EVIDENCE:
    {combined_evidence}
    """

    try:
        r = GROQ_CLIENT.chat.completions.create(
            model=MODEL_2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        print(f"LLM Error: {e}")
        return {"prediction": -1, "rationale": f"LLM Call Failed: {e}"}

# ============================================================
# 6. MAIN PIPELINE
# ============================================================

def run_pipeline():
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_CSV} not found.")
        return

    results = []
    print(f"🚀 Running pipeline on {len(df)} rows")

    for _, row in df.iterrows():
        row_id = row["id"]
        book = row["book_name"]
        char = row["char"]
        claim = row["content"]
        caption = str(row["caption"]) if pd.notna(row["caption"]) else ""

        print(f"\n▶ Processing ID {row_id}...")

        constraints = agent_constraint_extractor(char, claim, caption)

        queries = agent_generate_queries(char, constraints ,claim)

        q_pro = queries[0] if len(queries) > 0 else ""
        q_def = queries[1] if len(queries) > 1 else q_pro


        ev_pro = get_evidence(book, q_pro)
        ev_def = get_evidence(book, q_def)

        # 🔹 FIX: silence handling
        if not ev_pro and not ev_def:
            results.append([row_id, "consistent", "No contradictory evidence found in the text."])
            continue

        judge_out = agent_judge(claim, constraints, ev_pro, ev_def)

        pred_int = judge_out.get("prediction", 1)
        rationale_text = judge_out.get("rationale", "No rationale provided.")

        if pred_int == 1:
            label_str = "1"
        elif pred_int == 0:
            label_str = "0"
        else:
            label_str = "1"  # 🔹 FIX: submission-safe label

        print(f"   ⚖️ Verdict: {label_str}")
        results.append([row_id, label_str, rationale_text])

        time.sleep(SLEEP_TIME)

    pd.DataFrame(results, columns=["id", "label", "rationale"]).to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Finished. Saved to {OUTPUT_CSV}")

# ============================================================
# 7. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_pipeline()
