import os
import pandas as pd
import requests
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# CONFIG
PATHWAY_URL = "http://127.0.0.1:8000/v1/retrieve"
GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))
INPUT_CSV = "input.csv"
OUTPUT_CSV = "results.csv"

def get_evidence(book_name, query):
    """
    AGENT 1 TOOL: SCOPED RETRIEVAL
    Only searches within the specific book to avoid pollution.
    """
    response = requests.post(
        PATHWAY_URL,
        json={
            "query": query,
            "k": 5,
            # THIS IS THE KEY: Filter by the 'book_name' column we created in server.py
            "filters": {"book_name": book_name} 
        }
    )
    if response.status_code == 200:
        results = response.json()
        return "\n---\n".join([r['text'] for r in results])
    return ""

def run_pipeline():
    # 1. Load Data
    df = pd.read_csv(INPUT_CSV)
    results = []

    print(f"🚀 Starting Multi-Agent Pipeline on {len(df)} rows...")

    for index, row in df.iterrows():
        book = row['book_name']
        char = row['char']
        claim = row['content']

        if pd.notna(row.get('caption')) and str(row['caption']).strip() != "":
            caption_context = f"({row['caption']})"
        else:
            caption_context = ""        
        
        # Refined Search Query
        search_query = f"{char} {caption_context}: {claim}"
        
        print(f"\nProcessing ID {row['id']}...")
        print(f"🔎 Querying: {search_query}")

        # --- AGENT 1: FACT CHECKER ---
        # "How this is helpful": It combines Character + Content for a precise query
        # and filters strictly by the book name.
        
        evidence_text = get_evidence(book, search_query)
        
        if not evidence_text:
            print("⚠️ No evidence found (check filenames). Defaulting to Inconsistent.")
            results.append([row['id'], 0, "No evidence found in text."])
            continue

        # --- AGENT 2 & 3: PSYCHOLOGIST & JUDGE (Combined for efficiency) ---
        system_prompt = """
        You are an AI Judge for a literary consistency contest.
        You have two sub-personas:
        1. Fact Checker: Checks dates, names, and hard events.
        2. Psychologist: Checks character voice, personality, and motivation.
        
        Input:
        - Backstory Claim (A new history proposed for a character)
        - Evidence (Actual excerpts from the book)
        - Sometimes You get the caption or sometimes it is None, which helps you understand the context better.
        
        Task:
        Determine if the Backstory is CONSISTENT (1) or CONTRADICTORY (0) with the Evidence.
        
        Rules:
        - If the text explicitly contradicts the claim (e.g. claim says "orphan", text mentions "father"), label 0.
        - If the claim fits the character's vibe and has no hard contradictions, label 1.
        - Output JSON: {"prediction": 0 or 1, "rationale": "Quote from text + explanation"}
        """

        user_prompt = f"""
        BOOK: {book}
        CHARACTER: {char}
        BACKSTORY CLAIM: "{claim}"
        CAPTION CONTEXT: "{caption_context}"
        
        RETRIEVED EVIDENCE FROM BOOK:
        {evidence_text}
        """

        try:
            completion = GROQ_CLIENT.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"}
            )
            
            response_json = json.loads(completion.choices[0].message.content)
            pred = response_json.get("prediction", 0)
            rationale = response_json.get("rationale", "Rationale generation failed.")
            
            print(f"✅ Verdict: {pred} | {rationale[:50]}...")
            results.append([row['id'], pred, rationale])

        except Exception as e:
            print(f"❌ Error: {e}")
            results.append([row['id'], 0, "Error in processing"])

    # 3. Save Results
    output_df = pd.DataFrame(results, columns=["id", "label", "rationale"])
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n🎉 Done! Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_pipeline()