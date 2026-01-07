import os
import requests
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

# Fix: Explicitly tell Python where the .env file is
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Debug: Print to see if it works (Remove this later)
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ ERROR: Key still not found! Check your .env file.")
else:
    print(f"✅ Key loaded: {api_key[:5]}...")

# NOW initialize the client
GROQ_CLIENT = Groq(api_key=api_key)

# Config
PATHWAY_URL = "http://127.0.0.1:8000/v1/retrieve"


def check_consistency(backstory_claim):
    print(f"\n🔎 Analyzing Claim: '{backstory_claim}'...")

    # 1. RETRIEVE EVIDENCE FROM PATHWAY
    # We ask for the top 5 most relevant chunks from the book
    try:
        response = requests.post(
            PATHWAY_URL,
            json={"query": backstory_claim, "k": 5}
        )
        results = response.json()
        
        # Combine the retrieved text into a single context block
        evidence_text = "\n---\n".join([r['text'] for r in results])
    except Exception as e:
        print(f"Error connecting to Pathway: {e}")
        return

    # 2. REASONING WITH GROQ (Llama-3-70b is great for this)
    # We construct a prompt that forces the model to cite evidence.
    system_prompt = """
    You are a rigorous Consistency Checker for a novel. 
    Your job is to validate a 'Backstory Claim' against 'Book Excerpts'.
    
    Rules:
    1. If the claim contradicts the text, output Label: 0.
    2. If the claim fits (even loosely), output Label: 1.
    3. You MUST quote the text to support your decision.
    """

    user_prompt = f"""
    BACKSTORY CLAIM: "{backstory_claim}"
    
    EVIDENCE FROM NOVEL:
    {evidence_text}
    
    Task:
    1. Analyze the evidence.
    2. Output a JSON with: "prediction" (0 or 1) and "rationale" (string).
    """

    chat_completion = GROQ_CLIENT.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="llama-3.3-70b-versatile", # Using the large model for better reasoning
        response_format={"type": "json_object"} # Force valid JSON
    )

    print("🤖 Groq Verdict:")
    print(chat_completion.choices[0].message.content)

if __name__ == "__main__":
    # Example usage for Hackathon testing
    # In the real submission, you would loop through the input CSV file here.
    
    # Test 1: A claim that might be true
    check_consistency("Captain Grant set sail from Peru in late 1863, hoping to establish a new colony before his disappearance.")
    
    # Test 2: A claim that is definitely false
    check_consistency("Lady Helena was a timid, fragile woman who detested the sea and refused to look at anything violent or exciting.")