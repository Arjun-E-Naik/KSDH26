import pathway as pw
import re
from pathway.xpacks.llm.vector_store import VectorStoreServer
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder

DATA_DIR = "./data/"
HOST = "0.0.0.0"
PORT = 8000

# ============================================================
# 1. ADVANCED SIGNAL EXTRACTION (DETERMINISTIC)
# ============================================================

ACTION_VERBS = [
    "run", "ran", "sprinted", "walk", "swim", "dived", "fight", "fought",
    "climb", "shoot", "shot", "grab", "carry", "lift", "escape", "attack",
    "kill", "murder", "died", "born"
]

EMOTION_WORDS = [
    "fear", "afraid", "terror", "panic", "calm", "confident",
    "hesitate", "anger", "furious", "joy", "sad", "wept", "cried",
    "happy", "love", "hate"
]

def extract_temporal_signals(text):
    """
    CRITICAL FOR TRACK A: Extracts years (e.g., 1865) to help Agent 
    check chronological consistency.
    """
    # Regex for years 1000-2999
    years = re.findall(r'\b(1\d{3}|20\d{2})\b', text)
    if years:
        return f"Timeline: {', '.join(sorted(set(years)))}"
    return ""

def extract_semantic_signals(text):
    """
    Extracts behavioral keywords to boost 'Thematic Search'.
    """
    text_l = text.lower()
    actions = [v for v in ACTION_VERBS if v in text_l]
    emotions = [e for e in EMOTION_WORDS if e in text_l]
    
    tags = []
    if actions: tags.append(f"Actions: {', '.join(set(actions))}")
    if emotions: tags.append(f"Emotions: {', '.join(set(emotions))}")
    
    return " | ".join(tags)

# ============================================================
# 2. SMART "AUGMENTED" CHUNKER
# ============================================================

def augmented_chapter_splitter(text):
    """
    Splits text but INJECTS metadata into the chunk content.
    This ensures the embedding captures both the 'Concept' and the 'Story'.
    """
    # 1. Split by double newlines (Paragraphs)
    raw_paragraphs = text.split("\n\n")
    
    enriched_chunks = []
    current_chapter = "General Context"
    buffer_text = ""
    
    for para in raw_paragraphs:
        para = para.strip()
        if not para: continue
        
        # Chapter Detection
        if "CHAPTER" in para.upper()[:40]:
            current_chapter = para.split("\n")[0].strip()
            continue
            
        # 2. Smart Merging: Don't drop small lines (dialogue is crucial!)
        # Accumulate until we have a decent chunk size (~300 chars)
        buffer_text += "\n" + para
        
        if len(buffer_text) >= 400:
            # 3. ENRICHMENT STEP
            time_tag = extract_temporal_signals(buffer_text)
            semantic_tag = extract_semantic_signals(buffer_text)
            
            # Construct the "Super Chunk"
            # Format: [META] Raw Text
            header_parts = [f"SOURCE: {current_chapter}"]
            if time_tag: header_parts.append(time_tag)
            if semantic_tag: header_parts.append(semantic_tag)
            
            header = f"[{' | '.join(header_parts)}]"
            
            # The Final Chunk has the Context HEADER + The STORY
            final_chunk = f"{header}\n{buffer_text.strip()}"
            enriched_chunks.append(final_chunk)
            
            # Reset buffer with slight overlap (keep last 50 chars for continuity)
            buffer_text = buffer_text[-50:] 
            
    # Flush remaining buffer
    if len(buffer_text) > 50:
        final_chunk = f"[{current_chapter} | END_FRAGMENT]\n{buffer_text.strip()}"
        enriched_chunks.append(final_chunk)

    return enriched_chunks

def extract_book_name(data, metadata):
    return metadata["path"].split("/")[-1].replace(".txt", "")

# ============================================================
# 3. PATHWAY PIPELINE
# ============================================================

def run():
    # 1. Read
    files = pw.io.fs.read(
        DATA_DIR,
        format="plaintext",
        mode="streaming",
        with_metadata=True
    )

    # 2. Transform (Augment)
    documents = (
        files.select(
            book_name=pw.apply(extract_book_name, pw.this.data, pw.this._metadata),
            # Apply the new Augmented Splitter
            chunks=pw.apply(augmented_chapter_splitter, pw.this.data),
            _metadata=pw.this._metadata
        )
        .flatten(pw.this.chunks)
        .select(
            data=pw.this.chunks,
            book_name=pw.this.book_name,
            _metadata=pw.this._metadata
        )
    )

    # 3. Embed
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")

    # 4. Serve
    vector_store = VectorStoreServer(
        documents,
        embedder=embedder,
        parser=None,
        splitter=None # We did the splitting manually
    )

    print(f"🚀 Augmented Vector Store running on {HOST}:{PORT}")
    
    # threaded=False is safer for Codespaces/Docker environments
    vector_store.run_server(
        host=HOST,
        port=PORT,
        threaded=False, 
        with_cache=True
    )

if __name__ == "__main__":
    run()