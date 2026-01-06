# src/utils.py
def smart_chapter_splitter(text):
    """
    Splits text into chunks but preserves context (Chapter headers).
    """
    # Simple split by double newlines (paragraphs)
    # In production, use a more robust regex for "Chapter X"
    raw_chunks = text.split("\n\n")
    
    enriched_chunks = []
    current_chapter = "General Context"
    
    for chunk in raw_chunks:
        # Update context if we see a chapter header
        if "CHAPTER" in chunk.upper()[:20]: 
            current_chapter = chunk.strip()
        
        if len(chunk) > 50:  # Ignore tiny artifacts
            # Prepend context so the vector store knows WHERE this happened
            enriched_chunks.append(f"[{current_chapter}] {chunk}")
            
    return enriched_chunks