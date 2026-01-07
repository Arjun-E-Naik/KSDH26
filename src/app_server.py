import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder

DATA_DIR = "./data/"
HOST = "0.0.0.0"
PORT = 8000

def smart_chapter_splitter(text):
    """
    Splits text into chunks but prepends the Chapter Header to each chunk.
    This ensures the LLM knows the 'Time/Place' of every event.
    """
    # Split by double newlines to get paragraphs
    raw_chunks = text.split("\n\n")
    
    enriched_chunks = []
    current_chapter = "General Context"
    
    for chunk in raw_chunks:
        # Detect Chapter Headers (e.g., "CHAPTER V", "Chapter 5")
        if "CHAPTER" in chunk.upper()[:20]: 
            current_chapter = chunk.strip()
        
        # Only keep chunks with actual content (>50 chars)
        if len(chunk) > 50:
            # Result: "[CHAPTER 1] The shark swam..." 
            enriched_chunks.append(f"[{current_chapter}] {chunk}")
            
    return enriched_chunks

def add_book_metadata(data, metadata):
    """
    Extracts 'book_name' from the filename so we can filter searches.
    """
    filename = metadata["path"].split("/")[-1]
    book_name = filename.replace(".txt", "")
    return book_name

def run():
    # 1. Read Files
    files = pw.io.fs.read(DATA_DIR, format="plaintext", mode="streaming", with_metadata=True)

    # 2. Transform: Add Metadata -> Smart Split -> Flatten
    documents = files.select(
        # Extract book name first
        book_name=pw.apply(add_book_metadata, pw.this.data, pw.this._metadata),
        # Apply your smart splitter
        chunks=pw.apply(smart_chapter_splitter, pw.this.data),
        # Keep original metadata
        _metadata=pw.this._metadata
    ).flatten(pw.this.chunks).select(
        # Rename for the Vector Store
        data=pw.this.chunks,
        book_name=pw.this.book_name,
        _metadata=pw.this._metadata
    )

    # 3. Embed
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")

    # 4. Vector Store
    # : splitter=None because we already split it manually above!
    vector_store = VectorStoreServer(
        documents,
        embedder=embedder,
        parser=None, 
        splitter=None 
    )

    print(f"✅ Vector Store running on {HOST}:{PORT}")
    vector_store.run_server(host=HOST, port=PORT, threaded=False, with_cache=True)

if __name__ == "__main__":
    run()