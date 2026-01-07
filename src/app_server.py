import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
from utils import smart_chapter_splitter

DATA_DIR = "./data/"
HOST = "0.0.0.0"
PORT = 8000

def run():
    # 1. Ingest Data
    files = pw.io.fs.read(
        DATA_DIR,
        format="plaintext",
        mode="streaming",
        with_metadata=True
    )

    # 2. Transform (Chunking & Renaming)
    # We rename 'chunks' -> 'data' (required by VectorStore)
    # We keep '_metadata' -> '_metadata' (required for filtering to work without warnings)
    documents = files.select(
        chunks=pw.apply(smart_chapter_splitter, pw.this.data),
        metadata=pw.this._metadata
    ).flatten(pw.this.chunks).select(
        data=pw.this.chunks,
        _metadata=pw.this.metadata  # Fixes the "Filtering will not work" warning
    )

    # 3. Embedding
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")

    # 4. Vector Store
    vector_store = VectorStoreServer(
        documents,
        embedder=embedder,
        splitter=None,
    )

    print(f"✅ Vector Store running on {HOST}:{PORT}")
    
    # CRITICAL FIX: threaded=False ensures the script stays alive!
    vector_store.run_server(
        host=HOST, 
        port=PORT, 
        threaded=False,  # <--- CHANGED FROM True TO False
        with_cache=True
    )

if __name__ == "__main__":
    run()