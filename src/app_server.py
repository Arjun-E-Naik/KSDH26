# src/app_server.py
import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
from utils import smart_chapter_splitter

# 1. Config
DATA_DIR = "./data/"
HOST = "0.0.0.0"
PORT = 8000

def run():
    # 2. Ingestion (Real-time watching of the folder)
    files = pw.io.fs.read(
        DATA_DIR,
        format="plaintext",
        mode="streaming",
        with_metadata=True
    )

    # 3. Transformation (Apply smart chunking)
    documents = files.select(
        chunks=pw.apply(smart_chapter_splitter, pw.this.data),
        metadata=pw.this._metadata
    ).flatten(pw.this.chunks)

    # 4. Embedding (Using local CPU embedding to save costs/latency)
    # 'all-MiniLM-L6-v2' is fast and standard for RAG
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")

    # 5. Launch Vector Store
    vector_store = VectorStoreServer(
        *documents,
        embedder=embedder,
        splitter=None, # We already split it manually
    )

    print(f"✅ Pathway Vector Store running on {HOST}:{PORT}")
    vector_store.run_server(host=HOST, port=PORT, threaded=True, with_cache=True)

if __name__ == "__main__":
    run()