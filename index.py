import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

###########################################################

CHUNKS_FILE = Path("chunks/chunks.jsonl")   # output of ingest.py
DB_DIR = "chroma_db"                        # Chroma stores its files here
COLLECTION = "papers"                       # name of our table inside Chroma
MODEL_NAME = "all-MiniLM-L6-v2"             # 384-number vectors, fast and small

###########################################################

def loadChunks(pth):
    """Read one JSON record per line."""
    return [json.loads(line) for line in pth.read_text(encoding="utf-8").splitlines()]

def buildIndex():
    """Embed every chunk and store it in a fresh chroma collection. """
    chunks = loadChunks(CHUNKS_FILE)
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode([c["text"] for c in chunks],show_progress_bar=True)

    client = chromadb.PersistentClient(path=DB_DIR)           # saved on disk, survives restarts

    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)                   # rebuild from scratch each run
        
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=vectors.tolist(),
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
    )
    
    print(collection.count(), "chunks indexed")

if __name__ == "__main__":
    buildIndex()