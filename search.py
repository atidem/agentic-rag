
import chromadb
from sentence_transformers import SentenceTransformer

###########################################################

DB_DIR =  "chroma_db"
COLLECTION = "papers"
MODEL_NAME = "all-MiniLM-L6-v2" 

###########################################################

model = SentenceTransformer(MODEL_NAME)
collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)

###########################################################

def search(query,k=5):
    """Return K chunks closest in meaning to the query."""
    vector = model.encode(query).tolist()
    res = collection.query(query_embeddings=[vector],n_results=k)
    return [
        {"id":i,"score":round(1-d,3),"source": m["source"], "page":m["page"],"section": m["section"],"text":t}
        for i,d,m,t in zip(res["ids"][0],res["distances"][0],res["metadatas"][0],res["documents"][0])
    ]

if __name__ == "__main__":
    query = input("Question: ")
    for r in search(query):
        print(f"\n[{r['score']}] {r['source']} p.{r['page']} · {r['section']} ({r['id']})")
        print(r["text"][:300].replace("\n", " "))