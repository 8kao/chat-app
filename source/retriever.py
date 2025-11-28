import chromadb
import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="../data/vector_db")
collection = chroma_client.get_collection("corpus_collection")
embedding_cache = {}

def retrieve_context(query, k=8, min_relevance=0.7):
    
    #on check si la question a déjà été posée pour eviter l'embedding inutile
    if query in embedding_cache:
        q_embed = embedding_cache[query]
    else:
        q_embed = client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
            dimensions=512
        ).data[0].embedding
        
        embedding_cache[query] = q_embed

    results = collection.query(
        query_embeddings=[q_embed],
        n_results=k * 3,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    #filtrage par score de similarité des chunks avec la question (score minimum : 0.7)
    filtered = []
    for doc, meta, dist in zip(docs, metas, dists):
        if dist <= min_relevance:
            filtered.append((doc, meta, dist))

    seen = set()
    unique = []

    #on évite de renvoyer plusieurs fois le meme chunk au LLM
    for doc, meta, dist in filtered:
        src = meta.get("source", "")
        if src not in seen:
            seen.add(src)
            unique.append((doc, meta, dist))

    unique = sorted(unique, key=lambda x: x[2])
    unique = unique[:k]

    return [
        {"content": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in unique
    ]
