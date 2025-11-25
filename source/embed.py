import chromadb
from split import build_chunks
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="../data/vector_db")

def embed_text(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding

def build_vector_db(chunks, batch_size=50):

    # on crée la base vectorielle dans chromadb
    # on remplit le contenu des listes
    # puis embedding en batch de 50 parce que sinon c'est trop long d'envoyer les chunks un par un à openai

    collection = chroma_client.get_or_create_collection(
        name="corpus_collection",
        metadata={"hnsw:space": "cosine"}
    )

    ids = []
    texts = []
    metadatas = []

    for i, ch in enumerate(chunks):
        ids.append(f"chunk_{i}")
        texts.append(ch["content"])
        metadatas.append({"source": ch["source"]})

    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Batch {i} -> {i + len(batch)}")

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )

        for item in response.data:
            embeddings.append(item.embedding)

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print("base vectorielle créée!!!!!")

if __name__ == "__main__":
    chunks = build_chunks("data/raw_docs")
    build_vector_db(chunks)
