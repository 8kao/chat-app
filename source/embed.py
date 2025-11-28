import chromadb
from split import build_chunks
import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="../data/vector_db")

def embed_text(text):
    # text-embedding-3-small produces 1,536 dimensions by default,
    # this is why we set the dimensions to 512 for faster search and retrieval
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        dimensions=512
    )

    return response.data[0].embedding

def build_vector_db(chunks, batch_size=50):

    # create the vector database in chromadb
    # fill the content of the lists
    # then embed in batches of 50 because otherwise it takes too long to send chunks one by one to openai

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

    max_batch = 5461
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if (i + len(batch)) > max_batch:
            print("Corpus too large for ChromaDB. Please try again with a smaller corpus")
            return
        print(f"Batch {i} -> {i + len(batch)}")
        
        # same dimensions as the ones used for the embeddings, otherwise it will not work
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
            dimensions=512
        )

        for item in response.data:
            embeddings.append(item.embedding)

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print("Vector database created!!!!!")

if __name__ == "__main__":
    chunks = build_chunks("data/docs_corpus")
    build_vector_db(chunks)
