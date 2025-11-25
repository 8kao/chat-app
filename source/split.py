import os

def load_files(base_path):

    files = []

    for root, dirs, filenames in os.walk(base_path):
        for f in filenames:
            files.append(os.path.join(root, f))
    return files

def read_files(path):

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except:
        return ""
    
def chunk_text(text, chunk_size=500, overlap=100):
    
    # on découpe le texte en chunks avec overlap
    # chunk_size = longueur de base
    # overlap : on prend les 100 derniers chars du previous chunk pour garder le contexte entre morceaux

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

def build_chunks(base_path):

    # 1 : liste des fichiers
    # 2 : lecture
    # 3 : chunking
    # return une liste : [{"source": file, "content": chunk}, ...]

    all_files = load_files(base_path)
    all_chunks = []

    for path in all_files:
        text = read_files(path)
        if not text.strip():
            continue
        chunks = chunk_text(text)

        for c in chunks:
            all_chunks.append({"source": path, "content": c})

    return all_chunks 
