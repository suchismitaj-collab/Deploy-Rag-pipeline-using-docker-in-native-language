import os
import ollama
import chromadb
from sentence_transformers import SentenceTransformer

# --- Setup ---
DATA_FILE = "biodata.txt"
CHUNK_SIZE = 500  # characters per chunk

# Multilingual embedding model (supports Bengali)
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Local vector database
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="my_notes")

# --- Step 1: Load and chunk your notes ---

def load_and_chunk():
    chunks = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    for i in range(0, len(text), CHUNK_SIZE):
        chunk = text[i:i+CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
    return chunks



# --- Step 2: Embed and store chunks (only run once / when data changes) ---
def index_notes():
    chunks = load_and_chunk()
    if not chunks:
        print("No .txt files found in data/ folder.")
        return
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    print(f"Indexed {len(chunks)} chunks from your notes.")

# --- Step 3: Retrieve relevant chunks for a question ---
def retrieve(question, top_k=3):
    q_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=top_k)
    return results["documents"][0]

# --- Step 4: Ask the LLM, grounded in retrieved chunks ---
def ask(question):
    context_chunks = retrieve(question)
    context = "\n\n".join(context_chunks)

    prompt = f"""নিচের তথ্যের ভিত্তিতে প্রশ্নের উত্তর বাংলায় দিন।

তথ্য:
{context}

প্রশ্ন: {question}

উত্তর:"""

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# --- Main ---
if __name__ == "__main__":
    if collection.count() == 0:
        index_notes()
    else:
        print(f"Using existing index with {collection.count()} chunks.")

    print("\nআপনার প্রশ্ন লিখুন (exit লিখে বের হন):\n")
    while True:
        q = input("প্রশ্নঃ ")
        if q.lower() == "exit":
            break
        answer = ask(q)
        print("\nউত্তরঃ", answer, "\n")