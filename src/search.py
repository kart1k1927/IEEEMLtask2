import yaml
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load config
# -----------------------------
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# -----------------------------
# Load model from Hugging Face
# -----------------------------
model = SentenceTransformer(config["model_name"])

# -----------------------------
# Load corpus
# -----------------------------
corpus = pd.read_csv("data/corpus.csv")
documents = corpus["text"].tolist()
doc_ids = corpus["doc_id"].tolist()

# -----------------------------
# Encode documents
# -----------------------------
doc_embeddings = model.encode(
    documents,
    convert_to_numpy=True,
    show_progress_bar=True
)

# -----------------------------
# Build FAISS index
# -----------------------------
dim = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)   # Inner Product (cosine if normalized)
faiss.normalize_L2(doc_embeddings)
index.add(doc_embeddings)

# -----------------------------
# Search function
# -----------------------------
def search(query, top_k=10):
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    print(f"\nQuery: {query}\n")
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        print(
            f"{rank}. [DocID: {doc_ids[idx]}] "
            f"{documents[idx][:200]}... "
            f"(score={score:.4f})"
        )

# -----------------------------
# Example query (THIS is important)
# -----------------------------
if __name__ == "__main__":
    search("machine learning applications", top_k=config["top_k"])


    results = search(query, config["top_k"])
    for rank, (doc, score) in enumerate(results, 1):
        print(f"{rank}. [DocID: {doc_id}] {doc[:200]}... (score={score:.4f})")

