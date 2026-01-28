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
# Load model
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
index = faiss.IndexFlatIP(dim)   # Inner Product (cosine similarity)

faiss.normalize_L2(doc_embeddings)
index.add(doc_embeddings)

# -----------------------------
# Search function
# -----------------------------
def search(query, top_k):
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    # never ask more than available docs
    k = min(top_k, len(documents))

    scores, indices = index.search(query_embedding, k)

    results = []

    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        if score < -1e10:   # filter FAISS garbage values
            continue

        results.append({
            "doc_id": doc_ids[idx],
            "text": documents[idx],
            "score": float(score)
        })

    return results

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    query = input("Enter your query: ")

    results = search(query, config["top_k"])

    print(f"\nQuery: {query}\n")
    for rank, r in enumerate(results, start=1):
        print(
            f"{rank}. [DocID: {r['doc_id']}] "
            f"{r['text'][:200]}... "
            f"(score={r['score']:.4f})"
        )
