import yaml
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load fine-tuned model from Hugging Face
model = SentenceTransformer(config["model_name"])

# MACRO subset
import pandas as pd

corpus = pd.read_csv("data/corpus.csv")
documents = corpus["text"].tolist()
doc_ids = corpus["doc_id"].tolist()


# Encode documents
doc_embeddings = model.encode(documents, convert_to_numpy=True)

# Build FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(doc_embeddings)
index.add(doc_embeddings)

def search(query, top_k=5):
    query_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)

    scores, indices = index.search(query_emb, top_k)
    results = []

    for score, idx in zip(scores[0], indices[0]):
       results.append((doc_ids[idx], documents[idx], float(score)))


    return results

if __name__ == "__main__":
    while True:
        query = input("\nEnter query (or type exit): ")
        if query.lower() == "exit":
            break

        results = search(query, config["top_k"])
        for rank, (doc, score) in enumerate(results, 1):
          print(f"{rank}. [DocID: {doc_id}] {doc[:200]}... (score={score:.4f})")

