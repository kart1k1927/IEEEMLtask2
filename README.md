# Overview

This project implements a semantic search system using Sentence Transformers and FAISS.
Instead of traditional keyword-based search, this system retrieves documents based on semantic similarity, meaning it understands the context of a query rather than exact word matches.

The project was developed as part of IEEE ML Task 2.

# Objective

Convert text documents into dense vector embeddings

Index embeddings using FAISS for fast similarity search

Retrieve the most relevant documents for a given query

Demonstrate practical understanding of NLP embeddings and vector databases


# Technologies Used

Python 3

Sentence Transformers (Hugging Face)

FAISS (CPU) – Facebook AI Similarity Search

NumPy

Pandas

PyYAML

# Dataset (Corpus)

The dataset consists of 45 IEEE-style textual entries related to:

Machine Learning

Deep Learning

NLP

AI applications

Neural Networks

Each entry contains:

doc_id – unique document identifier

text – document content

Example:

doc_id,text
D1,"Machine learning enables systems to learn from data without explicit programming."

🔧 Configuration

All configurations are managed via configs/config.yaml.

Example:

model_name: sentence-transformers/all-MiniLM-L6-v2
top_k: 5


model_name: Pretrained Sentence Transformer model

top_k: Number of top results returned during search

# Installation
Step 1: Clone the repository
git clone https://github.com/kart1k1927/IEEEMLtask2.git
cd IEEEMLtask2

Step 2: Install dependencies
pip install -r requirements.txt

# Training (Embedding & Index Creation)

Run the following command to:

Load the corpus

Generate sentence embeddings

Build a FAISS similarity index

python src/train.py


This step converts textual data into vector representations and stores them in a FAISS index for efficient search.

# Semantic Search

To perform semantic search:

python src/search.py


You will be prompted to enter a query:

Enter your query: deep learning applications

Example Output:
Query: deep learning applications

1. [DocID: D12] Deep learning models are widely used in computer vision... (score=0.78)
2. [DocID: D7] Neural networks have transformed image recognition... (score=0.65)
3. [DocID: D19] Artificial intelligence applications include healthcare... (score=0.52)


Documents are ranked based on cosine similarity.

# Evaluation

Since this is an unsupervised semantic search system, traditional accuracy metrics are not applicable.

Evaluation is performed qualitatively by:

Observing relevance of top-k retrieved documents

Comparing similarity scores

Testing multiple natural language queries

Screenshots of final search outputs are provided as part of submission.

# Features

Context-aware semantic search

Fast similarity lookup using FAISS

Config-driven design

Modular and clean codebase

Easily extensible for larger datasets

# Future Improvements

Add quantitative evaluation metrics (MRR / NDCG)

Support larger datasets

Save & reload FAISS index from disk

Web-based UI for querying

Experiment with larger transformer models

