import os
import sys
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Constants
CHUNKS_FILE="output/chunks.json"
DEFAULT_TOP_K=3


def load_chunks(file_path=CHUNKS_FILE):
    """Loads chunks from the json file generated in Task 1."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Chunks file not found at: {file_path}")

    with open(file_path,"r",encoding="utf-8") as f:
        data=json.load(f)

    chunks=data.get("chunks",[])
    if not chunks:
        raise ValueError(f"No chunks found in {file_path}")

    return chunks


def build_vectorizer(chunks):
    """Generates TF-IDF vector embeddings for all text chunks."""
    corpus=[c["text"] for c in chunks]
    vectorizer=TfidfVectorizer(stop_words="english",ngram_range=(1,2))
    matrix=vectorizer.fit_transform(corpus)
    return vectorizer,matrix


def search(query,vectorizer,matrix,chunks,top_k=DEFAULT_TOP_K):
    """
    Finds the most relevant chunks for a user query using cosine similarity.
    Handles empty or whitespace-only queries gracefully.
    """
    if not query or not query.strip():
        return []

    # Transform query to vector
    query_vec=vectorizer.transform([query.strip()])

    # Compute cosine similarity
    similarities=cosine_similarity(query_vec,matrix).flatten()

    # Get top matching indices sorted by score descending
    sorted_indices=np.argsort(similarities)[::-1]

    results=[]
    for idx in sorted_indices[:top_k]:
        score=float(similarities[idx])
        if score > 0:
            results.append({
                "chunk_id": chunks[idx]["chunk_id"],
                "source": chunks[idx]["source"],
                "score": round(score,4),
                "text": chunks[idx]["text"]
            })

    return results


def run_search_tool():
    """CLI helper to test semantic search interactively or with sample queries."""
    print(f"Loading knowledge base chunks from '{CHUNKS_FILE}'...")
    chunks=load_chunks()
    print(f"Loaded {len(chunks)} chunks.")

    vectorizer,matrix=build_vectorizer(chunks)
    print("TF-IDF vector embeddings generated successfully.\n")

    sample_questions=[
        "When should puppies get their core vaccines?",
        "What foods are toxic to cats?",
        "What are warning signs of a pet emergency?",
        "How often should I brush my dog's teeth?"
    ]

    print("Running sample clinic questions:")
    for question in sample_questions:
        print(f"\n--- Question: '{question}' ---")
        matches=search(question,vectorizer,matrix,chunks,top_k=2)
        if not matches:
            print("No relevant passages found.")
        for rank,m in enumerate(matches,1):
            print(f" Match #{rank} (Score: {m['score']}) [{m['source']}]:")
            preview=m["text"][:140]+"..." if len(m["text"]) > 140 else m["text"]
            print(f"   \"{preview}\"")


if __name__=="__main__":
    if len(sys.argv) > 1:
        cli_query=" ".join(sys.argv[1:])
        chunks=load_chunks()
        vec,matrix=build_vectorizer(chunks)
        matches=search(cli_query,vec,matrix,chunks,top_k=3)
        print(f"Results for '{cli_query}':\n")
        if not matches:
            print("No matching clinic guidelines found.")
        for rank,m in enumerate(matches,1):
            print(f"#{rank} [{m['source']}] (Score: {m['score']}):")
            print(f"   {m['text']}\n")
    else:
        run_search_tool()
