# PawPals Pet Care Knowledge Agent

This repository contains the AI knowledge agent for PawPals Pet Clinic, developed across tasks.

## Task 1: Knowledge Base Setup and Chunking
- Reads 6 clinical pet care guides from `data/`.
- Cleans and chunks the documents into overlapping segments (`CHUNK_SIZE=500`, `CHUNK_OVERLAP=100`).
- Keeps words intact at boundaries and saves metadata to `output/chunks.json`.

Run Task 1:
```bash
python3 chunk_documents.py
```

## Task 2: Vector Embeddings and Semantic Search
- Generates vector representations for all chunks from Task 1 using TF-IDF feature extraction with unigrams and bigrams.
- Implements cosine similarity search to retrieve the most relevant clinic passages for any user query.
- Gracefully handles empty, whitespace-only, or invalid inputs.
- Includes a test suite verifying search accuracy, ranking, and edge cases.

### How to Run Semantic Search
Run with sample clinic questions:
```bash
python3 semantic_search.py
```

Or pass your own question directly:
```bash
python3 semantic_search.py "When should puppies get their rabies vaccine?"
```

### Running Tests
Run the unit test suite:
```bash
python3 test_search.py
```

## Project Files
- `data/` – 6 pet care care sheets (vaccinations, cat nutrition, emergencies, dental care, fleas/ticks, puppy training).
- `chunk_documents.py` – Ingestion and chunking script (Task 1).
- `output/chunks.json` – Exported text chunks with source metadata.
- `semantic_search.py` – Embedding generation and cosine similarity search tool (Task 2).
- `test_search.py` – Unit tests verifying retrieval accuracy and empty input handling (Task 2).
