# PawPals Pet Care Knowledge Agent

An AI assistant built for PawPals Pet Clinic that retrieves clinical care guidelines and generates grounded, cited answers for pet owners.

---

## Overview by Task

### Task 1: Knowledge Base & Chunking
- Ingests 6 realistic pet care care sheets from `data/`.
- Cleans and splits text into overlapping chunks (`CHUNK_SIZE=500`, `CHUNK_OVERLAP=100`) preserving word boundaries.
- Saves structured chunks to `output/chunks.json`.

```bash
python3 chunk_documents.py
```

### Task 2: Vector Embeddings & Semantic Search
- Transforms text chunks into TF-IDF vector embeddings (unigrams + bigrams) using scikit-learn.
- Implements cosine similarity search to retrieve top relevant passages.
- Includes unit tests in `test_search.py` verifying ranking accuracy and handling empty queries.

```bash
python3 semantic_search.py
python3 test_search.py
```

### Task 3: LLM Answer Generation & Error Logging
- Combines semantic retrieval with a free LLM API (Groq or Gemini) using strict system prompts to prevent hallucinations.
- Cites the exact source care sheets (e.g. `dog_vaccination_schedule.txt`).
- **Error & Fallback Handling**:
  - Handles invalid or empty user input gracefully.
  - Returns a clinic referral fallback if no matching guidelines are found (`NO_CONTEXT`).
  - Provides a direct extractive fallback from retrieved guidelines if the LLM API is unavailable or unconfigured.
- Logs all interactions, timestamps, queries, and status codes to `agent.log`.

---

## Setup & Running the Agent

### 1. Requirements
Python 3.8+ with dependencies:
```bash
pip install -r requirements.txt
```
*(Dependencies: `scikit-learn`, `numpy`, `groq`, `google-generativeai`)*

### 2. Configure Free LLM API (Optional)
Export your free API key from [Groq](https://console.groq.com) or [Google AI Studio](https://aistudio.google.com):
```bash
export GROQ_API_KEY="your-free-groq-key"
# or
export GEMINI_API_KEY="your-free-gemini-key"
```
*(Note: If no key is set, the agent uses the local extractive fallback so it can run 100% offline).*

### 3. Run the Agent
Run automated demo questions:
```bash
python3 agent.py
```

Or ask a question directly:
```bash
python3 agent.py "When should my puppy get vaccinated?"
```

### 4. Run All Unit Tests
```bash
python3 -m unittest discover -s . -p "test_*.py"
```

---

## Demo Recording
- **Demo Video**: [Link to Screen Recording](https://github.com/UmedAli123/PawPals-Clinic-Pet-Care-Knowledge-Agent#demo-recording) *(record a 1-2 min screen demo of running `python3 agent.py` and paste link here)*

---

## Project Structure
```
pawpals-knowledge-agent/
├── data/                      # 6 pet care guideline sheets
├── output/                    # generated chunks JSON
├── chunk_documents.py         # Task 1: cleaning & chunking
├── semantic_search.py         # Task 2: TF-IDF vector search
├── test_search.py             # Task 2: search unit tests
├── agent.py                   # Task 3: RAG pipeline with LLM & fallbacks
├── test_agent.py              # Task 3: agent unit tests
├── agent.log                  # Task 3: interaction log file
├── .env.example               # sample API key configuration
└── README.md
```
