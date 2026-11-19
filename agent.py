import os
import sys
import json
import logging
from datetime import datetime
from semantic_search import load_chunks,build_vectorizer,search

# Configuration settings
LOG_FILE="agent.log"
CHUNKS_FILE="output/chunks.json"
DEFAULT_TOP_K=2
MIN_SIMILARITY_SCORE=0.08

# Setup logging configuration
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger=logging.getLogger("PawPalsAgent")


def load_env():
    """Loads environment variables from .env file if present."""
    env_file=".env"
    if os.path.exists(env_file):
        with open(env_file,"r",encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k,v=line.split("=",1)
                    k_clean,v_clean=k.strip(),v.strip().strip("\"'")
                    if k_clean not in os.environ:
                        os.environ[k_clean]=v_clean


def get_llm_client():
    """Detects and returns configured LLM provider and client."""
    load_env()
    groq_key=os.getenv("GROQ_API_KEY")
    gemini_key=os.getenv("GEMINI_API_KEY")

    if groq_key:
        try:
            from groq import Groq
            return "groq",Groq(api_key=groq_key)
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")

    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            return "gemini",genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")

    return None,None


def call_llm(provider,client,prompt,system_prompt=""):
    """Sends prompt to the configured LLM API with error handling."""
    if provider=="groq":
        candidate_models=["openai/gpt-oss-20b","qwen/qwen3.8-27b","llama-3.3-70b-versatile","llama-3.1-8b-instant"]
        last_err=None
        for model_name in candidate_models:
            try:
                response=client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system","content": system_prompt},
                        {"role": "user","content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=400
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_err=e
                continue
        raise last_err

    elif provider=="gemini":
        full_prompt=f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response=client.generate_content(full_prompt)
        return response.text.strip()

    raise ValueError(f"Unsupported LLM provider: {provider}")


def build_prompt(query,context_chunks):
    """Constructs prompt containing retrieved context and instructions."""
    system_prompt=(
        "You are the PawPals Pet Clinic AI Assistant.\n"
        "Answer the pet owner's question using ONLY the provided clinic guidelines below.\n"
        "Strict rules:\n"
        "1. Do not invent or assume any medical details outside the text.\n"
        "2. If the context does not answer the question, say you don't have that information.\n"
        "3. Always cite the source file at the end (e.g. Source: [filename]).\n"
        "4. Keep the answer friendly, caring, and concise."
    )

    context_text=""
    for i,chunk in enumerate(context_chunks,1):
        context_text+=f"\n--- Context Document {i} ({chunk['source']}) ---\n{chunk['text']}\n"

    user_prompt=f"Clinic Guidelines:\n{context_text}\nClient Question: {query}\n\nAnswer:"
    return system_prompt,user_prompt


def answer_question(query,vectorizer,matrix,chunks):
    """
    Core agent pipeline:
    1. Validates user input.
    2. Retrieves relevant chunks via semantic search.
    3. Handles empty retrieval fallback.
    4. Calls LLM with fallback for API errors.
    """
    logger.info(f"Incoming query: '{query}'")

    # 1. Input validation & fallback
    if not query or not query.strip():
        msg="Please enter a valid pet care question."
        logger.warning("Empty or invalid query received.")
        return {
            "answer": msg,
            "sources": [],
            "status": "INVALID_INPUT"
        }

    clean_query=query.strip()

    # 2. Retrieval step
    matches=search(clean_query,vectorizer,matrix,chunks,top_k=DEFAULT_TOP_K)

    # Filter out weak or irrelevant matches
    relevant_chunks=[m for m in matches if m["score"] >= MIN_SIMILARITY_SCORE]

    # 3. Empty retrieval fallback
    if not relevant_chunks:
        logger.warning(f"No relevant chunks found for query: '{clean_query}'")
        msg=(
            "I'm sorry, PawPals Clinic guidelines don't have information on that specific topic. "
            "Please call our front desk at (555) 0199 or speak with our veterinary team directly."
        )
        return {
            "answer": msg,
            "sources": [],
            "status": "NO_CONTEXT"
        }

    sources=list(dict.fromkeys([c["source"] for c in relevant_chunks]))
    logger.info(f"Retrieved {len(relevant_chunks)} chunks from: {', '.join(sources)}")

    # 4. LLM Generation step
    system_prompt,user_prompt=build_prompt(clean_query,relevant_chunks)
    provider,client=get_llm_client()

    if client is not None:
        try:
            logger.info(f"Generating answer using {provider}...")
            llm_answer=call_llm(provider,client,user_prompt,system_prompt)
            logger.info("Answer generated successfully.")
            return {
                "answer": llm_answer,
                "sources": sources,
                "status": "SUCCESS"
            }
        except Exception as err:
            logger.error(f"API call to {provider} failed: {err}")

    # Fallback when LLM API is unavailable, unconfigured or errored
    logger.info("Using local extractive fallback response.")
    source_citation=", ".join(sources)
    best_passage=relevant_chunks[0]["text"]

    fallback_answer=(
        f"Based directly on PawPals Clinic care guidelines ({source_citation}):\n"
        f"{best_passage}\n\n"
        f"Source: {source_citation}"
    )

    return {
        "answer": fallback_answer,
        "sources": sources,
        "status": "FALLBACK"
    }


def main():
    """Runs interactive terminal demo for the agent."""
    print("="*60)
    print("  PawPals Pet Clinic - AI Knowledge Agent Demo")
    print("="*60)
    print("Loading clinic guidelines and preparing vector search...")

    chunks=load_chunks(CHUNKS_FILE)
    vectorizer,matrix=build_vectorizer(chunks)
    print(f"Agent ready. Loaded {len(chunks)} knowledge chunks.")
    print("Logging interactions to 'agent.log'.\n")

    # Sample demo questions to verify pipeline automatically
    demo_questions=[
        "When should puppies receive their core vaccinations?",
        "What human foods are toxic to cats?",
        "How can I tell if my dog needs emergency care?",
        "How do I fly a spaceship to Mars?"  # Out of domain query to test fallback
    ]

    print("Running automated demo questions:\n")
    for q in demo_questions:
        print(f"Client Question: {q}")
        res=answer_question(q,vectorizer,matrix,chunks)
        print(f"Agent Status   : [{res['status']}]")
        if res["sources"]:
            print(f"Cited Sources  : {', '.join(res['sources'])}")
        print(f"Answer:\n{res['answer']}\n")
        print("-" * 60)

    # If command line arguments provided, process them
    if len(sys.argv) > 1:
        custom_query=" ".join(sys.argv[1:])
        print(f"\nProcessing custom query: '{custom_query}'")
        res=answer_question(custom_query,vectorizer,matrix,chunks)
        print(f"Status: [{res['status']}]")
        print(f"Answer:\n{res['answer']}")


if __name__=="__main__":
    main()


