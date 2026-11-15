import os
import json
import re
from datetime import datetime

# Configurable constants for chunking
CHUNK_SIZE=500
CHUNK_OVERLAP=100
DATA_DIR="data"
OUTPUT_FILE="output/chunks.json"


def load_documents(folder_path):
    """Loads all text files from the data folder."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    documents={}
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            file_path=os.path.join(folder_path,filename)
            with open(file_path,"r",encoding="utf-8") as f:
                documents[filename]=f.read()

    if not documents:
        raise ValueError(f"No .txt documents found in '{folder_path}'")

    return documents


def clean_text(raw_text):
    """Cleans up text by normalizing line breaks and removing extra spaces."""
    cleaned=re.sub(r"\s+"," ",raw_text)
    return cleaned.strip()


def chunk_text(text,chunk_size=CHUNK_SIZE,overlap=CHUNK_OVERLAP):
    """
    Splits text into overlapping chunks while preserving full words.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    chunks=[]
    start=0
    text_length=len(text)

    while start < text_length:
        end=start+chunk_size

        if end >= text_length:
            chunk=text[start:text_length].strip()
            if chunk:
                chunks.append((start,text_length,chunk))
            break

        # Look for the last space before end to avoid cutting words
        last_space=text.rfind(" ",start,end)
        if last_space > start:
            end=last_space

        chunk=text[start:end].strip()
        if chunk:
            chunks.append((start,end,chunk))

        # Step back by overlap amount
        start=end-overlap

        # If start is in the middle of a word, jump to the next word
        if start > 0 and text[start-1] != " ":
            next_space=text.find(" ",start)
            if next_space != -1 and next_space < end:
                start=next_space+1

    return chunks


def process_documents():
    print(f"Loading documents from '{DATA_DIR}'...")
    docs=load_documents(DATA_DIR)
    print(f"Found {len(docs)} files.")

    all_chunks=[]

    for filename,content in docs.items():
        clean_content=clean_text(content)
        doc_chunks=chunk_text(clean_content)

        base_id=os.path.splitext(filename)[0]
        for i,(start,end,chunk_str) in enumerate(doc_chunks):
            all_chunks.append({
                "chunk_id": f"{base_id}_{i:03d}",
                "source": filename,
                "chunk_index": i,
                "start_char": start,
                "end_char": end,
                "char_count": len(chunk_str),
                "text": chunk_str
            })

        print(f" -> {filename}: {len(doc_chunks)} chunks")

    # Ensure output directory exists before saving
    output_dir=os.path.dirname(OUTPUT_FILE)
    if output_dir:
        os.makedirs(output_dir,exist_ok=True)

    result={
        "created_at": datetime.now().isoformat(),
        "settings": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP
        },
        "total_chunks": len(all_chunks),
        "chunks": all_chunks
    }

    with open(OUTPUT_FILE,"w",encoding="utf-8") as f:
        json.dump(result,f,indent=2)

    print(f"\nDone! {len(all_chunks)} chunks saved to '{OUTPUT_FILE}'.")


if __name__=="__main__":
    process_documents()
