# PawPals Pet Care Knowledge Agent (Task 1)

This repository contains Task 1 of the PawPals Pet Clinic knowledge agent project. It loads clinical pet care guidelines, cleans the text, and splits them into searchable chunks for the clinic's knowledge base.

## What it does
- Reads 6 pet care care sheets from the `data/` folder.
- Normalizes spacing and line breaks.
- Slices text into overlapping chunks while keeping words intact.
- Saves the chunks and metadata into `output/chunks.json`.

## Files
- `data/` – Contains text files on dog vaccines, cat nutrition, dental health, emergency signs, flea/tick prevention, and puppy house training.
- `chunk_documents.py` – Main script that reads, cleans, and chunks the files.
- `output/chunks.json` – Generated structured output file.

## How to Run
Requires Python 3.8+ (uses standard library only):

```bash
python3 chunk_documents.py
```

## Configuration
You can adjust chunk parameters directly at the top of `chunk_documents.py`:
- `CHUNK_SIZE = 500`: character length of each chunk.
- `CHUNK_OVERLAP = 100`: character overlap between neighboring chunks.
- `DATA_DIR = "data"`: input folder containing `.txt` files.
- `OUTPUT_FILE = "output/chunks.json"`: path to save the output JSON.

## Design Choices
- **Overlap**: Keeps context connected across chunk borders so information isn't cut in half during semantic search (Task 2).
- **Word boundaries**: Avoids splitting words mid-sentence for cleaner text embeddings.
- **Metadata**: Each chunk includes its source document and start/end character offsets for citation in future tasks.
