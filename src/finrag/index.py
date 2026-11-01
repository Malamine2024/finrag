"""Embed the chunks with Ollama and store them in a persistent Chroma collection.

Incremental: only new chunks and chunks whose text changed are embedded again, and chunks that no longer
exist in chunks.jsonl are deleted. A re-ingestion that changes a few thousand chunks (table titles...)
therefore takes minutes, not hours. --reset rebuilds everything.
"""
import argparse
import json

import chromadb
import ollama
from tqdm import tqdm

from .config import CHROMA_DIR, CHUNKS_FILE, COLLECTION, EMBED_MODEL, OLLAMA_HOST

BATCH_SIZE = 32
READ_BATCH = 5000      # chunks read back from Chroma at a time, to compare their text


def get_ollama() -> ollama.Client:
    return ollama.Client(host=OLLAMA_HOST)


def embed(texts: list[str], client: ollama.Client | None = None) -> list[list[float]]:
    client = client or get_ollama()
    return client.embed(model=EMBED_MODEL, input=texts).embeddings


def get_collection(reset: bool = False):
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            chroma.delete_collection(COLLECTION)
        except Exception:
            pass
    return chroma.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def indexed_documents(collection) -> dict[str, str]:
    """id -> text of every chunk already in the collection."""
    documents = {}
    total = collection.count()
    for offset in tqdm(range(0, total, READ_BATCH), desc="Reading the index", disable=total == 0):
        batch = collection.get(include=["documents"], limit=READ_BATCH, offset=offset)
        documents.update(zip(batch["ids"], batch["documents"]))
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or update the vector index.")
    parser.add_argument("--reset", action="store_true", help="delete and rebuild the collection")
    args = parser.parse_args()

    if not CHUNKS_FILE.exists():
        raise SystemExit(f"{CHUNKS_FILE} not found. Run `python -m finrag.ingest` first.")
    chunks = [json.loads(line) for line in CHUNKS_FILE.open(encoding="utf-8") if line.strip()]

    collection = get_collection(reset=args.reset)
    existing = indexed_documents(collection)
    wanted = {c["id"] for c in chunks}
    todo = [c for c in chunks if existing.get(c["id"]) != c["text"]]
    stale = [i for i in existing if i not in wanted]
    print(f"{len(chunks)} chunks: {len(chunks) - len(todo)} unchanged, {len(todo)} to embed, "
          f"{len(stale)} to delete")

    for start in range(0, len(stale), READ_BATCH):
        collection.delete(ids=stale[start:start + READ_BATCH])
    client = get_ollama()
    for start in tqdm(range(0, len(todo), BATCH_SIZE), desc="Embedding", disable=not todo):
        batch = todo[start:start + BATCH_SIZE]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embed([c["text"] for c in batch], client),
            metadatas=[{k: c[k] for k in ("company", "year", "page", "kind", "source")} for c in batch],
        )
    print(f"Indexed {collection.count()} chunks in '{COLLECTION}'")


if __name__ == "__main__":
    main()
