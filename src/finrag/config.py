"""Central configuration. Values can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "annual_reports"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("FINRAG_EMBED_MODEL", "bge-m3")
LLM_MODEL = os.getenv("FINRAG_LLM_MODEL", "qwen3.5:9b")
NUM_CTX = int(os.getenv("FINRAG_NUM_CTX", "8192"))      # 0 = Ollama default (may truncate the prompt)
PROMPT_VERSION = os.getenv("FINRAG_PROMPT", "v5")        # see generate.PROMPTS
# Reasoning ("thinking") mode of models that have one (Qwen 3.x...): "on", "off", or "" to let Ollama decide.
# Leave it empty for models without a thinking mode, such as qwen2.5.
THINK = os.getenv("FINRAG_THINK", "off").strip().lower()

CHUNK_SIZE = 1200      # characters per text chunk
CHUNK_OVERLAP = 200    # characters shared between consecutive chunks
MIN_CHUNK_CHARS = 80   # drop near-empty chunks (page numbers, headers)
TOP_K = 10              # number of chunks given to the LLM
