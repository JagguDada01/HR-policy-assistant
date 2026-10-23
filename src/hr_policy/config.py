"""Shared project settings."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
PDF_PATH = ROOT_DIR / "data" / "source" / "hr_manual.pdf"
CHROMA_DIR = ROOT_DIR / "data" / "chroma"
COLLECTION_NAME = "hr_manual"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
