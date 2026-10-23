"""Load, split, embed, and persist the HR manual."""
import hashlib
import json
import re

import chromadb
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_chroma import Chroma
except ImportError:
    # Compatibility for environments that have not installed langchain-chroma yet.
    from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME,
                     EMBEDDING_MODEL, PDF_PATH)

MANIFEST_PATH = CHROMA_DIR / "index_manifest.json"
ANNEXURE_RE = re.compile(r"ANNEXURE\s*[-–:]?\s*([IVX]+)", re.IGNORECASE)
POLICY_RE = re.compile(r"^\s*(?:\[[A-Z]\]\s*)?([A-Z][A-Za-z &/()'-]{2,} Policy)\s*$", re.MULTILINE)


def _pdf_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pages(pdf_path: Path = PDF_PATH):
    """Load PDF pages and carry forward annexure and policy headings."""
    loaded = PyPDFLoader(str(pdf_path)).load()
    annexure = ""
    policy = ""
    for doc in loaded:
        text = doc.page_content
        annexure_match = ANNEXURE_RE.search(text)
        if annexure_match:
            annexure = f"Annexure {annexure_match.group(1).upper()}"
        heading = POLICY_RE.search(text)
        if heading:
            policy = heading.group(1).strip()
        doc.metadata["page_number"] = int(doc.metadata.get("page", 0)) + 1
        doc.metadata["annexure"] = annexure
        doc.metadata["policy"] = policy
        doc.metadata["section"] = ", ".join(x for x in (annexure, policy) if x)
    return loaded


def get_embeddings():
    """Create the local sentence-transformer embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(force: bool = False):
    """Build the persistent index once; reuse it when the source PDF is unchanged."""
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Place hr_manual.pdf at {PDF_PATH}")
    fingerprint = _pdf_hash(PDF_PATH)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    if not force and MANIFEST_PATH.exists():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text())
            store = Chroma(collection_name=COLLECTION_NAME, persist_directory=str(CHROMA_DIR),
                           embedding_function=get_embeddings())
            if manifest.get("sha256") == fingerprint and store._collection.count() > 0:
                return store
        except Exception:
            pass

    # Drop the old collection before rebuilding so removed chunks cannot remain searchable.
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    pages = load_pages()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    # The carried section fields stay attached to each split chunk.
    ids = [f"page-{d.metadata['page_number']}-chunk-{i}" for i, d in enumerate(chunks)]
    store = Chroma.from_documents(
        documents=chunks, embedding=get_embeddings(), ids=ids,
        collection_name=COLLECTION_NAME, persist_directory=str(CHROMA_DIR),
    )
    MANIFEST_PATH.write_text(json.dumps({"sha256": fingerprint, "chunks": len(chunks)}, indent=2))
    return store


if __name__ == "__main__":
    db = build_vector_store()
    print(f"Indexed {db._collection.count()} chunks from {PDF_PATH.name}")
