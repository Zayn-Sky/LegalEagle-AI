from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma

from .embeddings import get_embeddings
from .settings import CHROMA_COLLECTION, CHROMA_DIR


def load_vectorstore(persist_directory: Path = CHROMA_DIR) -> Chroma:
    sqlite_path = persist_directory / "chroma.sqlite3"
    if not sqlite_path.exists():
        raise FileNotFoundError(f"Missing Chroma index: {sqlite_path}. Run scripts/run_full_pipeline.py first.")
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(persist_directory),
        embedding_function=get_embeddings(),
    )
