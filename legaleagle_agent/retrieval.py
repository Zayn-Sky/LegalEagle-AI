from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_chroma import Chroma
from langchain_core.documents import Document

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


def search_laws(query: str, k: int = 8, persist_directory: Path = CHROMA_DIR) -> List[Document]:
    return load_vectorstore(persist_directory=persist_directory).similarity_search(
        query,
        k=k,
        filter={"status": "有效"},
    )


def format_documents(docs: Iterable[Document]) -> str:
    blocks: List[str] = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata
        citation = (
            f"{meta.get('title', '未知法律')}"
            f"｜{meta.get('article') or meta.get('unit_type', '片段')}"
            f"｜时效性:{meta.get('status', '')}"
            f"｜公布:{meta.get('publish_date', '')}"
            f"｜施行:{meta.get('effective_date', '')}"
        )
        blocks.append(f"[{idx}] {citation}\n{doc.page_content}")
    return "\n\n".join(blocks)
