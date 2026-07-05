from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legaleagle_agent.embeddings import get_embeddings
from legaleagle_agent.settings import CHROMA_COLLECTION


def iter_chunks(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def build(args: argparse.Namespace) -> None:
    chunks_path = Path(args.chunks)
    persist_dir = Path(args.persist_dir)
    if args.recreate and persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    docs: List[Document] = []
    ids: List[str] = []
    for chunk in iter_chunks(chunks_path):
        docs.append(Document(page_content=chunk["text"], metadata=chunk["metadata"]))
        ids.append(chunk["id"])
    if not docs:
        raise RuntimeError(f"No chunks found in {chunks_path}")

    vectorstore = Chroma(
        collection_name=args.collection,
        persist_directory=str(persist_dir),
        embedding_function=get_embeddings(),
    )
    try:
        for start in tqdm(range(0, len(docs), args.batch_size), desc="index"):
            end = start + args.batch_size
            vectorstore.add_documents(docs[start:end], ids=ids[start:end])
    except Exception:
        if args.recreate and persist_dir.exists():
            shutil.rmtree(persist_dir)
        raise

    indexed_count = vectorstore._collection.count()
    if indexed_count != len(docs):
        if args.recreate and persist_dir.exists():
            shutil.rmtree(persist_dir)
        raise RuntimeError(f"Chroma index incomplete: {indexed_count}/{len(docs)} chunks indexed.")

    print(f"Indexed {len(docs)} chunks into {persist_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chroma vector index from RAG chunks.")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--persist-dir", default="data/index/chroma")
    parser.add_argument("--collection", default=CHROMA_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--recreate", action="store_true", default=True)
    parser.add_argument("--no-recreate", dest="recreate", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
