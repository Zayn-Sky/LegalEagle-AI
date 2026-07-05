from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_vectorstore import build
from scripts.preprocess_rag import preprocess
from scripts.scrape_npc_laws import scrape


def run(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    index_dir = Path(args.index_dir)

    scrape(
        SimpleNamespace(
            output_dir=str(raw_dir),
            page_size=args.page_size,
            sleep=args.sleep,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
            trust_env_proxy=args.trust_env_proxy,
        )
    )
    preprocess(
        SimpleNamespace(
            raw_dir=str(raw_dir),
            output_dir=str(processed_dir),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    )
    build(
        SimpleNamespace(
            chunks=str(processed_dir / "chunks.jsonl"),
            persist_dir=str(index_dir),
            collection=args.collection,
            batch_size=args.batch_size,
            recreate=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the only supported full LegalEagle data and index pipeline.")
    parser.add_argument("--raw-dir", default="data/raw/npc")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--index-dir", default="data/index/chroma")
    parser.add_argument("--collection", default="npc_civil_social_laws")
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--trust-env-proxy", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
