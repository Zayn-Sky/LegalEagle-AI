from __future__ import annotations

import argparse
from pathlib import Path

from .legal_rag import answer_question
from .settings import CHROMA_DIR, CHUNKS_PATH, DEFAULT_AGENT_MODEL


def ask(
    question: str,
    model: str = DEFAULT_AGENT_MODEL,
    persist_dir: Path = CHROMA_DIR,
    chunks_path: Path = CHUNKS_PATH,
) -> str:
    return answer_question(question, model=model, persist_dir=persist_dir, chunks_path=chunks_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LegalEagle legal consultation agent.")
    parser.add_argument("question", nargs="?", help="法律问题")
    parser.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--persist-dir", default=str(CHROMA_DIR))
    parser.add_argument("--chunks-path", default=str(CHUNKS_PATH))
    args = parser.parse_args()

    question = args.question or input("请输入法律问题：").strip()
    print(
        ask(
            question,
            model=args.model,
            persist_dir=Path(args.persist_dir),
            chunks_path=Path(args.chunks_path),
        )
    )


if __name__ == "__main__":
    main()
