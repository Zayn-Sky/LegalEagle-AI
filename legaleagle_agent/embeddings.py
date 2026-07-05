from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from .settings import DEFAULT_EMBEDDING_MODEL, EMBEDDING_API_KEY, EMBEDDING_BASE_URL


def require_openai_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for the LLM in the only supported execution path.")


def require_embedding_api_key() -> None:
    if not EMBEDDING_API_KEY:
        raise RuntimeError("EMBEDDING_API_KEY is required for embeddings in the only supported execution path.")


def get_embeddings() -> Embeddings:
    require_embedding_api_key()
    return OpenAIEmbeddings(
        model=DEFAULT_EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL,
        check_embedding_ctx_length=False,
    )
