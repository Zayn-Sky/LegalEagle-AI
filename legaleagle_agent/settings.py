from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

DATA_DIR = PROJECT_ROOT / "data"
RAW_NPC_DIR = DATA_DIR / "raw" / "npc"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "index" / "chroma"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
LAWS_PATH = PROCESSED_DIR / "laws.jsonl"

DEFAULT_AGENT_MODEL = os.getenv("LEGAL_AGENT_MODEL", "gpt-5.4-mini")
DEFAULT_EMBEDDING_MODEL = os.getenv("LEGAL_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
CHROMA_COLLECTION = "npc_civil_social_laws"
REQUIRED_CATEGORY_CODES = {120, 150}
