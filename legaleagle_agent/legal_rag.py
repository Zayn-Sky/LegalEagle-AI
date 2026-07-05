from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from .embeddings import require_openai_api_key
from .retrieval import load_vectorstore
from .settings import CHROMA_DIR, CHUNKS_PATH, DEFAULT_AGENT_MODEL, OPENAI_BASE_URL


class SearchPlan(BaseModel):
    legal_issues: List[str] = Field(description="法律问题拆解，最多 4 个")
    search_queries: List[str] = Field(description="面向法规库检索的中文查询，2 到 4 个")
    missing_facts: List[str] = Field(default_factory=list, description="需要用户补充的关键事实")


def build_llm(model: str = DEFAULT_AGENT_MODEL) -> ChatOpenAI:
    require_openai_api_key()
    return ChatOpenAI(model=model, base_url=OPENAI_BASE_URL, temperature=0)


def plan_question(question: str, model: str = DEFAULT_AGENT_MODEL) -> SearchPlan:
    planner = build_llm(model).with_structured_output(SearchPlan, method="function_calling")
    return planner.invoke(
        [
            SystemMessage(
                content=(
                    "你是中国法律咨询检索规划器。把用户问题拆成可检索的法律争点，"
                    "输出适合在民法商法、社会法法规库中检索的短查询。"
                    "查询应包含法律术语、义务/责任/救济关键词，不要编造事实。"
                )
            ),
            HumanMessage(content=question),
        ]
    )


def tokenize(text: str) -> List[str]:
    compact = re.sub(r"\s+", "", text.lower())
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    for segment in re.findall(r"[\u4e00-\u9fff]+", compact):
        tokens.extend(segment[index : index + 2] for index in range(max(0, len(segment) - 1)))
        tokens.extend(segment[index : index + 3] for index in range(max(0, len(segment) - 2)))
    return tokens or [compact]


def load_chunks(path: Path = CHUNKS_PATH) -> List[Document]:
    docs: List[Document] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            metadata = row["metadata"]
            if metadata.get("status") == "有效":
                docs.append(Document(page_content=row["text"], metadata=metadata))
    return docs


@lru_cache(maxsize=2)
def bm25_index(chunks_path: str) -> tuple[BM25Okapi, List[Document]]:
    docs = load_chunks(Path(chunks_path))
    return BM25Okapi([tokenize(doc.page_content) for doc in docs]), docs


def bm25_search(query: str, k: int = 12, chunks_path: Path = CHUNKS_PATH) -> List[Document]:
    index, docs = bm25_index(str(chunks_path))
    scores = index.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda item: scores[item], reverse=True)[:k]
    return [docs[index_] for index_ in ranked if scores[index_] > 0]


def reciprocal_rank_fusion(groups: Iterable[List[Document]], k: int = 12, constant: int = 60) -> List[Document]:
    scores: Dict[str, float] = {}
    docs_by_id: Dict[str, Document] = {}
    for docs in groups:
        for rank, doc in enumerate(docs, start=1):
            chunk_id = doc.metadata.get("chunk_id") or f"{doc.metadata.get('title')}:{rank}"
            docs_by_id[chunk_id] = doc
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (constant + rank)
    return [docs_by_id[chunk_id] for chunk_id in sorted(scores, key=scores.get, reverse=True)[:k]]


def hybrid_retrieve(
    queries: List[str],
    persist_dir: Path = CHROMA_DIR,
    chunks_path: Path = CHUNKS_PATH,
    k_per_query: int = 12,
    final_k: int = 10,
) -> List[Document]:
    vectorstore = load_vectorstore(persist_directory=persist_dir)
    groups: List[List[Document]] = []
    for query in queries:
        groups.append(
            vectorstore.similarity_search(
                query,
                k=k_per_query,
                filter={"status": "有效"},
            )
        )
        groups.append(bm25_search(query, k=k_per_query, chunks_path=chunks_path))
    return reciprocal_rank_fusion(groups, k=final_k)


def format_sources(docs: List[Document]) -> str:
    blocks = []
    for index, doc in enumerate(docs, start=1):
        meta = doc.metadata
        blocks.append(
            "\n".join(
                [
                    f"[S{index}] {meta.get('title')}｜{meta.get('article') or meta.get('unit_type')}｜{meta.get('status')}",
                    f"公布日期：{meta.get('publish_date')}｜施行日期：{meta.get('effective_date')}",
                    doc.page_content,
                ]
            )
        )
    return "\n\n".join(blocks)


def generate_answer(
    question: str,
    plan: SearchPlan,
    docs: List[Document],
    model: str = DEFAULT_AGENT_MODEL,
) -> str:
    if not docs:
        raise RuntimeError("No legal basis retrieved from the complete local law database.")

    source_ids = ", ".join(f"[S{index}]" for index in range(1, len(docs) + 1))
    prompt = f"""你是中国法律咨询智能体，回答要像律师初步咨询意见，但不能替代律师正式意见。

必须遵守：
1. 只使用给定法规依据，不得编造法律、条款、章节、判例或行政规则。
2. 每个关键法律结论后必须引用来源编号，例如 [S1]。
3. 只允许引用这些来源编号：{source_ids}。
4. 如果事实不足，先给基于现有事实的初步结论，再列出需要补充的事实。
5. 优先使用现行有效法规；不要引用已废止依据。
6. 引用条款名称必须完整，不得写“第六”“第七”等残缺条款；多条法条要分别完整写出。
7. 答案结构固定为：结论、法律依据、法律分析、行动建议、风险提示、需要补充的事实。

用户问题：
{question}

检索规划：
法律争点：{"；".join(plan.legal_issues)}
需要补充事实：{"；".join(plan.missing_facts) or "无"}

法规依据：
{format_sources(docs)}
"""
    answer = build_llm(model).invoke(prompt).content
    validate_citations(str(answer), len(docs))
    return str(answer)


def validate_citations(answer: str, source_count: int) -> None:
    cited = {int(match) for match in re.findall(r"\[S(\d+)\]", answer)}
    if not cited:
        raise RuntimeError("Generated answer contains no source citations.")
    invalid = sorted(value for value in cited if value < 1 or value > source_count)
    if invalid:
        raise RuntimeError(f"Generated answer cites unknown sources: {invalid}")


def answer_question(
    question: str,
    model: str = DEFAULT_AGENT_MODEL,
    persist_dir: Path = CHROMA_DIR,
    chunks_path: Path = CHUNKS_PATH,
) -> str:
    plan = plan_question(question, model=model)
    queries = [question, *plan.search_queries]
    docs = hybrid_retrieve(queries, persist_dir=persist_dir, chunks_path=chunks_path)
    return generate_answer(question, plan, docs, model=model)
