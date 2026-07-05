from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from .embeddings import require_openai_api_key
from .retrieval import load_vectorstore
from .settings import (
    CHROMA_DIR,
    CHUNKS_PATH,
    DEFAULT_AGENT_MODEL,
    LAWS_PATH,
    OPENAI_BASE_URL,
    QUERY_EXPANSIONS_PATH,
)


class SearchPlan(BaseModel):
    legal_relations: List[str] = Field(default_factory=list, description="识别出的法律关系，最多 3 个")
    claim_bases: List[str] = Field(default_factory=list, description="请求权基础、抗辩或救济路径，最多 4 个")
    rule_elements: List[str] = Field(default_factory=list, description="构成要件或审查要点，最多 6 个")
    legal_issues: List[str] = Field(description="法律问题拆解，最多 4 个")
    search_queries: List[str] = Field(description="面向法规库检索的中文查询，2 到 4 个")
    preferred_categories: List[str] = Field(default_factory=list, description="民法商法或社会法")
    preferred_laws: List[str] = Field(default_factory=list, description="可能相关的准确法律名称")
    missing_facts: List[str] = Field(default_factory=list, description="需要用户补充的关键事实")
    needs_clarification: bool = Field(description="是否必须先追问事实才能给出有用答复")
    clarifying_questions: List[str] = Field(default_factory=list, description="必须先追问的问题，最多 4 个")


class RerankResult(BaseModel):
    ranked_source_ids: List[int] = Field(description="按相关性排序的候选来源编号")


class AnswerReview(BaseModel):
    approved: bool = Field(description="答案是否完全由来源支持")
    problems: List[str] = Field(default_factory=list, description="发现的问题")


def build_llm(model: str = DEFAULT_AGENT_MODEL) -> ChatOpenAI:
    require_openai_api_key()
    return ChatOpenAI(model=model, base_url=OPENAI_BASE_URL, temperature=0)


def plan_question(question: str, model: str = DEFAULT_AGENT_MODEL) -> SearchPlan:
    planner = build_llm(model).with_structured_output(SearchPlan, method="function_calling")
    return planner.invoke(
        [
            SystemMessage(
                content=(
                    "你是中国法律咨询检索规划器。按律师工作流将问题拆成法律关系、请求权基础、构成要件、法律争点和检索 query。"
                    "只允许把 preferred_categories 填为“民法商法”或“社会法”。"
                    "preferred_laws 必须尽量使用准确法律全称。"
                    "只有在缺少关键事实会导致无法判断法律关系、责任类型或救济路径时，"
                    "才将 needs_clarification 设为 true。若可以按不同事实情形给出一般法律规则，"
                    "例如有约定/无约定、已解除/未解除、已参保/未参保，应直接检索回答，并把待补事实放入 missing_facts。"
                )
            ),
            HumanMessage(content=question),
        ]
    )


def format_clarifying_questions(plan: SearchPlan) -> str:
    questions = plan.clarifying_questions or plan.missing_facts
    if not questions:
        raise RuntimeError("Clarification was requested but no questions were generated.")
    lines = ["为了给出更准确的法律分析，请先补充以下关键信息："]
    lines.extend(f"{index}. {question}" for index, question in enumerate(questions[:4], start=1))
    return "\n".join(lines)


def tokenize(text: str) -> List[str]:
    compact = re.sub(r"\s+", "", text.lower())
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    for segment in re.findall(r"[\u4e00-\u9fff]+", compact):
        tokens.extend(segment[index : index + 2] for index in range(max(0, len(segment) - 1)))
        tokens.extend(segment[index : index + 3] for index in range(max(0, len(segment) - 2)))
    return tokens or [compact]


def unique(items: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items:
        value = item.strip()
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


@lru_cache(maxsize=1)
def load_query_expansion_rules(path: str = str(QUERY_EXPANSIONS_PATH)) -> List[dict]:
    rules_path = Path(path)
    if not rules_path.exists():
        raise FileNotFoundError(f"Missing query expansion config: {rules_path}")
    return json.loads(rules_path.read_text(encoding="utf-8"))


def expand_queries(question: str, plan: SearchPlan) -> tuple[List[str], List[str], List[str]]:
    plan_terms = [*plan.legal_relations, *plan.claim_bases, *plan.rule_elements, *plan.legal_issues]
    text = "\n".join([question, *plan.search_queries, *plan_terms])
    queries = [question, *plan.search_queries, *plan_terms]
    categories = list(plan.preferred_categories)
    laws = list(plan.preferred_laws)
    for rule in load_query_expansion_rules():
        if any(trigger and trigger in text for trigger in rule.get("triggers", [])):
            queries.extend(rule.get("expansions", []))
            categories.extend(rule.get("categories", []))
            laws.extend(rule.get("laws", []))
    categories = [category for category in unique(categories) if category in {"民法商法", "社会法"}]
    return unique(queries), categories, unique(laws)


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


def doc_matches(doc: Document, categories: List[str] | None = None, laws: List[str] | None = None) -> bool:
    meta = doc.metadata
    return (
        (not categories or meta.get("category_name") in categories)
        and (not laws or meta.get("title") in laws)
    )


def bm25_search(
    query: str,
    k: int = 12,
    chunks_path: Path = CHUNKS_PATH,
    categories: List[str] | None = None,
    laws: List[str] | None = None,
) -> List[Document]:
    index, docs = bm25_index(str(chunks_path))
    scores = index.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda item: scores[item], reverse=True)
    result: List[Document] = []
    for doc_index in ranked:
        if scores[doc_index] <= 0:
            break
        doc = docs[doc_index]
        if doc_matches(doc, categories=categories, laws=laws):
            result.append(doc)
            if len(result) >= k:
                break
    return result


@lru_cache(maxsize=2)
def law_index(laws_path: str = str(LAWS_PATH)) -> tuple[BM25Okapi, List[dict]]:
    path = Path(laws_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing laws file: {path}")
    laws = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if row.get("status") == "有效":
                laws.append(row)
    texts = [
        "\n".join(
            [
                law.get("title", ""),
                law.get("category_name", ""),
                law.get("text", "")[:3000],
            ]
        )
        for law in laws
    ]
    return BM25Okapi([tokenize(text) for text in texts]), laws


def select_candidate_laws(
    queries: List[str],
    preferred_laws: List[str],
    categories: List[str],
    max_laws: int = 8,
    laws_path: Path = LAWS_PATH,
) -> List[str]:
    index, laws = law_index(str(laws_path))
    existing_titles = {law.get("title") for law in laws}
    selected = [law for law in preferred_laws if law in existing_titles]
    scores = [0.0 for _ in laws]
    for query in queries:
        query_scores = index.get_scores(tokenize(query))
        scores = [score + float(query_scores[index_]) for index_, score in enumerate(scores)]
    ranked = sorted(range(len(scores)), key=lambda item: scores[item], reverse=True)
    for law_index_ in ranked:
        law = laws[law_index_]
        if categories and law.get("category_name") not in categories:
            continue
        title = law.get("title")
        if title and title not in selected:
            selected.append(title)
        if len(selected) >= max_laws:
            break
    return selected


def metadata_filter(category: str | None = None, law: str | None = None) -> dict:
    clauses = [{"status": "有效"}]
    if category:
        clauses.append({"category_name": category})
    if law:
        clauses.append({"title": law})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def reciprocal_rank_fusion(groups: Iterable[List[Document]], k: int = 30, constant: int = 60) -> List[Document]:
    scores: dict[str, float] = {}
    docs_by_id: dict[str, Document] = {}
    for docs in groups:
        for rank, doc in enumerate(docs, start=1):
            chunk_id = doc.metadata.get("chunk_id") or f"{doc.metadata.get('title')}:{rank}"
            docs_by_id[chunk_id] = doc
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (constant + rank)
    return [docs_by_id[chunk_id] for chunk_id in sorted(scores, key=scores.get, reverse=True)[:k]]


def hybrid_retrieve(
    queries: List[str],
    categories: List[str],
    laws: List[str],
    persist_dir: Path = CHROMA_DIR,
    chunks_path: Path = CHUNKS_PATH,
    k_per_query: int = 8,
    candidate_k: int = 30,
) -> List[Document]:
    vectorstore = load_vectorstore(persist_directory=persist_dir)
    candidate_laws = select_candidate_laws(queries, laws, categories)
    active_queries = queries[:5]
    combined_query = "；".join(active_queries)
    groups: List[List[Document]] = []

    for query in active_queries:
        groups.append(vectorstore.similarity_search(query, k=k_per_query, filter=metadata_filter()))
        groups.append(bm25_search(query, k=k_per_query, chunks_path=chunks_path))

    for category in categories[:2]:
        groups.append(vectorstore.similarity_search(combined_query, k=k_per_query, filter=metadata_filter(category=category)))
        groups.append(bm25_search(combined_query, k=k_per_query, chunks_path=chunks_path, categories=[category]))

    for law in candidate_laws[:5]:
        groups.append(vectorstore.similarity_search(combined_query, k=4, filter=metadata_filter(law=law)))
        groups.append(bm25_search(combined_query, k=4, chunks_path=chunks_path, laws=[law]))

    return reciprocal_rank_fusion(groups, k=candidate_k)


def compact_source(doc: Document, index: int) -> str:
    meta = doc.metadata
    return "\n".join(
        [
            f"[C{index}] {meta.get('title')}｜{meta.get('article') or meta.get('unit_type')}｜{meta.get('category_name')}｜{meta.get('status')}",
            doc.page_content[:900],
        ]
    )


def format_evidence_matrix(plan: SearchPlan, docs: List[Document]) -> str:
    targets = unique([*plan.claim_bases, *plan.legal_issues])[:8]
    if not targets:
        targets = ["核心法律依据"]
    lines = ["| 争点/要件 | 支持来源 |", "|---|---|"]
    for target in targets:
        target_tokens = set(tokenize(target))
        scored: list[tuple[int, int]] = []
        for index, doc in enumerate(docs, start=1):
            meta = doc.metadata
            text = f"{meta.get('title', '')}{meta.get('article', '')}{doc.page_content}"
            score = len(target_tokens & set(tokenize(text)))
            if score:
                scored.append((score, index))
        sources = [f"[S{index}]" for _, index in sorted(scored, reverse=True)[:3]]
        lines.append(f"| {target} | {''.join(sources) or '未直接命中'} |")
    return "\n".join(lines)


def rerank_candidates(
    question: str,
    plan: SearchPlan,
    docs: List[Document],
    model: str = DEFAULT_AGENT_MODEL,
    final_k: int = 10,
) -> List[Document]:
    if not docs:
        raise RuntimeError("No retrieval candidates generated.")
    prompt = "\n\n".join(compact_source(doc, index) for index, doc in enumerate(docs, start=1))
    reranker = build_llm(model).with_structured_output(RerankResult, method="function_calling")
    result = reranker.invoke(
        [
            SystemMessage(
                content=(
                    "你是法律RAG轻量精排器。根据用户问题、请求权基础和构成要件，从候选法规片段中选择最能直接支持回答的来源。"
                    "优先覆盖每个核心要件；优先选择现行有效、条文正文、能直接支持责任构成、抗辩或救济路径的片段。"
                    f"返回最多 {final_k} 个候选编号，按相关性降序排列。"
                )
            ),
            HumanMessage(
                content=(
                    f"用户问题：{question}\n"
                    f"法律关系：{'；'.join(plan.legal_relations)}\n"
                    f"请求权基础/救济路径：{'；'.join(plan.claim_bases)}\n"
                    f"构成要件/审查要点：{'；'.join(plan.rule_elements)}\n"
                    f"法律争点：{'；'.join(plan.legal_issues)}\n\n"
                    f"候选来源：\n{prompt}"
                )
            ),
        ]
    )
    ids = unique(str(value) for value in result.ranked_source_ids)
    selected = [int(value) for value in ids if value.isdigit() and 1 <= int(value) <= len(docs)]
    if not selected:
        raise RuntimeError("Reranker returned no valid source ids.")
    return [docs[index_ - 1] for index_ in selected[:final_k]]


def format_sources(docs: List[Document]) -> str:
    blocks = []
    for index, doc in enumerate(docs, start=1):
        meta = doc.metadata
        blocks.append(
            "\n".join(
                [
                    f"[S{index}] {meta.get('title')}｜{meta.get('article') or meta.get('unit_type')}｜{meta.get('status')}",
                    f"法律部门：{meta.get('category_name')}｜公布日期：{meta.get('publish_date')}｜施行日期：{meta.get('effective_date')}",
                    doc.page_content,
                ]
            )
        )
    return "\n\n".join(blocks)


def validate_citations(answer: str, source_count: int) -> None:
    if re.search(r"\[S\d+(?:\s*[,，]\s*S?\d+)+\]", answer):
        raise RuntimeError("Generated answer uses grouped citations; cite sources separately like [S1][S2].")
    cited = {int(match) for match in re.findall(r"\[S(\d+)\]", answer)}
    if not cited:
        raise RuntimeError("Generated answer contains no source citations.")
    invalid = sorted(value for value in cited if value < 1 or value > source_count)
    if invalid:
        raise RuntimeError(f"Generated answer cites unknown sources: {invalid}")


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
4. 优先使用现行有效法规；不要引用已废止依据。
5. 引用条款名称必须完整，不得写“第六”“第七”等残缺条款；多条法条要分别完整写出。
6. 法律结论、救济路径、责任后果、期限和金额必须有来源支持；证据整理、书面沟通等实务建议不得写成法定义务或法定权利。
7. 引用必须拆开写成 [S1][S2]，不得写成 [S1, S2] 或 [S1，S2]。
8. 法律分析必须体现“大前提：法条规则；小前提：用户事实；结论：可主张事项/风险”的三段论。
9. 不得补充未在法规依据中出现的诉讼时效、管辖、执行条件、期限、比例、金额等具体法律规则。
10. 答案结构固定为：结论、证据矩阵、法律依据、法律分析、行动建议、风险提示、需要补充的事实。

用户问题：
{question}

检索规划：
法律关系：{"；".join(plan.legal_relations) or "无"}
请求权基础/救济路径：{"；".join(plan.claim_bases) or "无"}
构成要件/审查要点：{"；".join(plan.rule_elements) or "无"}
法律争点：{"；".join(plan.legal_issues)}
需要补充事实：{"；".join(plan.missing_facts) or "无"}

证据矩阵：
{format_evidence_matrix(plan, docs)}

法规依据：
{format_sources(docs)}
"""
    answer = str(build_llm(model).invoke(prompt).content)
    validate_citations(answer, len(docs))
    return answer


def review_answer(question: str, docs: List[Document], answer: str, model: str = DEFAULT_AGENT_MODEL) -> None:
    reviewer = build_llm(model).with_structured_output(AnswerReview, method="function_calling")
    result = reviewer.invoke(
        [
            SystemMessage(
                content=(
                    "你是法律RAG答案审查器。严格检查答案中的法律结论、救济路径、责任后果、期限、金额、"
                    "法律名称和条款是否均由给定来源支持，且关键法律结论是否有来源编号。"
                    "来源能够概括支持即可，不要求逐字一致。"
                    "只有存在实质性无来源法律结论、引用不存在来源、编造法律/条款、关键条件或法律后果错误时，"
                    "approved 才能为 false。"
                    "不要因为表达还能更严谨、来源对应关系还可更清晰、使用“通常/一般/可能”等审慎措辞而拒绝。"
                    "如果答案写出来源中没有的诉讼时效、管辖、执行条件、具体期限、比例或金额，应判定为不通过。"
                    "证据整理、书面沟通、咨询律师等普通实务建议，如果没有被表述为法定义务、法定权利或确定法律后果，"
                    "不应作为未获来源支持的问题。"
                )
            ),
            HumanMessage(
                content=(
                    f"用户问题：{question}\n\n"
                    f"来源：\n{format_sources(docs)}\n\n"
                    f"答案：\n{answer}"
                )
            ),
        ]
    )
    if not result.approved:
        raise RuntimeError("Answer failed post-generation review: " + "；".join(result.problems))


def answer_question(
    question: str,
    model: str = DEFAULT_AGENT_MODEL,
    persist_dir: Path = CHROMA_DIR,
    chunks_path: Path = CHUNKS_PATH,
) -> str:
    plan = plan_question(question, model=model)
    if plan.needs_clarification:
        return format_clarifying_questions(plan)
    queries, categories, laws = expand_queries(question, plan)
    candidates = hybrid_retrieve(queries, categories, laws, persist_dir=persist_dir, chunks_path=chunks_path)
    docs = rerank_candidates(question, plan, candidates, model=model)
    answer = generate_answer(question, plan, docs, model=model)
    review_answer(question, docs, answer, model=model)
    return answer
