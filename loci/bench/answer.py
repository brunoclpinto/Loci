import time
from dataclasses import dataclass

import httpx
from sqlalchemy.orm import Session

from loci.config import LociSettings
from loci.embeddings.client import EmbeddingClient
from loci.mcp_server.models import SearchResult
from loci.mcp_server.tools.search_knowledge import run_search
from loci.vectorstore.qdrant_client import VectorStore

ANSWER_SYSTEM_PROMPT = (
    "Answer the question using ONLY the knowledge provided below. Be concise — "
    "a short phrase or one sentence is preferred over a paragraph. If the "
    'provided knowledge does not contain the answer, reply exactly: "I don\'t '
    'know." Do not use outside knowledge.'
)


@dataclass
class AnswerResult:
    answer: str
    retrieval_time_s: float
    generation_time_s: float
    search_result: SearchResult


def _format_context(result: SearchResult) -> str:
    lines: list[str] = []
    for hit in result.chunk_hits:
        lines.append(f"- {hit.text} (source: {hit.citation.source_ref}, context: {hit.context_name})")
    for hit in result.entity_hits:
        attrs = ", ".join(f"{k}={v}" for k, v in hit.attributes.items())
        suffix = f": {attrs}" if attrs else ""
        lines.append(f"- {hit.canonical_name} ({hit.type}{suffix})")
    return "\n".join(lines) if lines else "(no relevant knowledge found)"


def answer_question(
    settings: LociSettings,
    session: Session,
    question: str,
    answer_model: str,
    context_names: list[str],
    embedding_client: EmbeddingClient | None = None,
    vector_store: VectorStore | None = None,
) -> AnswerResult:
    """Retrieval + concise-answer generation, reusing the same retrieval
    logic the search_knowledge MCP tool uses (via run_search) so bench
    results reflect the real product. This is bench-only tooling — it does
    not become a permanent MCP tool, keeping the product's
    supplier-not-assistant contract (structured data out, not prose)
    unaffected."""
    t0 = time.monotonic()
    result = run_search(
        settings,
        session,
        question,
        context_names=context_names,
        top_k=8,
        embedding_client=embedding_client,
        vector_store=vector_store,
    )
    retrieval_time_s = time.monotonic() - t0

    context_block = _format_context(result)
    t1 = time.monotonic()
    response = httpx.post(
        f"{settings.ollama.base_url}/api/chat",
        json={
            "model": answer_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Knowledge:\n{context_block}\n\nQuestion: {question}"},
            ],
            "options": {"temperature": 0},
        },
        timeout=settings.ollama.request_timeout_s,
    )
    response.raise_for_status()
    answer = response.json()["message"]["content"].strip()
    generation_time_s = time.monotonic() - t1

    return AnswerResult(
        answer=answer,
        retrieval_time_s=retrieval_time_s,
        generation_time_s=generation_time_s,
        search_result=result,
    )
