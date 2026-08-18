"""RAG quality evaluation harness.

Measures two things a plain "did it return something" check misses:

- Retrieval quality: for each question with a known source passage, is that passage
  among the top-k retrieved chunks (hit-rate@k), and how high (mean reciprocal rank)?
  This needs no LLM and is deterministic given the embedding model.
- Answer quality: if an LLM is configured, does the generated answer contain the known
  facts (keyword correctness)?

Run against a live Qdrant (and, for answers, a live OpenAI-compatible LLM such as a
local Ollama). Reusable for model comparison: point --base-url / --model at different
servers and compare the summaries.

    python -m eval.rag_eval                       # retrieval + answers (local Ollama)
    python -m eval.rag_eval --no-llm              # retrieval metrics only
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from app import config
from app.services import vector_service
from app.services.embedding_service import get_query_embedding
from app.services.ocr_service import extract_text_from_pdf
from app.services.rag_service import RAGService, chunk_text

EVAL_SET = os.path.join(os.path.dirname(__file__), "rag_eval_set.json")
SAMPLE_PDFS = os.path.join(os.path.dirname(__file__), "..", "sample_pdfs", "*.pdf")


def _normalize(text: str) -> str:
    """Whitespace- and case-insensitive form for robust substring checks over OCR text."""
    return " ".join(text.lower().split())


def reset_and_ingest(pdf_paths: list[str]) -> int:
    """Wipe the collection and ingest the given PDFs fresh, so runs are comparable."""
    try:
        vector_service.client.delete_collection(config.QDRANT_COLLECTION)
    except Exception:
        pass
    vector_service.init_collection()
    total = 0
    for path in pdf_paths:
        text = extract_text_from_pdf(path)
        chunks = chunk_text(text)
        source = os.path.basename(path)
        vector_service.store_document_chunks(
            chunks, metadata=[{"source": source}] * len(chunks)
        )
        total += len(chunks)
    return total


def evaluate(eval_set: dict, rag: RAGService | None) -> dict:
    """Run every question; return per-question rows and aggregate metrics."""
    top_k = eval_set.get("top_k", 5)
    rows = []
    for item in eval_set["questions"]:
        question = item["question"]
        query_vec = get_query_embedding(question)
        hits = vector_service.search_similar(query_vec, top_k=top_k)

        expected_source = _normalize(item["expected_source"])
        rank = 0
        for i, hit in enumerate(hits, start=1):
            if expected_source in _normalize(hit["text"]):
                rank = i
                break
        reciprocal_rank = 1.0 / rank if rank else 0.0

        answer, answer_correct = None, None
        if rag is not None:
            answer = rag.generate_answer(question)["answer"]
            norm_answer = _normalize(answer)
            answer_correct = all(
                _normalize(kw) in norm_answer for kw in item["expected_answer"]
            )

        rows.append(
            {
                "question": question,
                "hit": rank > 0,
                "rank": rank,
                "reciprocal_rank": reciprocal_rank,
                "answer_correct": answer_correct,
                "answer": answer,
            }
        )

    n = len(rows)
    answered = [r for r in rows if r["answer_correct"] is not None]
    summary = {
        "n_questions": n,
        "top_k": top_k,
        "hit_rate_at_k": round(sum(r["hit"] for r in rows) / n, 3),
        "mrr": round(sum(r["reciprocal_rank"] for r in rows) / n, 3),
        "answer_accuracy": (
            round(sum(r["answer_correct"] for r in answered) / len(answered), 3)
            if answered
            else None
        ),
    }
    return {"summary": summary, "rows": rows}


def _print_report(result: dict) -> None:
    print(f"{'hit':>4} {'rank':>4} {'ans':>4}  question")
    for r in result["rows"]:
        ans = "-" if r["answer_correct"] is None else ("ok" if r["answer_correct"] else "X")
        hit = "ok" if r["hit"] else "X"
        print(f"{hit:>4} {r['rank']:>4} {ans:>4}  {r['question']}")
    s = result["summary"]
    print("\nsummary:")
    print(f"  questions        : {s['n_questions']}")
    print(f"  hit-rate@{s['top_k']}      : {s['hit_rate_at_k']}")
    print(f"  MRR              : {s['mrr']}")
    print(f"  answer accuracy  : {s['answer_accuracy']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG quality evaluation")
    parser.add_argument("--no-llm", action="store_true", help="retrieval metrics only")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "qwen2.5:3b"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "ollama"))
    args = parser.parse_args()

    with open(EVAL_SET) as f:
        eval_set = json.load(f)

    n_chunks = reset_and_ingest(sorted(glob.glob(SAMPLE_PDFS)))
    print(f"ingested {n_chunks} chunks from sample PDFs\n")

    rag = None
    if not args.no_llm:
        rag = RAGService(
            embed_query=get_query_embedding,
            retriever=vector_service.search_similar,
            api_key=args.api_key,
            model=args.model,
            base_url=args.base_url,
        )

    result = evaluate(eval_set, rag)
    _print_report(result)


if __name__ == "__main__":
    main()
