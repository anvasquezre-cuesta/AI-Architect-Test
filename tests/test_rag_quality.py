"""Automated RAG retrieval-quality gate.

Runs the retrieval half of the evaluation harness (no LLM needed) against a live
Qdrant and asserts hit-rate@k and MRR stay above a floor. Skipped automatically when
Qdrant is not reachable, so the normal unit-test run stays hermetic and fast.
"""

import glob
import json
import os
import socket

import pytest

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))


def _qdrant_reachable() -> bool:
    try:
        with socket.create_connection((QDRANT_HOST, QDRANT_PORT), timeout=1):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _qdrant_reachable(),
    reason="Qdrant not reachable; retrieval-quality gate needs a live vector store",
)


def test_retrieval_quality_meets_threshold():
    from eval.rag_eval import EVAL_SET, SAMPLE_PDFS, evaluate, reset_and_ingest

    with open(EVAL_SET) as f:
        eval_set = json.load(f)

    reset_and_ingest(sorted(glob.glob(SAMPLE_PDFS)))
    summary = evaluate(eval_set, rag=None)["summary"]

    assert summary["hit_rate_at_k"] >= 0.9, (
        f"Retrieval hit-rate@{summary['top_k']} regressed: {summary['hit_rate_at_k']}"
    )
    assert summary["mrr"] >= 0.55, f"Retrieval MRR regressed: {summary['mrr']}"
