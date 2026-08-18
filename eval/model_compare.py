"""Compare candidate LLMs on the RAG task and pick one with data, not by taste.

Retrieval is identical across models (same embeddings and vector store), so it is not
a selection signal. Models are judged on what actually differs between them:

- answer_accuracy: fraction of questions whose answer contains the known facts. This is
  the end-user-visible quality and the primary decision metric.
- mean_latency_s: average answer time. Secondary, a speed/cost proxy.

Each model is one MLflow run logged to local files (./mlruns), so the comparison is
reproducible and browsable with `mlflow ui` without running any server.

    python -m eval.model_compare                          # default 3 local models
    python -m eval.model_compare --models qwen2.5:3b gemma2:2b
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import mlflow

from app.services import vector_service
from app.services.embedding_service import get_query_embedding
from app.services.rag_service import RAGService
from eval.rag_eval import EVAL_SET, SAMPLE_PDFS, evaluate, reset_and_ingest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_MODELS = ["qwen2.5:3b", "gemma2:2b", "phi3.5"]


def compare(models: list[str], base_url: str, api_key: str) -> list[dict]:
    with open(EVAL_SET) as f:
        eval_set = json.load(f)

    # Ingest once; every model queries the same corpus.
    n_chunks = reset_and_ingest(sorted(glob.glob(SAMPLE_PDFS)))
    print(f"ingested {n_chunks} chunks; comparing {len(models)} models\n")

    mlflow.set_tracking_uri("file:" + os.path.abspath(os.path.join(REPO_ROOT, "mlruns")))
    mlflow.set_experiment("rag-model-comparison")

    results = []
    for model in models:
        rag = RAGService(
            embed_query=get_query_embedding,
            retriever=vector_service.search_similar,
            api_key=api_key,
            model=model,
            base_url=base_url,
        )
        summary = evaluate(eval_set, rag)["summary"]
        with mlflow.start_run(run_name=model):
            mlflow.log_param("model", model)
            mlflow.log_param("base_url", base_url)
            mlflow.log_param("n_questions", summary["n_questions"])
            mlflow.log_metric("answer_accuracy", summary["answer_accuracy"])
            mlflow.log_metric("mean_latency_s", summary["mean_latency_s"])
            mlflow.log_metric("hit_rate_at_k", summary["hit_rate_at_k"])
            mlflow.log_metric("mrr", summary["mrr"])
        results.append({"model": model, **summary})
        print(
            f"  {model:<14} accuracy={summary['answer_accuracy']}  "
            f"mean_latency={summary['mean_latency_s']}s"
        )
    return results


def _pick_winner(results: list[dict]) -> dict:
    # Highest accuracy; break ties by lower latency.
    return sorted(
        results, key=lambda r: (-r["answer_accuracy"], r["mean_latency_s"])
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LLMs on the RAG task")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "ollama"))
    args = parser.parse_args()

    results = compare(args.models, args.base_url, args.api_key)
    winner = _pick_winner(results)
    print(f"\nwinner: {winner['model']} "
          f"(accuracy={winner['answer_accuracy']}, mean_latency={winner['mean_latency_s']}s)")
    print("browse runs with: mlflow ui --backend-store-uri ./mlruns")


if __name__ == "__main__":
    main()
