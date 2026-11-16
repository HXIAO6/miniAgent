"""Compare retrieval modes on a small labeled corpus without changing production data."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ingest import ingest_file
from rag import KnowledgeBase, select_context


def evaluate(top_k=4, candidate_k=20):
    fixture = json.loads((ROOT / "tests/fixtures/retrieval_cases.json").read_text(encoding="utf-8"))
    (ROOT / ".cache").mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="retrieval-evaluation-", dir=ROOT / ".cache"))
    kb = KnowledgeBase(root / "chroma", root / "knowledge")
    for document in fixture["documents"]:
        path = root / document["source"]
        path.write_text(document["text"], encoding="utf-8")
        ingest_file(path, kb)
    report = {"fixture": "tests/fixtures/retrieval_cases.json", "top_k": top_k,
              "candidate_k": candidate_k, "rows": [], "summary": {},
              "limitations": "Synthetic retrieval baseline, not answer accuracy. No threshold or reranker; unanswerable questions may still return candidates. Latency includes warmup/cache effects."}
    for mode in ["vector", "bm25", "hybrid"]:
        recalls, reciprocal_ranks = [], []
        unanswered_with_candidates = 0
        for question in fixture["questions"]:
            started = time.perf_counter()
            candidates = kb.search_candidates(question["query"], candidate_k, mode)
            selected = select_context(candidates, top_k)
            expected = set(question["expected_sources"])
            found = {hit["source"] for hit in selected}
            recall = len(expected & found) / len(expected) if expected else None
            rank = next((i for i, hit in enumerate(selected, 1) if hit["source"] in expected), None)
            if expected:
                recalls.append(recall)
                reciprocal_ranks.append(1 / rank if rank else 0)
            elif candidates:
                unanswered_with_candidates += 1
            report["rows"].append(dict(question, mode=mode, source_recall=recall,
                first_relevant_rank=rank, candidate_count=len(candidates),
                elapsed_seconds=round(time.perf_counter() - started, 4),
                selected=[{key: hit.get(key) for key in ["source", "location", "text", "similarity",
                           "bm25_score", "rrf_score", "vector_rank", "bm25_rank"]} for hit in selected]))
        report["summary"][mode] = {
            "mean_expected_source_recall": sum(recalls) / len(recalls),
            "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
            "unanswerable_questions_with_candidates": unanswered_with_candidates,
            "unanswerable_question_count": sum(q["type"] == "unanswerable" for q in fixture["questions"]),
        }
    output = root / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--candidate-k", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.top_k <= 10 or not args.top_k <= args.candidate_k <= 100:
        parser.error("Require 1 <= top-k <= 10 and top-k <= candidate-k <= 100")
    evaluate(args.top_k, args.candidate_k)
