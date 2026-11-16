"""在隔离的合成库校准重排阈值，并用保留问题检验；不读写用户知识库。"""
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ingest import ingest_file
from rag import KnowledgeBase, select_context
from rerank import reranker


def evidence_label(question, hit):
    # 这是受控合成数据的证据代理标注：来源正确且包含至少一个人工标出的已知事实。
    # 不把“同一文档中出现过”直接等同于“这个块支持答案”。
    return hit["source"] in question["expected_sources"] and any(
        fact in hit["text"] for fact in question["supported_facts"])


def metrics(rows, threshold):
    tp = fp = fn = 0
    answered = supported = refused = negative_count = 0
    for row in rows:
        selected = select_context([h for h in row["ranked"] if h["rerank_score"] >= threshold], 4)
        if row["question"]["type"] == "unanswerable":
            negative_count += 1
            refused += not selected
        else:
            supported += 1
            answered += any(evidence_label(row["question"], hit) for hit in selected)
        for hit in row["ranked"]:
            positive = evidence_label(row["question"], hit)
            accepted = hit["rerank_score"] >= threshold
            tp += positive and accepted
            fp += not positive and accepted
            fn += positive and not accepted
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {"threshold": threshold, "evidence_precision": precision, "evidence_recall": recall,
            "evidence_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
            "supported_questions_retained": answered, "supported_question_count": supported,
            "unanswerable_questions_refused": refused, "unanswerable_question_count": negative_count}


def evaluate():
    fixture = json.loads((ROOT / "tests/fixtures/retrieval_cases.json").read_text(encoding="utf-8"))
    validation = json.loads((ROOT / "tests/fixtures/rerank_validation.json").read_text(encoding="utf-8"))
    (ROOT / ".cache").mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="rerank-evaluation-", dir=ROOT / ".cache"))
    kb = KnowledgeBase(root / "chroma", root / "knowledge")
    for document in fixture["documents"]:
        path = root / document["source"]
        path.write_text(document["text"], encoding="utf-8")
        ingest_file(path, kb)
    rows = []
    for split, questions in [("calibration", fixture["questions"]), ("validation", validation["questions"])]:
        for question in questions:
            candidates = kb.search_candidates(question["query"], 20, "hybrid")
            ranked = reranker.score(question["query"], candidates[:20])
            rows.append({"split": split, "question": question, "ranked": ranked})
            print(f"{split} {question['id']}: best={ranked[0]['rerank_score']:.4f}", flush=True)
    calibration_rows = [row for row in rows if row["split"] == "calibration"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    sweep = [metrics(calibration_rows, i / 100) for i in range(5, 96, 5)]
    # 只用 calibration 选阈值；validation 不参与选参。先最大化证据 F1，再偏向精确率。
    chosen = max(sweep, key=lambda item: (item["evidence_f1"], item["evidence_precision"], item["threshold"]))
    report = {"model": "Xenova/bge-reranker-base INT8 CPU", "chosen_threshold": chosen["threshold"],
              "calibration": chosen, "validation": metrics(validation_rows, chosen["threshold"]),
              "sweep": sweep, "rows": rows,
              "limitations": "Small synthetic corpus, held-out questions share documents. Evidence proxy metrics, not LLM answer accuracy or probability calibration. Real user documents still require calibration."}
    output = root / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["chosen_threshold", "calibration", "validation"]}, ensure_ascii=False, indent=2))
    print(f"Report: {output}")
    return report


if __name__ == "__main__":
    evaluate()
