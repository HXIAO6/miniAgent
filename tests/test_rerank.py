"""阈值、无证据拒答、引用编号等边界使用固定分数测试，避免依赖模型偶然输出。"""
import asyncio
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rag import KnowledgeBase, candidate_table


class RerankPipelineTests(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase()
        self.hits = [dict(id=str(i), source=f"{i}.md", location="正文", text="测试资料",
                          rrf_score=0.03 - i * 0.001) for i in range(3)]

    def scorer(self, scores):
        scorer = Mock()
        scorer.score.side_effect = lambda query, hits: sorted(
            [dict(hit, rerank_score=score) for hit, score in zip(hits, scores)],
            key=lambda hit: -hit["rerank_score"])
        return scorer

    def test_rerank_filter_and_reference_numbering(self):
        with patch.object(self.kb, "search_candidates", return_value=self.hits):
            result = self.kb.retrieve("问题", top_k=2, threshold=.5, scorer=self.scorer([.2, .9, .6]))
        self.assertEqual([h["id"] for h in result["selected"]], ["1", "2"])
        table = candidate_table(result["candidates"], 2, result["selected"])
        self.assertEqual([row[1] for row in table], ["资料1", "资料2", "未送入回答"])
        self.assertEqual(table[-1][11], "低于阈值")

    def test_all_filtered_returns_zero_evidence(self):
        with patch.object(self.kb, "search_candidates", return_value=self.hits):
            result = self.kb.retrieve("无答案", threshold=.8, scorer=self.scorer([.1, .2, .3]))
        self.assertEqual(result["selected"], [])
        self.assertTrue(all(h["decision"] == "低于阈值" for h in result["candidates"]))

    def test_model_failure_is_visible_not_silently_ignored(self):
        scorer = Mock()
        scorer.score.side_effect = RuntimeError("model missing")
        with patch.object(self.kb, "search_candidates", return_value=self.hits):
            with self.assertRaisesRegex(RuntimeError, "model missing"):
                self.kb.retrieve("问题", scorer=scorer)

    def test_disabled_rerank_ignores_threshold_and_never_loads_model(self):
        scorer = Mock()
        with patch.object(self.kb, "search_candidates", return_value=self.hits):
            result = self.kb.retrieve("问题", top_k=2, use_rerank=False, threshold=1, scorer=scorer)
        self.assertEqual(len(result["selected"]), 2)
        scorer.score.assert_not_called()

    def test_invalid_threshold_rejected(self):
        for threshold in [float("nan"), -0.1, 1.1]:
            with self.assertRaises(ValueError):
                self.kb.retrieve("问题", threshold=threshold)

    def test_strict_mode_refuses_without_llm_or_history_compression(self):
        import knowledge_assistant as app
        (ROOT / ".cache").mkdir(exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="refusal-tests-", dir=ROOT / ".cache"))
        async def collect():
            return [result async for result in app.predict("无资料的问题", [], True)]
        with patch.object(app, "HISTORY_FILE", str(root / "history.json")), \
             patch.object(app.knowledge_base, "retrieve", return_value={"candidates": [], "selected": []}), \
             patch.object(app.Runner, "run_streamed") as run, \
             patch.object(app, "compress_history") as compress:
            results = asyncio.run(collect())
        run.assert_not_called()
        compress.assert_not_called()
        self.assertIn("没有找到足够相关", results[-1][1][-1]["content"])


if __name__ == "__main__":
    unittest.main()
