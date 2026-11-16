"""Deterministic regression checks for retrieval fusion, cache and reindexing."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ingest import ingest_file, make_chunks
from rag import KnowledgeBase, tokenize, select_context, candidate_table


class HybridTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".cache").mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="hybrid-tests-", dir=ROOT / ".cache"))
        self.kb = KnowledgeBase(self.root / "chroma", self.root / "knowledge")
        # Controlled vector ordering lets the test distinguish the two retrieval channels.
        self.kb.collection.add(ids=["exact", "semantic", "unrelated"],
            documents=["错误码 E1024 在 decode_instruction 中表示非法操作码。",
                       "指令解析失败后，可以重置译码器并重新启动。", "玫瑰花每周浇水一次。"],
            metadatas=[{"source": name, "location": "正文"}
                       for name in ["错误码.md", "恢复.md", "园艺.md"]],
            embeddings=[[0.8, 0.6], [1.0, 0.0], [0.0, 1.0]])

    def test_identifiers_and_chinese_tokenization(self):
        tokens = tokenize("请问 decode_instruction 的 E1024 和 E-103 怎么处理？")
        self.assertIn("decode_instruction", tokens)
        self.assertIn("e1024", tokens)
        self.assertIn("e-103", tokens)
        self.assertNotIn("的", tokens)
        self.assertIn("冷却液", tokenize("冷却液多久更换"))

    def test_bm25_works_without_ollama_and_has_no_unmatched_scores(self):
        with patch.object(self.kb, "embed", side_effect=AssertionError("must not use embedding")):
            hits = self.kb.search("E1024", mode="bm25")
            self.assertEqual([h["id"] for h in hits], ["exact"])
            self.assertGreater(hits[0]["bm25_score"], 0)
            self.assertIsNone(hits[0]["similarity"])
            self.assertEqual(self.kb.search("ZXQ999888", mode="bm25"), [])

    def test_rrf_combines_distinct_rankings_and_deduplicates(self):
        with patch.object(self.kb, "embed", return_value=[[1.0, 0.0]]):
            vector = self.kb.search("E1024", top_k=3, mode="vector")
            hybrid = self.kb.search("E1024", top_k=3, mode="hybrid")
        self.assertEqual(vector[0]["id"], "semantic")
        self.assertEqual(hybrid[0]["id"], "exact")
        self.assertEqual(len({h["id"] for h in hybrid}), len(hybrid))
        self.assertAlmostEqual(hybrid[0]["rrf_score"], 1 / 62 + 1 / 61)
        self.assertEqual(hybrid[0]["vector_rank"], 2)
        self.assertEqual(hybrid[0]["bm25_rank"], 1)
        table = candidate_table(hybrid, 1)
        self.assertEqual(table[0][1], "资料1")
        self.assertEqual(table[1][1], "未送入回答")

    def test_lexical_cache_reused_then_invalidated_on_import(self):
        self.kb.search("E1024", mode="bm25")
        original = self.kb._bm25_cache
        self.kb.search("decode_instruction", mode="bm25")
        self.assertIs(self.kb._bm25_cache, original)
        source = self.root / "新型号.md"
        source.write_text("型号 XJ900 的供电电压为 12V。", encoding="utf-8")
        with patch.object(self.kb, "embed", return_value=[[1.0, 0.0]]):
            ingest_file(source, self.kb)
        self.assertIsNone(self.kb._bm25_cache)
        self.assertEqual(self.kb.search("XJ900", mode="bm25")[0]["source"], source.name)

    def test_chunk_profile_change_reindexes_unchanged_document(self):
        source = self.root / "分块.md"
        source.write_text("冷却系统需要维护。" * 200, encoding="utf-8")
        def embeddings(texts):
            return [[1.0, 0.0] for _ in texts]
        with patch.object(self.kb, "embed", side_effect=embeddings):
            ingest_file(source, self.kb, 1200, 180)
            old = self.kb.collection.get(where={"source": source.name})
            ingest_file(source, self.kb, 400, 40)
            new = self.kb.collection.get(where={"source": source.name})
            self.assertGreater(len(new["ids"]), len(old["ids"]))
            self.assertFalse(set(old["ids"]) & set(new["ids"]))
            self.assertIn("跳过", ingest_file(source, self.kb, 400, 40))
        self.assertTrue(all(m["chunk_chars"] == 400 for m in new["metadatas"]))

    def test_failed_profile_change_keeps_old_ids_and_lexical_results(self):
        source = self.root / "保留.md"
        source.write_text("保留型号 KEEP123 的原始记录。", encoding="utf-8")
        with patch.object(self.kb, "embed", return_value=[[1.0, 0.0]]):
            ingest_file(source, self.kb)
        old = self.kb.collection.get(where={"source": source.name})["ids"]
        self.kb.search("KEEP123", mode="bm25")
        with patch.object(self.kb, "embed", side_effect=RuntimeError("offline")):
            with self.assertRaises(RuntimeError):
                ingest_file(source, self.kb, 400, 40)
        self.assertEqual(old, self.kb.collection.get(where={"source": source.name})["ids"])
        self.assertEqual(self.kb.search("KEEP123", mode="bm25")[0]["source"], source.name)

    def test_invalid_chunk_settings_and_total_context_budget(self):
        with self.assertRaises(ValueError):
            make_chunks([("正文", "正文")], 400, 200)
        candidates = [{"text": "x" * 4000} for _ in range(10)]
        self.assertEqual(len(select_context(candidates, 10)), 3)
        self.assertEqual(len(select_context(candidates, 2)), 2)


if __name__ == "__main__":
    unittest.main()
