"""真实 ONNX 模型测试：排序方向、长块窗口和长问题限制。"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rerank import reranker, DEFAULT_MODEL_DIR


@unittest.skipUnless((DEFAULT_MODEL_DIR / "onnx/model_int8.onnx").is_file(), "需要先下载重排模型")
class RealRerankerTests(unittest.TestCase):
    def test_relevant_text_scores_above_unrelated_text(self):
        rows = [dict(id="good", source="维护.md", text="星舟设备的冷却液每隔九天更换一次。", rrf_score=.02),
                dict(id="bad", source="园艺.md", text="玫瑰花每周浇水一次。", rrf_score=.03)]
        ranked = reranker.score("星舟设备多久换一次冷却液？", rows)
        self.assertEqual(ranked[0]["id"], "good")
        self.assertGreater(ranked[0]["rerank_score"], ranked[1]["rerank_score"])
        self.assertTrue(all(0 <= h["rerank_score"] <= 1 for h in ranked))

    def test_long_text_scores_overflow_windows_including_end(self):
        row = dict(id="long", source="记录.md", rrf_score=.02,
                   text="玫瑰花的土壤需要透气。" * 100 + "\n星舟设备的冷却液每隔九天更换一次。")
        ranked = reranker.score("星舟设备多久换一次冷却液？", [row])
        self.assertGreater(ranked[0]["rerank_windows"], 1)
        self.assertGreater(ranked[0]["rerank_score"], .2)
        long_question = "请告诉我星舟设备的冷却液更换周期。" * 40
        result = reranker.score(long_question, [dict(row, text="星舟设备每九天更换冷却液。")])
        self.assertTrue(result[0]["query_truncated"])


if __name__ == "__main__":
    unittest.main()
