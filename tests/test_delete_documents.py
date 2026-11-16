"""文档删除的一致性与路径边界测试；只操作 .cache 下的隔离测试库。"""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rag import KnowledgeBase


class DocumentDeletionTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".cache").mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="deletion-tests-", dir=ROOT / ".cache"))
        self.kb = KnowledgeBase(self.root / "data/chroma", self.root / "knowledge")
        self.kb.knowledge_dir.mkdir()
        self.source = "维护.md"
        (self.kb.knowledge_dir / self.source).write_text("冷却液每九天更换。", encoding="utf-8")
        (self.kb.knowledge_dir / "保留.md").write_text("保留此文档。", encoding="utf-8")
        self.kb.collection.add(ids=["delete-1", "delete-2", "keep"],
            documents=["型号 DEL123 冷却液每九天更换。", "型号 DEL123 维护前关机。", "型号 KEEP456 保留此文档。"],
            embeddings=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
            metadatas=[{"source": source, "location": "正文"} for source in [self.source, self.source, "保留.md"]])
        # 预热缓存，确保测试能够发现“数据库删了但 BM25 还命中”的回归。
        self.kb.search("DEL123", mode="bm25")

    def test_delete_removes_both_retrieval_channels_and_keeps_recovery_copy(self):
        archive = self.kb.knowledge_dir / ".versions/hash"
        archive.mkdir(parents=True)
        (archive / self.source).write_text("旧版本保留", encoding="utf-8")
        result = self.kb.delete_document(self.source)
        batch = Path(result["trash_path"])
        self.assertEqual(result["chunk_count"], 2)
        self.assertFalse((self.kb.knowledge_dir / self.source).exists())
        self.assertTrue((batch / self.source).is_file())
        record = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "deleted")
        self.assertEqual(len(record["snapshot"]["embeddings"]), 2)
        self.assertIsNone(self.kb._bm25_cache)
        self.assertEqual(self.kb.search("DEL123", mode="bm25"), [])
        with patch.object(self.kb, "embed", return_value=[[1.0, 0.0]]):
            self.assertEqual([h["source"] for h in self.kb.search("DEL123", mode="vector")], ["保留.md"])
        self.assertEqual([d["source"] for d in self.kb.list_documents()], ["保留.md"])
        self.assertTrue((archive / self.source).exists())

    def test_missing_original_can_still_remove_residual_index(self):
        # 模拟用户手工移走原文，不使用删除命令操作正式资料。
        (self.kb.knowledge_dir / self.source).rename(self.root / "moved.md")
        result = self.kb.delete_document(self.source)
        self.assertFalse(result["file_recycled"])
        self.assertEqual(self.kb.collection.count(), 1)

    def test_unindexed_original_can_be_recycled(self):
        (self.kb.knowledge_dir / "未索引.txt").write_text("保留原文", encoding="utf-8")
        result = self.kb.delete_document("未索引.txt")
        self.assertEqual(result["chunk_count"], 0)
        self.assertTrue((Path(result["trash_path"]) / "未索引.txt").exists())

    def test_move_failure_does_not_touch_index(self):
        real_rename = Path.rename
        def fail_original(path, destination):
            if path == self.kb.knowledge_dir / self.source:
                raise PermissionError("文件被占用")
            return real_rename(path, destination)
        with patch.object(Path, "rename", fail_original):
            with self.assertRaisesRegex(RuntimeError, "文件被占用"):
                self.kb.delete_document(self.source)
        self.assertEqual(self.kb.collection.count(), 3)
        self.assertTrue((self.kb.knowledge_dir / self.source).exists())

    def test_partial_database_delete_rolls_back_vectors_and_file(self):
        collection = self.kb.collection
        real_delete = collection.delete
        def delete_one_then_fail(*args, **kwargs):
            real_delete(ids=[kwargs["ids"][0]])
            raise RuntimeError("模拟数据库中途失败")
        with patch.object(collection, "delete", side_effect=delete_one_then_fail):
            with self.assertRaisesRegex(RuntimeError, "原文与索引已恢复"):
                self.kb.delete_document(self.source)
        self.assertEqual(self.kb.collection.count(), 3)
        self.assertTrue((self.kb.knowledge_dir / self.source).exists())
        self.assertTrue(self.kb.search("DEL123", mode="bm25"))
        record = next((self.root / "data/trash").glob("*/manifest.json"))
        self.assertEqual(json.loads(record.read_text(encoding="utf-8"))["state"], "rolled_back")

    def test_rollback_failure_reports_preserved_snapshot(self):
        collection = self.kb.collection
        with patch.object(collection, "delete", side_effect=RuntimeError("删除失败")), \
             patch.object(collection, "upsert", side_effect=RuntimeError("恢复失败")):
            with self.assertRaisesRegex(RuntimeError, "索引恢复失败"):
                self.kb.delete_document(self.source)
        manifest = next((self.root / "data/trash").glob("*/manifest.json"))
        record = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "rollback_failed")
        self.assertEqual(len(record["snapshot"]["ids"]), 2)
        self.assertTrue((self.kb.knowledge_dir / self.source).exists())

    def test_path_traversal_and_unknown_names_are_rejected(self):
        for name in ["../维护.md", "..\\维护.md", "D:\\other.md", "维护.md:stream", ".env", "missing.md"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.kb.delete_document(name)
        self.assertEqual(self.kb.collection.count(), 3)

    def test_ui_requires_confirmation_and_resets_it_on_selection_change(self):
        import knowledge_assistant as app
        with patch.object(app, "knowledge_base", self.kb):
            result = asyncio.run(app.delete_documents([self.source], False))
            self.assertIn("勾选确认", result[0])
            self.assertEqual(self.kb.collection.count(), 3)
            preview, checked = app.preview_deletion([self.source])
            self.assertIn(self.source, preview)
            self.assertFalse(checked)
            result = asyncio.run(app.delete_documents([self.source], True))
        self.assertIn("已删除", result[0])
        self.assertEqual(result[-1], [])
        self.assertFalse(result[4])
        self.assertEqual(len(result[1]), 1)


if __name__ == "__main__":
    unittest.main()
