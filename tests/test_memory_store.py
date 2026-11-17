"""SQLite 短期记忆迁移与事务测试。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memory_store import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="memory-tests-", dir=ROOT / ".cache"))

    def test_migrates_legacy_json_once_and_keeps_source(self):
        legacy = self.root / "knowledge_history.json"
        legacy.write_text(json.dumps([
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么可以帮你？"},
            {"role": "tool", "content": "不应导入"},
            {"role": "assistant", "content": ""},
        ], ensure_ascii=False), encoding="utf-8")
        store = MemoryStore(self.root / "memory.sqlite3", legacy)
        self.assertEqual(store.load_messages(), [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么可以帮你？"},
        ])
        store.append_turn("第二个问题", "第二个回答")
        legacy.write_text("损坏后也不应重复迁移", encoding="utf-8")
        reopened = MemoryStore(self.root / "memory.sqlite3", legacy)
        self.assertEqual(reopened.message_count(), 4)
        self.assertTrue(legacy.exists())

    def test_append_turn_is_atomic_and_clear_keeps_database(self):
        store = MemoryStore(self.root / "memory.sqlite3")
        store.append_turn("问题", "回答")
        self.assertEqual(store.message_count(), 2)
        store.append_turn("", "不应写入")
        self.assertEqual(store.message_count(), 2)
        store.clear()
        self.assertEqual(store.message_count(), 0)
        self.assertTrue((self.root / "memory.sqlite3").exists())
        self.assertTrue((self.root / "memory.sqlite3").is_file())

    def test_invalid_json_does_not_mark_migration_complete(self):
        legacy = self.root / "history.json"
        legacy.write_text("{bad", encoding="utf-8")
        store = MemoryStore(self.root / "memory.sqlite3", legacy)
        self.assertEqual(store.message_count(), 0)
        legacy.write_text(json.dumps([{"role": "user", "content": "后来修复"}], ensure_ascii=False), encoding="utf-8")
        reopened = MemoryStore(self.root / "memory.sqlite3", legacy)
        self.assertEqual(reopened.load_messages(), [{"role": "user", "content": "后来修复"}])


if __name__ == "__main__":
    unittest.main()
