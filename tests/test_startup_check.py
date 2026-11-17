"""自检使用模拟服务与隔离目录，不联网、不操作正式向量库。"""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config
import startup_check


class StartupCheckTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".cache").mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="startup-tests-", dir=ROOT / ".cache"))
        self.settings = replace(config.settings, rerank_model_dir=self.root / "models",
                                chroma_dir=self.root / "chroma", embedding_model="bge-m3:latest",
                                openai_base_url="https://example.com/v1", openai_api_key="secret-key",
                                tavily_api_key="")

    def test_detects_missing_exact_tag_and_never_prints_key(self):
        response = Mock()
        response.json.return_value = {"models": [{"name": "bge-m3:other"}]}
        with patch.object(startup_check, "settings", self.settings), \
             patch.object(startup_check.httpx, "get", return_value=response):
            checks = startup_check.run_startup_checks()
        self.assertFalse(checks[0][1])
        self.assertFalse(checks[1][1])
        rendered = startup_check.format_startup_checks(checks)
        self.assertNotIn("secret-key", rendered)
        self.assertNotIn("secret-key", repr(self.settings))
        self.assertTrue(next(ok for name, ok, _ in checks if name == "ChromaDB"))

    def test_offline_error_redacted_and_bad_db_reported(self):
        self.settings.chroma_dir.mkdir()
        (self.settings.chroma_dir / "chroma.sqlite3").write_bytes(b"not a sqlite file")
        with patch.object(startup_check, "settings", self.settings), \
             patch.object(startup_check.httpx, "get", side_effect=RuntimeError("secret-key in URL")):
            checks = startup_check.run_startup_checks()
        self.assertFalse(next(ok for name, ok, _ in checks if name == "ChromaDB"))
        self.assertNotIn("secret-key", startup_check.format_startup_checks(checks))

    def test_relative_model_path_uses_project_not_working_directory(self):
        with patch.dict("os.environ", {"RAG_RERANK_MODEL_DIR": "./models/custom"}):
            self.assertEqual(config._path("RAG_RERANK_MODEL_DIR", self.root), ROOT / "models/custom")


if __name__ == "__main__":
    unittest.main()
