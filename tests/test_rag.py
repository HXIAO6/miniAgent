"""Integration checks with real local Ollama; never touch the production collection."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ingest import ingest_file, make_chunks, extract_sections
from rag import KnowledgeBase, evidence_json, source_footer


class RagIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / ".cache").mkdir(exist_ok=True)
        # Chroma holds SQLite handles on Windows; retain isolated test output under .cache.
        cls.root = Path(tempfile.mkdtemp(prefix="rag-tests-", dir=ROOT / ".cache"))
        cls.kb = KnowledgeBase(cls.root / "chroma", cls.root / "knowledge")

    def test_01_import_search_and_duplicate(self):
        path = self.root / "星舟维护手册.md"
        path.write_text("# 星舟设备维护\n星舟设备必须每隔七天更换冷却液。\n", encoding="utf-8")
        self.assertIn("已导入", ingest_file(path, self.kb))
        self.assertIn("跳过", ingest_file(path, self.kb))
        self.assertEqual(self.kb.collection.count(), 1)
        hit = self.kb.search("星舟设备多久更换冷却液？")[0]
        self.assertEqual(hit["source"], path.name)
        self.assertIn("七天", hit["text"])
        self.assertIn("星舟设备维护", hit["location"])
        self.assertIn("资料1", evidence_json([hit]))
        self.assertIn(path.name, source_footer([hit]))

    def test_02_update_replaces_old_chunks_and_keeps_original(self):
        path = self.root / "星舟维护手册.md"
        path.write_text("# 星舟设备维护\n星舟设备必须每隔九天更换冷却液。\n", encoding="utf-8")
        ingest_file(path, self.kb)
        self.assertEqual(self.kb.collection.count(), 1)
        self.assertIn("九天", self.kb.search("更换冷却液的间隔")[0]["text"])
        backups = list((self.kb.knowledge_dir / ".versions").rglob("*.md"))
        self.assertEqual(len(backups), 1)
        self.assertIn("七天", backups[0].read_text(encoding="utf-8"))

    def test_03_embedding_failure_preserves_existing_document(self):
        path = self.root / "星舟维护手册.md"
        path.write_text("更换为无法导入的新版本。", encoding="utf-8")
        with patch.object(self.kb, "embed", side_effect=RuntimeError("offline")):
            with self.assertRaises(RuntimeError):
                ingest_file(path, self.kb)
        self.assertEqual(self.kb.collection.count(), 1)
        self.assertIn("九天", (self.kb.knowledge_dir / path.name).read_text(encoding="utf-8"))

    def test_04_chunk_coverage_and_pdf_rejection(self):
        text = "段落的测试文本。" * 500
        chunks = make_chunks([("正文", text)])
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c["text"]) <= 1200 for c in chunks))
        from pypdf import PdfWriter
        path = self.root / "empty.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.write(path)
        with self.assertRaisesRegex(ValueError, "OCR"):
            ingest_file(path, self.kb)

    def test_05_restart_and_unsupported_file(self):
        reopened = KnowledgeBase(self.kb.db_path, self.kb.knowledge_dir)
        self.assertEqual(reopened.collection.count(), 1)
        self.assertIn("九天", reopened.search("星舟维护")[0]["text"])
        path = self.root / "secret.env"
        path.write_text("do not index", encoding="utf-8")
        with self.assertRaises(ValueError):
            ingest_file(path, self.kb)

    def test_06_source_html_is_escaped(self):
        footer = source_footer([{"source": "<script>.txt", "location": "正文",
                                 "text": "<script>alert(1)</script>"}])
        self.assertNotIn("<script>", footer)
        self.assertIn("&lt;script&gt;", footer)

    def test_07_pdf_text_and_page_location(self):
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        writer = PdfWriter()
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 20 250 Td (Cooling interval: nine days.) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
        path = self.root / "maintenance.pdf"
        writer.write(path)
        sections = extract_sections(path)
        self.assertIn("1", sections[0][0])
        self.assertIn("nine days", sections[0][1])


if __name__ == "__main__":
    unittest.main()
