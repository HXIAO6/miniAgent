"""通过真实 Gradio API 验证文档列表、确认与删除，不调用 LLM 或正式知识库。"""
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gradio_client import Client
import knowledge_assistant as app
from rag import KnowledgeBase


def main():
    (ROOT / ".cache").mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="document-ui-smoke-", dir=ROOT / ".cache"))
    kb = KnowledgeBase(root / "data/chroma", root / "knowledge")
    kb.knowledge_dir.mkdir()
    source = "删除测试.md"
    (kb.knowledge_dir / source).write_text("用于隔离验证", encoding="utf-8")
    kb.collection.add(ids=["smoke"], documents=["用于隔离验证"], embeddings=[[1.0, 0.0]],
                       metadatas=[{"source": source, "location": "正文"}])
    app.knowledge_base = kb
    try:
        app.demo.launch(server_name="127.0.0.1", server_port=7862,
                        prevent_thread_lock=True, inbrowser=False, quiet=True)
        client = Client("http://127.0.0.1:7862", verbose=False)
        panel = client.predict(api_name="/refresh_document_panel")
        assert panel[0]["data"][0][0] == source
        blocked = client.predict([source], False, api_name="/delete_documents")
        assert "勾选确认" in blocked[0]
        assert kb.collection.count() == 1
        removed = client.predict([source], True, api_name="/delete_documents")
        assert "已删除" in removed[0], removed
        assert removed[1]["data"] == []
        assert removed[-1]["data"] == []
        assert kb.collection.count() == 0
        assert not (kb.knowledge_dir / source).exists()
        record = next((root / "data/trash").glob("*/manifest.json"))
        assert json.loads(record.read_text(encoding="utf-8"))["state"] == "deleted"
        print("Document list, confirmation, deletion, trash and cleared candidates: OK", flush=True)
    finally:
        app.demo.close()


if __name__ == "__main__":
    main()
