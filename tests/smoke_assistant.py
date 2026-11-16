"""Optional full smoke test: local UI plus one request to the configured LLM."""
import asyncio
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import httpx
import knowledge_assistant as app
from ingest import ingest_file
from rag import KnowledgeBase


async def main():
    (ROOT / ".cache").mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="assistant-smoke-", dir=ROOT / ".cache"))
    kb = KnowledgeBase(root / "chroma", root / "knowledge")
    source = root / "验证维护手册.md"
    source.write_text("# 星舟设备\n星舟设备的冷却液更换周期是九天。", encoding="utf-8")
    await asyncio.to_thread(ingest_file, source, kb)
    app.knowledge_base = kb
    app.HISTORY_FILE = str(root / "history.json")
    try:
        app.demo.launch(server_name="127.0.0.1", server_port=7861,
                        prevent_thread_lock=True, inbrowser=False, quiet=True)
        with httpx.Client(trust_env=False) as client:
            response = client.get("http://127.0.0.1:7861", timeout=30)
            response.raise_for_status()
            assert "本地知识库助手" in response.text
        print("UI HTTP check: OK", flush=True)
        final = None
        async for _, history in app.predict("根据维护手册，星舟设备多久更换一次冷却液？请简短回答并引用来源。", [], True):
            final = history[-1]["content"]
        assert "发生错误" not in final, final
        answer = final.split("</details>", 1)[1].split("<!-- rag-sources -->", 1)[0]
        assert "九天" in answer or "9" in answer, answer
        assert "[资料1]" in answer, answer
        assert "验证维护手册.md" in final
        print("LLM answer and citation check: OK", flush=True)
        print(answer.strip(), flush=True)
    finally:
        app.demo.close()


if __name__ == "__main__":
    asyncio.run(main())
