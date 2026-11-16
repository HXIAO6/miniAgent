"""Optional full smoke test: local UI plus one request to the configured LLM."""
import asyncio
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import httpx
from gradio_client import Client
import knowledge_assistant as app
from ingest import ingest_file
from rag import KnowledgeBase, DEFAULT_RERANK_THRESHOLD


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
        client = Client("http://127.0.0.1:7861", verbose=False)
        result = await asyncio.to_thread(client.predict,
            "根据维护手册，星舟设备多久更换一次冷却液？请简短回答并引用来源。",
            [], True, 2, 20, "hybrid", True, DEFAULT_RERANK_THRESHOLD, 20, True,
            api_name="/predict_with_scores")
        final = app.message_content_to_text(result[1][-1]["content"])
        assert result[2]["data"][0][1] == "资料1", result[2]
        assert len(result[2]["data"][0]) == 13
        assert result[2]["data"][0][10] >= DEFAULT_RERANK_THRESHOLD
        assert "发生错误" not in final, final
        answer = final.split("</details>", 1)[1].split("<!-- rag-sources -->", 1)[0]
        assert "九天" in answer or "9" in answer, answer
        assert "[资料1]" in answer, answer
        assert "验证维护手册.md" in final
        print("LLM answer and citation check: OK", flush=True)
        print(answer.strip(), flush=True)
        # 完全无关问题应在程序层被拒答，且不会分配伪来源编号。
        refusal = await asyncio.to_thread(client.predict,
            "月球陨石坑的命名规则是什么？", [], True, 2, 20, "hybrid",
            True, DEFAULT_RERANK_THRESHOLD, 20, True, api_name="/predict_with_scores")
        refusal_text = app.message_content_to_text(refusal[1][-1]["content"])
        assert "没有找到足够相关" in refusal_text, refusal_text
        assert all(row[1] == "未送入回答" for row in refusal[2]["data"])
        print("Strict refusal and rejected-candidate table check: OK", flush=True)
        # 对部分有依据的问题，模型只回答已知周期，不能编造资料里没有的品牌。
        partial = await asyncio.to_thread(client.predict,
            "星舟冷却液多久更换一次，指定品牌是什么？请仅依据资料回答。",
            [], True, 2, 20, "hybrid", True, DEFAULT_RERANK_THRESHOLD, 20, True,
            api_name="/predict_with_scores")
        partial_text = app.message_content_to_text(partial[1][-1]["content"])
        partial_answer = partial_text.split("</details>", 1)[1].split("<!-- rag-sources -->", 1)[0]
        assert "九天" in partial_answer or "9" in partial_answer, partial_answer
        assert any(word in partial_answer for word in ["未提供", "未说明", "未提及", "没有", "不足", "无法", "不详", "未知"]), partial_answer
        print("Partial evidence answer check: OK", flush=True)
        print(partial_answer.strip(), flush=True)
    finally:
        app.demo.close()


if __name__ == "__main__":
    asyncio.run(main())
