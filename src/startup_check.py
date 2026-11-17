"""启动前检查外部服务和本地模型，返回可读结果，不暴露密钥。"""
from __future__ import annotations

import os
from urllib.parse import urlsplit
import httpx

if __package__:
    from .config import settings, logger
else:
    from config import settings, logger


def run_startup_checks() -> list[tuple[str, bool, str]]:
    checks = []
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5, trust_env=False)
        response.raise_for_status()
        models = {item.get("name", "") for item in response.json().get("models", [])}
        expected = settings.embedding_model if ":" in settings.embedding_model else settings.embedding_model + ":latest"
        found = expected in models
        checks.append(("Ollama", True, f"已连接，embedding 模型{'可用' if found else '未找到'}：{settings.embedding_model}"))
        if not found:
            checks[-1] = ("Ollama", False, f"已连接，但未找到模型：{settings.embedding_model}")
    except Exception as error:
        checks.append(("Ollama", False, f"无法读取本地模型列表（{type(error).__name__}），请检查地址与服务。"))
    rerank_model = settings.rerank_model_dir / "onnx" / "model_int8.onnx"
    tokenizer = settings.rerank_model_dir / "tokenizer.json"
    ready = rerank_model.is_file() and tokenizer.is_file()
    checks.append(("重排模型", ready, "本地模型文件已准备" if ready else f"缺少模型，请运行 src/setup_reranker.py：{settings.rerank_model_dir}"))
    try:
        parsed = urlsplit(settings.openai_base_url or "https://api.openai.com/v1")
        configured = parsed.scheme in {"http", "https"} and bool(parsed.hostname) and bool(settings.openai_api_key)
    except ValueError:
        configured = False
    checks.append(("生成模型配置", configured,
                   f"模型：{settings.chat_model}；摘要模型：{settings.summary_model}。"
                   + ("地址和密钥已配置；未发送计费请求，尚未验证模型可用性。" if configured
                      else "请检查 OPENAI_API_KEY 和 OPENAI_BASE_URL。")))
    try:
        # 只读打开已有 SQLite，不创建 collection、不写入用户向量库。
        import sqlite3
        database = settings.chroma_dir / "chroma.sqlite3"
        if database.exists():
            with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
                result = connection.execute("PRAGMA quick_check").fetchone()[0]
            if result != "ok":
                raise RuntimeError("SQLite integrity check failed")
            detail = "已有 SQLite 文件可读且完整性检查通过；未执行向量查询。"
        else:
            parent = settings.chroma_dir
            while not parent.exists():
                parent = parent.parent
            if not os.access(parent, os.W_OK):
                raise PermissionError("Directory not writable")
            detail = "尚未创建向量库，父目录可写；首次使用时初始化。"
        checks.append(("ChromaDB", True, detail))
    except Exception as error:
        checks.append(("ChromaDB", False, f"本地数据库检查失败（{type(error).__name__}）。"))
    if not settings.tavily_api_key:
        checks.append(("联网搜索", True, "未配置 TAVILY_API_KEY，联网搜索不可用但不影响本地 RAG"))
    logger.info("启动自检完成：%s", "; ".join(f"{name}={'通过' if ok else '注意'}" for name, ok, _ in checks))
    return checks


def format_startup_checks(checks: list[tuple[str, bool, str]]) -> str:
    return "\n".join(f"{'✅' if ok else '⚠️'} {name}：{message}" for name, ok, message in checks)


if __name__ == "__main__":
    print(format_startup_checks(run_startup_checks()))
