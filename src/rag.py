"""Local embeddings through Ollama and a persistent Chroma knowledge base."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import threading

import chromadb
from chromadb.config import Settings
import httpx

#检索相关片段，生成来源信息


PROJECT_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = PROJECT_DIR / "knowledge"
CHROMA_DIR = PROJECT_DIR / "data" / "chroma"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


class KnowledgeBase:
    def __init__(self, db_path=None, knowledge_dir=None, model=None, ollama_url=None):
        self.knowledge_dir = Path(knowledge_dir or KNOWLEDGE_DIR).resolve()
        self.db_path = Path(db_path or CHROMA_DIR).resolve()
        self.model = model or os.getenv("RAG_EMBED_MODEL", "bge-m3:latest")
        self.ollama_url = (ollama_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self._lock = threading.RLock()
        self._collection = None
        self._client = None

    @property
    def collection(self):
        with self._lock:
            if self._collection is None:
                self.db_path.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=str(self.db_path), settings=Settings(anonymized_telemetry=False)
                )
                collection = self._client.get_or_create_collection(
                    name="documents_v1", embedding_function=None,
                    metadata={"embedding_model": self.model, "pipeline_version": "1"},
                    configuration={"hnsw": {"space": "cosine"}},
                )
                if collection.metadata.get("embedding_model") != self.model:
                    raise ValueError("向量库使用了其他 embedding 模型，请使用独立的新向量库目录。")
                self._collection = collection
            return self._collection

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Serialize GPU requests; never silently truncate oversized chunks."""
        vectors = []
        with self._lock, httpx.Client(timeout=180, trust_env=False) as client:
            for start in range(0, len(texts), 8):
                batch = texts[start:start + 8]
                try:
                    response = client.post(f"{self.ollama_url}/api/embed", json={
                        "model": self.model, "input": batch, "truncate": False,
                        "keep_alive": "10m",
                    })
                    response.raise_for_status()
                    result = response.json()["embeddings"]
                except (httpx.HTTPError, KeyError, ValueError) as error:
                    raise RuntimeError(
                        f"Ollama 向量生成失败，请确认服务和模型 {self.model} 可用：{error}"
                    ) from error
                if len(result) != len(batch):
                    raise RuntimeError("Ollama 返回的向量数量与输入不一致。")
                for vector in result:
                    norm = math.sqrt(sum(x * x for x in vector))
                    if not norm or not math.isfinite(norm):
                        raise RuntimeError("Ollama 返回了无效向量。")
                    vectors.append([x / norm for x in vector])
        return vectors

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if not query.strip():
            return []
        with self._lock:
            count = self.collection.count()
            if not count:
                return []
            result = self.collection.query(
                query_embeddings=self.embed([query]), n_results=min(max(top_k, 1), 8, count),
                include=["documents", "metadatas", "distances"],
            )
        return [dict(id=chunk_id, text=text, **metadata, distance=float(distance))
                for chunk_id, text, metadata, distance in zip(
                    result["ids"][0], result["documents"][0],
                    result["metadatas"][0], result["distances"][0])]

    def status(self) -> str:
        with self._lock:
            result = self.collection.get(include=["metadatas"])
        sources = sorted({m["source"] for m in result["metadatas"]})
        return f"知识库：{len(sources)} 个文档，{len(result['ids'])} 个文本块。\n" + "\n".join(sources)


def evidence_json(hits: list[dict]) -> str:
    return json.dumps([{
        "citation": f"资料{i}", "source": h["source"],
        "location": h["location"], "text": h["text"],
    } for i, h in enumerate(hits, 1)], ensure_ascii=False)


def source_footer(hits: list[dict]) -> str:
    """Render sources directly from retrieval metadata, never from model guesses."""
    import html
    if not hits:
        return ""
    lines = ["\n\n<!-- rag-sources -->\n---\n本次检索资料（具体依据见正文引用）："]
    for i, hit in enumerate(hits, 1):
        title = html.escape(f"[资料{i}] {hit['source']} · {hit['location']}")
        text = html.escape(hit["text"])
        lines.append(f"<details><summary>{title}</summary><pre>{text}</pre></details>")
    return "\n\n".join(lines)


knowledge_base = KnowledgeBase()
