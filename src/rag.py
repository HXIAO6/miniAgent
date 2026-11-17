"""Local embeddings through Ollama and a persistent Chroma knowledge base."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import re
import threading
import uuid

import chromadb
from chromadb.config import Settings
import httpx
import jieba
from rank_bm25 import BM25Plus
if __package__:
    from .config import settings, logger
else:
    from config import settings, logger

# 检索分三阶段：两路召回 → RRF 融合 → 可选重排与阈值过滤。


PROJECT_DIR = settings.project_dir
KNOWLEDGE_DIR = settings.knowledge_dir
CHROMA_DIR = settings.chroma_dir
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
RRF_K = 60  # 融合只使用各通道名次，避免直接相加不同尺度的原始分数。
DEFAULT_RERANK_THRESHOLD = 0.2  # 合成校准集的初始工作点；真实资料仍需调参，不代表正确率。
_tokenizer = jieba.Tokenizer()
_tokenizer.tmp_dir = str(PROJECT_DIR / ".cache")
_tokenizer.cache_file = "jieba-rag.cache"
_tokenizer_lock = threading.Lock()
_stopwords = {"的", "了", "是", "在", "和", "与", "吗", "呢", "请", "问", "我", "你",
              "什么", "怎么", "如何", "哪些", "根据", "一下", "一个"}


def tokenize(text: str) -> list[str]:
    """中文分词；保留错误码、函数名整体，英文统一小写。"""
    tokens = []
    for part in re.findall(r"[a-zA-Z0-9_]+(?:[./:\-][a-zA-Z0-9_]+)*|[\u3400-\u9fff]+", text):
        if re.fullmatch(r"[\u3400-\u9fff]+", part):
            with _tokenizer_lock:
                (PROJECT_DIR / ".cache").mkdir(exist_ok=True)
                words = list(_tokenizer.cut(part))
            tokens.extend(word for word in words if word.strip() and word not in _stopwords)
        else:
            tokens.append(part.casefold())
    return tokens


class KnowledgeBase:
    def __init__(self, db_path=None, knowledge_dir=None, model=None, ollama_url=None):
        self.knowledge_dir = Path(knowledge_dir or KNOWLEDGE_DIR).resolve()
        self.db_path = Path(db_path or CHROMA_DIR).resolve()
        self.model = model or settings.embedding_model
        self.ollama_url = (ollama_url or settings.ollama_base_url).rstrip("/")
        self._lock = threading.RLock()
        self._collection = None
        self._client = None
        self._bm25_cache = None

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
        """串行调用本地 Ollama；过长输入报错，不静默截断后再生成向量。"""
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
                    logger.error("Embedding 请求失败，类型=%s", type(error).__name__)
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

    def invalidate_lexical_index(self):
        """资料变化后清除内存 BM25 索引，下次检索再从最新文本构建。"""
        with self._lock:
            self._bm25_cache = None

    def _lexical_index(self):
        if self._bm25_cache is None:
            result = self.collection.get(include=["documents", "metadatas"])
            rows = sorted([dict(id=chunk_id, text=text, **meta)
                           for chunk_id, text, meta in zip(
                               result["ids"], result["documents"], result["metadatas"])],
                          key=lambda row: row["id"])
            corpus = [tokenize(row["source"] + " " + row["text"])
                      for row in rows]
            # delta=0：保持正 IDF，但不给完全没有匹配词的文档额外加分。
            index = BM25Plus(corpus, delta=0) if any(corpus) else None
            self._bm25_cache = (rows, index)
        return self._bm25_cache

    def search_candidates(self, query: str, candidate_k: int = 20, mode: str = "hybrid") -> list[dict]:
        """只召回候选；分数相近不代表证据充分，过滤放在 retrieve 中完成。"""
        if mode not in {"hybrid", "vector", "bm25"}:
            raise ValueError("不支持的检索模式。")
        candidate_k = int(candidate_k)
        if not 1 <= candidate_k <= 100:
            raise ValueError("每路候选数量必须在 1–100 之间。")
        if not query.strip():
            return []
        with self._lock:
            count = self.collection.count()
            if not count:
                return []
            merged = {}
            if mode in {"hybrid", "vector"}:
                result = self.collection.query(
                    query_embeddings=self.embed([query]), n_results=min(candidate_k, count),
                    include=["documents", "metadatas", "distances"],
                )
                for rank, (chunk_id, text, meta, distance) in enumerate(zip(
                        result["ids"][0], result["documents"][0], result["metadatas"][0],
                        result["distances"][0]), 1):
                    merged[chunk_id] = dict(id=chunk_id, text=text, **meta,
                                            distance=float(distance), similarity=1 - float(distance),
                                            vector_rank=rank, bm25_rank=None, bm25_score=None,
                                            rrf_score=1 / (RRF_K + rank))
            if mode in {"hybrid", "bm25"}:
                rows, index = self._lexical_index()
                terms = tokenize(query)
                if index is not None and terms:
                    scores = index.get_scores(terms)
                    order = sorted((i for i, score in enumerate(scores) if score > 0),
                                   key=lambda i: (-float(scores[i]), rows[i]["id"]))[:candidate_k]
                    for rank, i in enumerate(order, 1):
                        row = rows[i]
                        if row["id"] not in merged:
                            merged[row["id"]] = dict(row, distance=None, similarity=None,
                                                      vector_rank=None, bm25_rank=None,
                                                      bm25_score=None, rrf_score=0.0)
                        hit = merged[row["id"]]
                        hit.update(bm25_rank=rank, bm25_score=float(scores[i]))
                        hit["rrf_score"] += 1 / (RRF_K + rank)
            return sorted(merged.values(), key=lambda hit: (-hit["rrf_score"], hit["id"]))

    def search(self, query: str, top_k: int = 4, candidate_k: int = 20,
               mode: str = "hybrid") -> list[dict]:
        top_k = int(top_k)
        if not 1 <= top_k <= 10:
            raise ValueError("最终片段上限必须在 1–10 之间。")
        return self.search_candidates(query, max(top_k, int(candidate_k)), mode)[:top_k]

    def retrieve(self, query: str, top_k=4, candidate_k=20, mode="hybrid",
                 use_rerank=True, threshold=None, rerank_k=20, scorer=None):
        """完整检索入口：保留诊断候选，但只把通过过滤的块交给生成模型。"""
        top_k, rerank_k = int(top_k), int(rerank_k)
        threshold = DEFAULT_RERANK_THRESHOLD if threshold is None else float(threshold)
        if not 1 <= top_k <= 10 or not 1 <= rerank_k <= 100:
            raise ValueError("片段上限须为 1–10，重排候选上限须为 1–100。")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("重排阈值必须是 0–1 的有限数值。")
        candidates = self.search_candidates(query, max(int(candidate_k), top_k), mode)
        if use_rerank and candidates:
            if scorer is None:
                if __package__:
                    from .rerank import reranker
                else:
                    from rerank import reranker
                scorer = reranker
            # 仅精排融合后的前 N 个，控制 CPU 延迟；未重排候选不越过过滤进入回答。
            limit = max(top_k, rerank_k)
            ranked = scorer.score(query, candidates[:limit])
            candidates = ranked + candidates[limit:]
            selected = select_context([h for h in ranked if h["rerank_score"] >= threshold], top_k)
        else:
            selected = select_context(candidates, top_k)
        selected_ids = {hit["id"] for hit in selected}
        for hit in candidates:
            if hit["id"] in selected_ids:
                hit["decision"] = "已选入回答"
            elif use_rerank and hit.get("rerank_score") is None:
                hit["decision"] = "未参与重排"
            elif use_rerank and hit["rerank_score"] < threshold:
                hit["decision"] = "低于阈值"
            else:
                hit["decision"] = "超出数量或长度上限"
        logger.info("检索完成，方式=%s，候选=%d，入选=%d，重排=%s", mode, len(candidates), len(selected), use_rerank)
        return {"candidates": candidates, "selected": selected,
                "threshold": threshold if use_rerank else None,
                "use_rerank": use_rerank}

    def status(self) -> str:
        with self._lock:
            result = self.collection.get(include=["metadatas"])
        sources = sorted({m["source"] for m in result["metadatas"]})
        return f"知识库：{len(sources)} 个文档，{len(result['ids'])} 个文本块。\n" + "\n".join(sources)

    @staticmethod
    def _validate_source(source: str):
        """删除只接受资料文件名，不接受路径、Windows 流名称或特殊目录。"""
        if (not isinstance(source, str) or not source or source in {".", ".."}
                or re.search(r'[<>:"/\\|?*\x00-\x1f]', source)
                or source.endswith((" ", ".")) or Path(source).suffix.lower() not in SUPPORTED_EXTENSIONS):
            raise ValueError("非法资料文件名；只能删除知识库列表中的文档。")

    def list_documents(self) -> list[dict]:
        """合并向量库与原文件清单，原文缺失或尚未索引的文件也能显示。"""
        with self._lock:
            data = self.collection.get(include=["metadatas"])
            counts = {}
            for metadata in data["metadatas"]:
                name = metadata["source"]
                counts[name] = counts.get(name, 0) + 1
            if self.knowledge_dir.exists():
                for path in self.knowledge_dir.iterdir():
                    if path.suffix.lower() in SUPPORTED_EXTENSIONS and (path.is_file() or path.is_symlink()):
                        counts.setdefault(path.name, 0)
            return [{"source": name, "chunk_count": count,
                     "has_file": (self.knowledge_dir / name).is_file()}
                    for name, count in sorted(counts.items())]

    def delete_document(self, source: str) -> dict:
        """从检索库移除文档，原文与索引快照保留在回收站，失败时尽量回滚。"""
        self._validate_source(source)
        with self._lock:
            if source not in {row["source"] for row in self.list_documents()}:
                raise ValueError("该文档已不在知识库列表中，请刷新后重试。")
            original = self.knowledge_dir / source
            # 禁止通过链接或目录跳转移动知识库外部的文件。
            if original.is_symlink() or original.resolve() != original or (original.exists() and not original.is_file()):
                raise ValueError("原文件路径异常，未执行删除。")
            trash_root = self.db_path.parent / "trash"
            if trash_root.resolve() != trash_root or trash_root.is_symlink():
                raise ValueError("回收站路径异常，未执行删除。")
            trash_root.mkdir(parents=True, exist_ok=True)
            batch = trash_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex)
            batch.mkdir()
            # Chroma 与文件系统没有跨库事务；先持久化完整快照，确保异常时仍有恢复材料。
            snapshot = self.collection.get(where={"source": source},
                                           include=["documents", "metadatas", "embeddings"])
            vectors = snapshot["embeddings"]
            snapshot_data = {"ids": snapshot["ids"], "documents": snapshot["documents"],
                             "metadatas": snapshot["metadatas"],
                             "embeddings": vectors.tolist() if hasattr(vectors, "tolist") else vectors}
            record = {"source": source, "state": "prepared", "original_path": str(original),
                      "chunk_count": len(snapshot["ids"]), "snapshot": snapshot_data}
            manifest = batch / "manifest.json"

            def save_record():
                temporary = batch / "manifest.json.tmp"
                temporary.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
                os.replace(temporary, manifest)

            save_record()
            recycled = batch / source
            moved = False
            db_attempted = False
            try:
                # 先移动原文；若文件被占用，数据库仍保持原样。
                if original.exists():
                    original.rename(recycled)
                    moved = True
                self.invalidate_lexical_index()
                if snapshot["ids"]:
                    db_attempted = True
                    self.collection.delete(ids=snapshot["ids"])
                    if self.collection.get(where={"source": source}, include=[])["ids"]:
                        raise RuntimeError("向量库仍存在该文档的记录。")
                record["state"] = "deleted"
                save_record()
            except Exception as error:
                logger.error("文档删除失败，开始回滚，类型=%s", type(error).__name__)
                # 删除调用可能已经部分生效，因此只要尝试过就恢复快照；BM25 下次读取最新库。
                rollback_errors = []
                if db_attempted:
                    try:
                        for start in range(0, len(snapshot["ids"]), 64):
                            self.collection.upsert(**{key: value[start:start + 64]
                                for key, value in snapshot_data.items()})
                    except Exception as rollback_error:
                        rollback_errors.append(f"索引恢复失败：{rollback_error}")
                if moved:
                    try:
                        if original.exists():
                            raise RuntimeError("原路径已被占用，保留回收站原文，避免覆盖。")
                        recycled.rename(original)
                    except Exception as rollback_error:
                        rollback_errors.append(f"原文恢复失败：{rollback_error}")
                self.invalidate_lexical_index()
                record["state"] = "rollback_failed" if rollback_errors else "rolled_back"
                record["error"] = str(error)
                record["rollback_errors"] = rollback_errors
                try:
                    save_record()
                except Exception as record_error:
                    rollback_errors.append(f"状态写入失败：{record_error}")
                detail = "；".join(rollback_errors) if rollback_errors else "原文与索引已恢复"
                logger.warning("删除回滚结果=%s", record["state"])
                raise RuntimeError(f"删除失败：{error}；{detail}。恢复材料：{batch}") from error
            logger.info("文档删除完成，移除块数=%d，原文件回收=%s", len(snapshot["ids"]), moved)
            return {"source": source, "chunk_count": len(snapshot["ids"]),
                    "file_recycled": moved, "trash_path": str(batch)}


def evidence_json(hits: list[dict]) -> str:
    """只给模型已选中的原文与来源，不把候选分数包装成可信度。"""
    return json.dumps([{
        "citation": f"资料{i}", "source": h["source"],
        "location": h["location"], "text": h["text"],
    } for i, h in enumerate(hits, 1)], ensure_ascii=False)


def source_footer(hits: list[dict]) -> str:
    """来源标题直接来自索引元数据，并转义原文，避免模型编造来源或注入 HTML。"""
    import html
    if not hits:
        return ""
    lines = ["\n\n<!-- rag-sources -->\n---\n本次检索资料（具体依据见正文引用）："]
    for i, hit in enumerate(hits, 1):
        title = html.escape(f"[资料{i}] {hit['source']} · {hit['location']}")
        text = html.escape(hit["text"])
        scores = html.escape(score_description(hit))
        lines.append(f"<details><summary>{title}</summary><p>{scores}</p><pre>{text}</pre></details>")
    return "\n\n".join(lines)


def score_description(hit: dict) -> str:
    def show(key):
        value = hit.get(key)
        return "未入选该路候选" if value is None else f"{value:.4f}"
    return (f"向量相似度：{show('similarity')}；BM25：{show('bm25_score')}；"
            f"RRF：{show('rrf_score')}；重排：{show('rerank_score')}。这些分数不是答案正确率。")


def candidate_table(hits: list[dict], top_k: int, selected=None) -> list[list]:
    # 阈值可能让中间若干块被排除，引用编号必须按实际入选集合编号，不能按候选行号推算。
    selected = hits[:top_k] if selected is None else selected
    references = {hit["id"]: f"资料{i}" for i, hit in enumerate(selected, 1)}
    return [[i, references.get(hit["id"], "未送入回答"), hit["source"], hit["location"],
             hit.get("similarity"), hit.get("bm25_score"), hit.get("rrf_score"),
             hit.get("vector_rank"), hit.get("bm25_rank"), hit["text"][:180],
             hit.get("rerank_score"), hit.get("decision", ""), hit.get("rerank_windows")]
            for i, hit in enumerate(hits, 1)]


def select_context(candidates: list[dict], top_k: int, max_chars: int = 12000) -> list[dict]:
    """同时限制片段数量和总字符数，避免用户调大 K 后耗尽上下文。"""
    selected, used = [], 0
    for hit in candidates[:top_k]:
        if used + len(hit["text"]) > max_chars:
            break
        selected.append(hit)
        used += len(hit["text"])
    return selected


knowledge_base = KnowledgeBase()
