"""文档入库：解析与分块后，把原文、位置、向量一起保存到本地知识库。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from dotenv import load_dotenv


# 解析、分块、调用 Ollama BGE-M3、更新向量库


load_dotenv(Path(__file__).resolve().parents[1] / ".env")
from rag import KnowledgeBase, SUPPORTED_EXTENSIONS, knowledge_base

MAX_FILE_BYTES = 30 * 1024 * 1024
CHUNK_CHARS = 1200
OVERLAP_CHARS = 180


def extract_sections(path: Path) -> list[tuple[str, str]]:
    """按可追溯位置提取文字。PDF 使用文件页序号；Markdown 保留标题与行范围。"""
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise ValueError("PDF 解析需要 pypdf；当前可先上传 TXT 或 Markdown。") from error
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise ValueError("暂不支持加密 PDF，请先提供未加密副本。")
        sections = [(f"PDF 文件第 {i} 页", page.extract_text() or "")
                    for i, page in enumerate(reader.pages, 1)]
    else:
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = raw.decode("gb18030")
            except UnicodeDecodeError as error:
                raise ValueError("无法识别文本编码，请另存为 UTF-8。") from error
        sections = []
        heading = "正文"
        buffer = []
        first_line = 1
        for line_number, line in enumerate(text.splitlines(), 1):
            if path.suffix.lower() == ".md" and re.match(r"^#{1,6}\s+", line):
                if buffer:
                    sections.append((f"{heading}；第 {first_line}–{line_number - 1} 行", "\n".join(buffer)))
                heading = re.sub(r"^#{1,6}\s+", "", line).strip()
                buffer, first_line = [], line_number
            buffer.append(line)
        if buffer:
            sections.append((f"{heading}；第 {first_line}–{len(text.splitlines())} 行", "\n".join(buffer)))
    if not any(text.strip() for _, text in sections):
        raise ValueError("文档没有可提取的文字；扫描 PDF 需要先进行 OCR。")
    return sections


def validate_chunk_settings(chunk_chars: int, overlap_chars: int) -> tuple[int, int]:
    """分块参数同时用于前端和命令行，统一校验以避免大量重复块。"""
    chunk_chars, overlap_chars = int(chunk_chars), int(overlap_chars)
    if not 200 <= chunk_chars <= 4000:
        raise ValueError("每块大小必须在 200–4000 字符之间。")
    if not 0 <= overlap_chars < chunk_chars // 2:
        raise ValueError("重叠大小必须非负且小于每块大小的一半。")
    return chunk_chars, overlap_chars


def make_chunks(sections: list[tuple[str, str]], chunk_chars: int = CHUNK_CHARS,
                overlap_chars: int = OVERLAP_CHARS) -> list[dict]:
    # 尽量在换行处断开；保留重叠，使跨边界的事实仍有机会完整出现在某一块里。
    chunk_chars, overlap_chars = validate_chunk_settings(chunk_chars, overlap_chars)
    chunks = []
    for location, text in sections:
        start = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            if end < len(text):
                boundary = text.rfind("\n", start + chunk_chars // 2, end)
                if boundary >= 0:
                    end = boundary + 1
            content = text[start:end].strip()
            if content:
                chunks.append({"text": content,
                               "location": f"{location}；节内字符 {start + 1}–{end}"})
            if end == len(text):
                break
            start = max(start + 1, end - overlap_chars)
    return chunks


def ingest_file(file_path: str | Path, kb: KnowledgeBase = knowledge_base,
                chunk_chars: int = CHUNK_CHARS, overlap_chars: int = OVERLAP_CHARS) -> str:
    chunk_chars, overlap_chars = validate_chunk_settings(chunk_chars, overlap_chars)
    # 内容没变但参数变了，也必须重新索引；配置指纹还用于隔离新旧块 ID。
    profile = hashlib.sha256(json.dumps({"parser_version": 2, "chunk_chars": chunk_chars,
                                         "overlap_chars": overlap_chars}, sort_keys=True).encode()).hexdigest()[:16]
    path = Path(file_path).resolve(strict=True)
    if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("仅支持 TXT、Markdown 和有文字层的 PDF。")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("单个文档不能超过 30 MiB。")
    source = path.name
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    # 入库与召回共享锁，避免查询读取同一文档尚未更新完整的状态。
    with kb._lock:
        collection = kb.collection
        old = collection.get(where={"source": source}, include=["metadatas"])
        if old["ids"] and all(m.get("file_hash") == digest and m.get("index_profile") == profile
                              for m in old["metadatas"]):
            return f"跳过：{source} 内容和分块设置未发生变化（{len(old['ids'])} 块）。"
        chunks = make_chunks(extract_sections(path), chunk_chars, overlap_chars)
        vectors = kb.embed([chunk["text"] for chunk in chunks])
        kb.knowledge_dir.mkdir(parents=True, exist_ok=True)
        destination = kb.knowledge_dir / source
        if destination.is_symlink() or destination.resolve().parent != kb.knowledge_dir:
            raise ValueError("资料保存路径不安全。")
        # 同名更新先归档旧原文，再替换副本；用户之前的资料仍可恢复。
        if destination.exists() and destination.read_bytes() != raw:
            old_raw = destination.read_bytes()
            archive = kb.knowledge_dir / ".versions" / hashlib.sha256(old_raw).hexdigest()
            archive.mkdir(parents=True, exist_ok=True)
            backup = archive / source
            if not backup.exists():
                backup.write_bytes(old_raw)
        document_key = hashlib.sha256(source.encode()).hexdigest()[:24]
        ids = [f"{document_key}:{digest}:{profile}:{i}" for i in range(len(chunks))]
        added = []
        kb.invalidate_lexical_index()
        try:
            for start in range(0, len(chunks), 64):
                batch = chunks[start:start + 64]
                batch_ids = ids[start:start + 64]
                collection.upsert(
                    ids=batch_ids, embeddings=vectors[start:start + 64],
                    documents=[chunk["text"] for chunk in batch],
                    metadatas=[{"source": source, "file_hash": digest,
                                "index_profile": profile, "chunk_chars": chunk_chars,
                                "overlap_chars": overlap_chars,
                                "location": chunk["location"], "chunk_index": start + i}
                               for i, chunk in enumerate(batch)],
                )
                added.extend(batch_ids)
            fd, temporary = tempfile.mkstemp(dir=kb.knowledge_dir, suffix=".upload")
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(raw)
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        except Exception:
            if added:
                collection.delete(ids=added)
            raise
        obsolete = list(set(old["ids"]) - set(ids))
        # 只有新块写入、原文件保存成功后才清理旧块；同时失效 BM25 缓存。
        if obsolete:
            collection.delete(ids=obsolete)
    return f"已导入：{source}（{len(chunks)} 块）。"


def import_files(paths, chunk_chars: int = CHUNK_CHARS, overlap_chars: int = OVERLAP_CHARS) -> str:
    if not paths:
        return "请先选择资料文件。"
    results = []
    for path in paths:
        try:
            results.append(ingest_file(path, chunk_chars=chunk_chars, overlap_chars=overlap_chars))
        except Exception as error:
            results.append(f"导入失败：{Path(path).name}：{error}")
    return "\n".join(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将指定文档导入本地知识库")
    parser.add_argument("files", nargs="+")
    parser.add_argument("--chunk-chars", type=int, default=CHUNK_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=OVERLAP_CHARS)
    arguments = parser.parse_args()
    print(import_files(arguments.files, arguments.chunk_chars, arguments.overlap_chars))
