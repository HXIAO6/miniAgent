"""下载固定版本的公开 ONNX 重排模型；只写入项目的 D 盘目录。"""
import hashlib
import json
from pathlib import Path

import httpx
if __package__:
    from .config import settings
else:
    from config import settings

MODEL_DIR = settings.rerank_model_dir
REPO = "Xenova/bge-reranker-base"
REVISION = "280bcc27a84e0b898c251e06fddb25171bd9b101"
FILES = {
    "onnx/model_int8.onnx": "2059d8ef0b6e935b4845e11b38c9af9e9e2e7b91f69fc99efe03254e0a7da8d3",
    "tokenizer.json": "48564c5c7d3fa64d85d95e65414a542385f88b0f128fd8d4163fd7a57f2be05c",
}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download():
    # 不在聊天请求中自动下载。固定 revision 与 SHA256，避免损坏或版本漂移。
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for name, expected in FILES.items():
            target = MODEL_DIR / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if file_hash(target) == expected:
                    print(f"已存在并校验：{name}", flush=True)
                    continue
                raise RuntimeError(f"现有文件校验失败，请先检查并保留副本：{target}")
            partial = target.with_suffix(target.suffix + ".partial")
            print(f"下载：{name}", flush=True)
            with client.stream("GET", f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}") as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for block in response.iter_bytes(1024 * 1024):
                        stream.write(block)
            if file_hash(partial) != expected:
                raise RuntimeError(f"下载校验失败，保留临时文件供检查：{partial}")
            partial.replace(target)
    (MODEL_DIR / "manifest.json").write_text(json.dumps({
        "repository": REPO, "revision": REVISION, "sha256": FILES,
        "runtime": "ONNX Runtime CPU INT8", "base_model": "BAAI/bge-reranker-base",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"模型已准备：{MODEL_DIR}", flush=True)


if __name__ == "__main__":
    download()
