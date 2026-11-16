"""本地 Cross-Encoder：逐对阅读问题和片段，独立于 embedding 的相似度评分。"""
from __future__ import annotations

import math
import os
from pathlib import Path
import threading

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_DIR / "models" / "bge-reranker-base"


class LocalReranker:
    def __init__(self, model_dir=None, batch_size=4, threads=6):
        self.model_dir = Path(model_dir or os.getenv("RAG_RERANK_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
        self.batch_size = batch_size
        self.threads = threads
        self._lock = threading.Lock()
        self._session = None
        self._tokenizer = None
        self._query_tokenizer = None

    def _load(self):
        """第一次用到时才加载；文件缺失时报错，不会静默跳过重排。"""
        if self._session is not None:
            return
        model_path = self.model_dir / "onnx" / "model_int8.onnx"
        tokenizer_path = self.model_dir / "tokenizer.json"
        if not model_path.is_file() or not tokenizer_path.is_file():
            raise RuntimeError("本地重排模型未准备好，请先运行 src/setup_reranker.py，或关闭重排。")
        import onnxruntime as ort
        from tokenizers import Tokenizer
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model_path), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # 模型最多接收 512 token。长块用带重叠的窗口评分，避免只看开头而漏掉后文证据。
        tokenizer.no_padding()
        tokenizer.enable_truncation(max_length=512, strategy="only_second", stride=96)
        query_tokenizer = Tokenizer.from_file(str(tokenizer_path))
        query_tokenizer.no_truncation()
        query_tokenizer.no_padding()
        self._tokenizer, self._query_tokenizer, self._session = tokenizer, query_tokenizer, session

    def score(self, query: str, hits: list[dict]) -> list[dict]:
        if not hits:
            return []
        with self._lock:
            self._load()
            # 问题预留最多 128 token，确保长问题仍给文档留下可评分的空间。
            query_tokens = self._query_tokenizer.encode(query, add_special_tokens=False).ids
            shortened = len(query_tokens) > 128
            if shortened:
                query = self._query_tokenizer.decode(query_tokens[:128], skip_special_tokens=True)
            encodings, owners = [], []
            for i, hit in enumerate(hits):
                # 文件名辅助识别型号/文档主题，正文仍是核心证据。
                encoded = self._tokenizer.encode(query, hit["source"] + "\n" + hit["text"])
                windows = [encoded] + list(encoded.overflowing)
                encodings.extend(windows)
                owners.extend([i] * len(windows))
            maximums = [float("-inf")] * len(hits)
            window_counts = [0] * len(hits)
            input_names = {item.name for item in self._session.get_inputs()}
            pad_id = self._tokenizer.token_to_id("<pad>")
            for start in range(0, len(encodings), self.batch_size):
                batch = encodings[start:start + self.batch_size]
                width = max(len(e.ids) for e in batch)
                ids = np.full((len(batch), width), pad_id, dtype=np.int64)
                masks = np.zeros_like(ids)
                types = np.zeros_like(ids)
                for i, encoding in enumerate(batch):
                    ids[i, :len(encoding.ids)] = encoding.ids
                    masks[i, :len(encoding.ids)] = encoding.attention_mask
                    types[i, :len(encoding.ids)] = encoding.type_ids
                values = {"input_ids": ids, "attention_mask": masks, "token_type_ids": types}
                logits = np.asarray(self._session.run(None, {k: values[k] for k in input_names})[0]).reshape(-1)
                if len(logits) != len(batch) or not np.all(np.isfinite(logits)):
                    raise RuntimeError("重排模型返回了无效分数。")
                for offset, logit in enumerate(logits):
                    owner = owners[start + offset]
                    maximums[owner] = max(maximums[owner], float(logit))
                    window_counts[owner] += 1
            result = []
            for hit, logit, count in zip(hits, maximums, window_counts):
                # Sigmoid 便于使用 0–1 的阈值；这个值不是“答案正确概率”。
                score = 1 / (1 + math.exp(-max(-80, min(80, logit))))
                result.append(dict(hit, rerank_score=score, rerank_logit=logit,
                                   rerank_windows=count, query_truncated=shortened))
            return sorted(result, key=lambda hit: (-hit["rerank_score"], -hit["rrf_score"], hit["id"]))


reranker = LocalReranker()
