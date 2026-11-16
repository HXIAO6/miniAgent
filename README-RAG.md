# 本地知识库助手

入口 `src/knowledge_assistant.py` 从原来的 `ds7.py` 重命名并扩展；`ds6.py` 不参与本次修改。

## 启动

在 PowerShell 中运行：

```powershell
Set-Location D:\miniAgent
& .\.venv\Scripts\python.exe .\src\knowledge_assistant.py
```

浏览器访问 `http://127.0.0.1:7860`。先确保 Ollama 已启动，且 `ollama list` 包含 `bge-m3:latest`。
大模型继续使用项目 `.env` 中原有连接配置。
VS Code 的项目默认解释器也已指向 `.venv`；若之前手动选择过解释器，
请在状态栏重新选择 `D:\miniAgent\.venv\Scripts\python.exe`。

本地 `.venv` 位于 D 盘，通过 `--system-site-packages` 复用已有的
`C:\Users\PC\.conda\envs\agent-dev` 依赖；只在 D 盘补装了 pypdf。
该环境依赖原有 agent-dev 环境，并非可独立搬走的环境。
`requirements-rag.txt` 提供独立重建环境时的依赖清单。

## 使用

1. 在“知识库管理”中选择 TXT、Markdown 或文字版 PDF，点击“导入 / 更新资料”。
2. 等待看到导入成功，再提问。默认启用“优先检索知识库”；闲聊可关闭。
3. 回答正文使用 `[资料1]` 等编号。回答下方列出本次检索到的候选资料，展开可核对原文。
   候选列表不表示每个片段都支持回答中的所有结论，以正文引用和实际原文为准。
4. “清空聊天记录”仅清空此入口的聊天记录，保留原始资料和向量库。

原文块、向量及来源位置一起保存在 ChromaDB。文档与问题都由 Ollama 的同一个 BGE-M3 模型编码。
当前使用稠密向量和余弦距离，每次最多检索 4 块；尚未加入重排、BM25 或相关性阈值。
即使资料无关也可能返回相近候选，模型应在依据不足时说明，引用需要核对。
当前开关启用时每次提问都检索，不依赖模型自主决定是否调用检索工具。

## 文件位置与更新规则

- `knowledge/`：导入的原始文档，按文件名识别同一文档。
- `knowledge/.versions/<内容哈希>/`：同名更新时保留旧原文；不同资料请使用不同文件名。
- `data/chroma/`：持久化向量库；不要在程序运行时手工修改里面的文件。
- `.cache/gradio/`：网页上传缓存。
- `sessions/knowledge_history.json`：新助手专用聊天记录，不覆盖旧版 `history.json`。
- `.venv/`：D 盘虚拟环境。

相同文件重复导入会跳过；同名文件内容变化后重新生成向量，成功后清理旧块。
删除原始文件不会自动删除向量记录；本版没有文档删除界面。
单文件限制 30 MiB。PDF 标注的是文件页序号，不一定等于书页印刷页码。
Markdown 记录标题与行范围，TXT 记录行范围，另附块在该章节内的字符范围。
扫描 PDF 需先做 OCR。仅导入显式选择的文档，不扫描项目目录或 `.env`。

Ollama 已有模型继续由 Ollama 管理，不重复下载，也不自动迁移到 `models/`。
所有新增项目数据目录均已加入 `.gitignore`。

## 可选配置

在 `.env` 中按需增加（不修改也可运行）：

```dotenv
OLLAMA_BASE_URL=http://127.0.0.1:11434
RAG_EMBED_MODEL=bge-m3:latest
```

不要把已有 collection 切换到其他 embedding 模型；新模型需要独立库并重新索引。
本版面向单进程本机应用，不支持多个进程同时导入同一库。
文档解析、向量生成及检索都在本机完成，但命中的文本块会发送到当前配置的大模型服务。

## 命令行导入与检查

```powershell
& .\.venv\Scripts\python.exe .\src\ingest.py .\knowledge\你的资料.md
& .\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

测试使用真实的本机 Ollama embedding 和独立临时向量库，需运行 Ollama。
测试文件与临时向量库位于 D 盘 `.cache/` 下，与正式库隔离。
