"""SQLite 短期记忆存储。

SQLite 只负责可靠保存会话和消息；长期记忆提取、向量化和事实冲突处理暂不在本模块实现。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading

if __package__:
    from .config import logger
else:
    from config import logger


class MemoryStore:
    def __init__(self, database_path: str | Path, legacy_json_path: str | Path | None = None):
        self.database_path = Path(database_path).resolve()
        self.legacy_json_path = Path(legacy_json_path).resolve() if legacy_json_path else None
        self._lock = threading.RLock()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self):
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '默认对话',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_time
                    ON messages(conversation_id, id);
                CREATE TABLE IF NOT EXISTS summaries (
                    conversation_id INTEGER PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)
            now = self._now()
            connection.execute("""INSERT OR IGNORE INTO conversations(id, title, created_at, updated_at)
                VALUES (1, '默认对话', ?, ?)""", (now, now))
        self._migrate_legacy_json()

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _clean_messages(messages):
        cleaned = []
        for item in messages if isinstance(messages, list) else []:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant", "system"} and isinstance(content, str) and content.strip():
                cleaned.append({"role": role, "content": content})
        return cleaned

    def _migrate_legacy_json(self):
        """首次创建数据库时导入旧 JSON；保留原文件，重复启动不会重复插入。"""
        if not self.legacy_json_path or not self.legacy_json_path.is_file():
            return
        with self._lock, self._connect() as connection:
            if connection.execute("SELECT 1 FROM metadata WHERE key = 'legacy_json_migrated'").fetchone():
                return
            try:
                data = json.loads(self.legacy_json_path.read_text(encoding="utf-8"))
                messages = self._clean_messages(data)
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                logger.warning("旧聊天 JSON 未迁移，文件保留：%s", type(error).__name__)
                return
            now = self._now()
            with connection:
                connection.execute("DELETE FROM messages WHERE conversation_id = 1")
                connection.executemany(
                    "INSERT INTO messages(conversation_id, role, content, created_at) VALUES (1, ?, ?, ?)",
                    [(item["role"], item["content"], now) for item in messages],
                )
                connection.execute("INSERT INTO metadata(key, value) VALUES ('legacy_json_migrated', ?)",
                                   (str(self.legacy_json_path),))
                connection.execute("UPDATE conversations SET updated_at = ? WHERE id = 1", (now,))
            logger.info("已将旧聊天 JSON 迁移到 SQLite，消息数=%d", len(messages))

    def load_messages(self) -> list[dict]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT role, content FROM messages WHERE conversation_id = 1 ORDER BY id"
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def append_turn(self, user_content: str, assistant_content: str):
        """一次事务追加一轮消息，避免分别写入造成半轮历史。"""
        if not user_content.strip() or not assistant_content.strip():
            return
        now = self._now()
        with self._lock, self._connect() as connection:
            with connection:
                connection.executemany(
                    "INSERT INTO messages(conversation_id, role, content, created_at) VALUES (1, ?, ?, ?)",
                    [("user", user_content, now), ("assistant", assistant_content, now)],
                )
                connection.execute("UPDATE conversations SET updated_at = ? WHERE id = 1", (now,))

    def clear(self):
        with self._lock, self._connect() as connection:
            with connection:
                connection.execute("DELETE FROM messages WHERE conversation_id = 1")
                connection.execute("DELETE FROM summaries WHERE conversation_id = 1")
                connection.execute("UPDATE conversations SET updated_at = ? WHERE id = 1", (self._now(),))

    def message_count(self) -> int:
        with self._lock, self._connect() as connection:
            return connection.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = 1").fetchone()[0]
