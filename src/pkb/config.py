"""配置与路径管理。

知识库根目录由 `.kb/config.json` 标记；知识内容放在 `wiki/<collection>/` 下，
collection 即目录名，默认 `demo`。索引状态存放在 `.kb/lancedb/`。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = ".kb"
CONFIG_FILE = "config.json"
WIKI_DIR = "wiki"
INDEX_DIR = "lancedb"
DEFAULT_COLLECTION = "demo"
DEFAULT_MODEL = "google/embeddinggemma-300m"
DEFAULT_BACKEND = "sentence-transformers"
DEFAULT_RERANKER = "Qwen/Qwen3-Reranker-0.6B"


class KBError(Exception):
    """知识库操作错误，带用户可读的中文信息。"""


@dataclass
class Config:
    root: Path
    model: str = DEFAULT_MODEL
    backend: str = DEFAULT_BACKEND
    rerank: bool = True
    reranker_model: str = DEFAULT_RERANKER
    default_top_k: int = 5
    chunk_max_chars: int = 800
    excluded_patterns: list[str] = field(default_factory=list)

    @property
    def wiki_dir(self) -> Path:
        return self.root / WIKI_DIR

    @property
    def index_dir(self) -> Path:
        return self.root / CONFIG_DIR / INDEX_DIR

    @property
    def config_file(self) -> Path:
        return self.root / CONFIG_DIR / CONFIG_FILE


def find_root(start: Path | None = None) -> Path:
    """从 start（默认当前目录）向上查找包含 .kb/config.json 的目录。"""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / CONFIG_DIR / CONFIG_FILE).is_file():
            return candidate
    raise KBError(
        "未找到知识库（缺少 .kb/config.json）。请先在知识库根目录执行 kb init。"
    )


def load_config(root: Path) -> Config:
    raw = json.loads((root / CONFIG_DIR / CONFIG_FILE).read_text(encoding="utf-8"))
    return Config(
        root=root,
        model=raw.get("model", DEFAULT_MODEL),
        backend=raw.get("backend", DEFAULT_BACKEND),
        rerank=raw.get("rerank", True),
        reranker_model=raw.get("reranker_model", DEFAULT_RERANKER),
        default_top_k=raw.get("default_top_k", 5),
        chunk_max_chars=raw.get("chunk_max_chars", 800),
        excluded_patterns=raw.get("excluded_patterns", []),
    )


def save_config(config: Config) -> None:
    config.config_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": config.model,
        "backend": config.backend,
        "rerank": config.rerank,
        "reranker_model": config.reranker_model,
        "default_top_k": config.default_top_k,
        "chunk_max_chars": config.chunk_max_chars,
        "excluded_patterns": config.excluded_patterns,
    }
    config.config_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def collections(config: Config) -> list[str]:
    """列出 wiki/ 下所有 collection（子目录名，按名称排序）。"""
    wiki = config.wiki_dir
    if not wiki.is_dir():
        return []
    return sorted(p.name for p in wiki.iterdir() if p.is_dir())
