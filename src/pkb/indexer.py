"""增量索引：扫描 wiki/ 下的 Markdown，按文件哈希对比，只重建有变化的文件。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pkb.chunker import chunk_document
from pkb.config import Config
from pkb.documents import load_document
from pkb.embedder import Embedder
from pkb.store import ChunkStore


@dataclass
class IndexReport:
    added: list[str]
    updated: list[str]
    removed: list[str]
    total_chunks: int

    @property
    def unchanged(self) -> bool:
        return not (self.added or self.updated or self.removed)

    def summary(self) -> str:
        if self.unchanged:
            return f"索引无变化，共 {self.total_chunks} 个知识块。"
        parts = []
        if self.added:
            parts.append(f"新增 {len(self.added)}：{self._names(self.added)}")
        if self.updated:
            parts.append(f"更新 {len(self.updated)}：{self._names(self.updated)}")
        if self.removed:
            parts.append(f"移除 {len(self.removed)}：{self._names(self.removed)}")
        return "；".join(parts) + f"。当前共 {self.total_chunks} 个知识块。"

    @staticmethod
    def _names(paths: list[str]) -> str:
        return "、".join(Path(p).name for p in paths[:5]) + ("…" if len(paths) > 5 else "")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan_documents(config: Config) -> dict[str, tuple[Path, str, str]]:
    """扫描 wiki/，返回 rel_path -> (abs_path, collection, hash)。"""
    found: dict[str, tuple[Path, str, str]] = {}
    wiki = config.wiki_dir
    if not wiki.is_dir():
        return found
    for md in wiki.rglob("*.md"):
        collection = md.relative_to(wiki).parts[0]
        rel = md.relative_to(config.root).as_posix()
        found[rel] = (md, collection, file_hash(md))
    return found


def build_index(config: Config, embedder: Embedder) -> IndexReport:
    docs = scan_documents(config)
    store = ChunkStore(config.index_dir, embedder.dim)
    indexed = store.file_hashes()

    to_write = [rel for rel, (_, _, h) in docs.items() if indexed.get(rel) != h]
    to_remove = [rel for rel in indexed if rel not in docs]

    store.delete_paths(to_write + to_remove)

    written: dict[str, list] = {}
    for rel in to_write:
        abs_path, collection, _ = docs[rel]
        doc = load_document(abs_path, config.root, collection)
        written[rel] = chunk_document(doc, config.chunk_max_chars)

    # 按文件批量向量化后写入
    for rel, chunks in written.items():
        _, collection, h = docs[rel]
        vectors = embedder.embed([c.text for c in chunks])
        store.add_chunks(chunks, vectors, h)

    removed_existing = [rel for rel in to_remove]
    return IndexReport(
        added=[rel for rel in to_write if rel not in indexed],
        updated=[rel for rel in to_write if rel in indexed],
        removed=removed_existing,
        total_chunks=store.count(),
    )
