"""LanceDB 向量库封装：单表存 chunk，支持增量增删和带过滤的向量检索。"""

from __future__ import annotations

from pathlib import Path

import lancedb
import pyarrow as pa

from pkb.chunker import Chunk

_TABLE = "chunks"

_SCHEMA = pa.schema(
    [
        ("vector", pa.list_(pa.float32(), 512)),  # 定长在 open 时按实际维度重建
        ("text", pa.string()),
        ("path", pa.string()),
        ("collection", pa.string()),
        ("title", pa.string()),
        ("headings", pa.string()),
        ("file_hash", pa.string()),
    ]
)


class ChunkStore:
    def __init__(self, index_dir: Path, dim: int):
        self._db = lancedb.connect(str(index_dir))
        self.dim = dim
        schema = pa.schema([("vector", pa.list_(pa.float32(), dim)), *list(_SCHEMA)[1:]])
        try:
            self._table = self._db.open_table(_TABLE)
        except Exception:
            self._table = self._db.create_table(_TABLE, schema=schema, mode="create")

    def count(self) -> int:
        return self._table.count_rows()

    def file_hashes(self) -> dict[str, str]:
        """返回已索引的 path -> file_hash 映射。"""
        rows = self._table.to_arrow().select(["path", "file_hash"]).to_pydict()
        return dict(zip(rows["path"], rows["file_hash"], strict=True))

    def add_chunks(self, chunks: list[Chunk], vectors, file_hash: str) -> None:
        rows = [
            {
                "vector": vec.tolist(),
                "text": chunk.text,
                "path": chunk.path,
                "collection": chunk.collection,
                "title": chunk.title,
                "headings": chunk.headings,
                "file_hash": file_hash,
            }
            for chunk, vec in zip(chunks, vectors, strict=True)
        ]
        if rows:
            self._table.add(rows)

    def delete_paths(self, paths: list[str]) -> None:
        if not paths:
            return
        quoted = ", ".join(f"'{p}'" for p in paths)
        self._table.delete(f"path IN ({quoted})")

    def search(self, query_vector, top_k: int, collection: str | None = None) -> list[dict]:
        query = self._table.search(query_vector.tolist()).limit(top_k)
        if collection:
            query = query.where(f"collection = '{collection}'")
        return query.to_list()
