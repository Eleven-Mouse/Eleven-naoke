"""语义检索：查询向量化 -> LanceDB 近邻 -> 带引用的可读结果。"""

from __future__ import annotations

from dataclasses import dataclass

from pkb.config import Config
from pkb.embedder import Embedder
from pkb.store import ChunkStore


@dataclass
class SearchHit:
    path: str
    title: str
    headings: str
    text: str
    score: float

    def format(self, snippet_chars: int = 300) -> str:
        snippet = self.text[:snippet_chars] + ("…" if len(self.text) > snippet_chars else "")
        return (
            f"{self.path} · {self.headings}（相关度 {self.score:.2f}）\n{snippet}"
        )


def search(
    config: Config,
    embedder: Embedder,
    query: str,
    top_k: int | None = None,
    collection: str | None = None,
    reranker=None,
) -> list[SearchHit]:
    k = top_k or config.default_top_k
    store = ChunkStore(config.index_dir, embedder.dim)
    if store.count() == 0:
        return []
    vector = embedder.embed([query])[0]
    # 有重排器时多召回一批候选，精排后再截断到 top_k（候选量控制 CPU 推理耗时）
    fetch_k = k if reranker is None else min(max(k * 3, 12), 24)
    rows = store.search(vector, top_k=fetch_k, collection=collection)
    hits = [
        SearchHit(
            path=row["path"],
            title=row["title"],
            headings=row["headings"],
            text=row["text"],
            # LanceDB 返回 L2 距离；归一化向量下 dist² ∈ [0,4]，折算为 [0,1] 相关度
            score=max(0.0, 1.0 - float(row.get("_distance", 0.0)) ** 2 / 4),
        )
        for row in rows
    ]
    if reranker is not None:
        scores = reranker.rerank(query, [hit.text for hit in hits])
        reranked = sorted(
            zip(hits, (float(s) for s in scores), strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        hits = [
            SearchHit(**{**vars(hit), "score": score})
            for hit, score in reranked[:k]
        ]
    return hits


def format_results(hits: list[SearchHit]) -> str:
    if not hits:
        return "没有命中的知识。请先执行 kb index 建立索引。"
    return "\n\n".join(f"{i}. {hit.format()}" for i, hit in enumerate(hits, 1))
