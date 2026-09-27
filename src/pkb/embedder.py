"""向量化封装：可插拔后端。

- fastembed：轻量 ONNX 路线（默认小模型，CPU 毫秒级）
- sentence-transformers：完整权重路线，支持 google/embeddinggemma-300m 等原版模型
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """向量化接口，便于测试时注入确定性实现。"""

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class FastEmbedder:
    """fastembed ONNX 后端（如 BAAI/bge-small-zh-v1.5）。"""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self.dim = len(next(iter(self._model.embed(["维度探测"]))))
        self.model_name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.array(list(self._model.embed(texts)), dtype=np.float32)


class STEmbedder:
    """sentence-transformers 后端（如 google/embeddinggemma-300m）。

    模型自带 query/document 提示词模板时会自动启用：查询单条时用 query 模板，
    索引文档时用默认编码，对检索质量有实际影响。
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        if hasattr(self._model, "get_embedding_dimension"):
            self.dim = self._model.get_embedding_dimension()
        else:
            self.dim = self._model.get_sentence_embedding_dimension()
        self.model_name = model_name
        self._query_prompt = self._detect_query_prompt()

    def _detect_query_prompt(self) -> str | None:
        try:
            prompts = self._model.prompts or {}
        except Exception:
            return None
        for key in ("query", "search_query"):
            if key in prompts:
                return key
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # 仅单条（即真实查询）时使用 query 提示词；批量（索引文档）走默认模板
        if self._query_prompt and len(texts) == 1:
            return np.array(
                self._model.encode(texts, prompt_name=self._query_prompt),
                dtype=np.float32,
            )
        return np.array(self._model.encode(texts), dtype=np.float32)


def create_embedder(model_name: str, backend: str = "fastembed") -> Embedder:
    if backend == "sentence-transformers":
        return STEmbedder(model_name)
    if backend == "fastembed":
        return FastEmbedder(model_name)
    raise ValueError(f"未知向量化后端：{backend}（可选 fastembed / sentence-transformers）")
