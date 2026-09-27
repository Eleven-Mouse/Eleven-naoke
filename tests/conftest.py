"""测试夹具：确定性 Stub 向量化器 + 临时知识库。"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from pkb.config import Config, save_config

DIM = 64


class StubEmbedder:
    """基于文本哈希的确定性向量化，测试无需下载模型。"""

    dim = DIM

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % DIM
                rows[i, index] += 1.0
            norm = np.linalg.norm(rows[i])
            if norm > 0:
                rows[i] /= norm
        return rows


@pytest.fixture
def stub_embedder() -> StubEmbedder:
    return StubEmbedder()


@pytest.fixture
def kb_root(tmp_path) -> object:
    """一个已初始化、含两个 demo 页面的知识库。"""
    config = Config(root=tmp_path)
    save_config(config)
    demo = config.wiki_dir / "demo"
    demo.mkdir(parents=True)
    (demo / "module-订单超时.md").write_text(
        "---\ntitle: 订单超时处理\ntags: [module]\n---\n"
        "# 订单超时处理\n\n## 现象\n\n订单提交后超过 30 秒未返回。\n\n## 处理口径\n\n先查支付回调日志，再核对超时配置 CALLBACK_TIMEOUT。\n",
        encoding="utf-8",
    )
    (demo / "concept-重试策略.md").write_text(
        "---\ntitle: 重试策略\n---\n"
        "# 重试策略\n\n幂等接口允许三次重试，非幂等接口禁止自动重试。\n",
        encoding="utf-8",
    )
    return tmp_path
