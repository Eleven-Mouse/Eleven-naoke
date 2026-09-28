"""重排链路测试：用确定性 Stub 重排器验证召回扩量、精排排序和截断。"""

import numpy as np

from pkb.config import load_config
from pkb.indexer import build_index
from pkb.reranker import DEFAULT_TEMPERATURE, yes_no_score
from pkb.retriever import search


class StubReranker:
    """把包含关键词的文本排前的确定性重排器。"""

    def rerank(self, query: str, texts: list[str]) -> np.ndarray:
        keyword = query.strip()
        return np.array([1.0 if keyword in t else 0.1 for t in texts], dtype=np.float32)


def test_reranker_reorders_and_truncates(kb_root, stub_embedder):
    config = load_config(kb_root)
    build_index(config, stub_embedder)

    hits = search(
        config,
        stub_embedder,
        "非幂等",
        top_k=2,
        reranker=StubReranker(),
    )
    # 候选超过 top_k，重排后只保留 top_k 条
    assert len(hits) == 2
    # 排在第一的文本必须真的包含关键词（重排生效）
    assert "非幂等" in hits[0].text
    # 分数来自重排器而不是向量距离
    assert hits[0].score == 1.0
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_yes_no_score_temperature_scaling():
    """温度缩放：裸 logit 差饱和时仍能区分优劣，且温度越高分布越平。"""
    true_v = np.array([7.3, -7.7])  # 真实查询上观测到的强匹配/不匹配量级
    false_v = np.array([0.0, 0.0])

    scores = yes_no_score(true_v, false_v)
    # 默认温度下展开到 ~0.05–0.95，不再饱和成 1.00/0.00
    assert 0.9 < scores[0] < 0.99
    assert 0.01 < scores[1] < 0.1
    assert scores[0] > scores[1]

    # 温度越高，分数越向中间收拢
    flat = yes_no_score(true_v, false_v, temperature=DEFAULT_TEMPERATURE * 10)
    assert (flat[0] - flat[1]) < (scores[0] - scores[1])
