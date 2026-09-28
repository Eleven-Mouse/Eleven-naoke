"""重排封装：Qwen3-Reranker 官方权重（原项目用的是它的 GGUF 量化版，这里是同一模型）。

按官方模型卡实现：checkpoint 本体是 CausalLM（不含分类头），通过比较最后一个
token 处 "yes"/"no" 的概率得出相关度。Cross-encoder 逐条精排比向量检索更准，
代价是每条候选一次前向推理，只在查询时对少量候选使用。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"

# 温度缩放：真实查询上 yes/no logit 差约在 ±7.5，除以 T=2.5 后落在 ±3，
# sigmoid 展开到 ~0.05–0.95，避免强匹配全部饱和成 1.00、无法区分优劣。
DEFAULT_TEMPERATURE = 2.5

# Qwen3-Reranker 模型卡要求的对话模板
_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on "
    'the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Reranker(Protocol):
    def rerank(self, query: str, texts: list[str]) -> np.ndarray: ...


def yes_no_score(true_v, false_v, temperature: float = DEFAULT_TEMPERATURE) -> np.ndarray:
    """yes/no logit 差经温度缩放后过 sigmoid，映射到 [0,1] 相关度。

    接受 torch 张量或 ndarray，返回 float64 ndarray；独立成纯函数便于单测。
    """
    z = (true_v - false_v).detach().cpu().numpy() if hasattr(true_v, "detach") else np.asarray(true_v) - np.asarray(false_v)
    return 1.0 / (1.0 + np.exp(-z / temperature))


class Qwen3Reranker:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        max_length: int = 1024,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self._model.eval()
        self.model_name = model_name
        self._max_length = max_length
        self._true_id = self._tokenizer.encode("yes", add_special_tokens=False)[0]
        self._false_id = self._tokenizer.encode("no", add_special_tokens=False)[0]
        self.temperature = temperature

    def rerank(self, query: str, texts: list[str]) -> np.ndarray:
        """返回每条 text 的相关度得分（logit 差温度缩放后 sigmoid 到 [0,1]）。"""
        if not texts:
            return np.zeros(0, dtype=np.float32)
        pairs = [
            f"{_PREFIX}<Instruct>: {_INSTRUCT}\n<Query>: {query}\n<Document>: {t}{_SUFFIX}"
            for t in texts
        ]
        inputs = self._tokenizer(
            pairs,
            padding=True,
            truncation="longest_first",
            max_length=self._max_length,
            return_tensors="pt",
        )
        with self._torch.no_grad():
            # logits_to_keep=1：只算最后一个位置，避免整段序列 × 15 万词表的内存开销
            try:
                logits = self._model(**inputs, logits_to_keep=1).logits[:, -1, :]
            except TypeError:  # 兼容旧版参数名
                logits = self._model(**inputs, num_logits_to_keep=1).logits[:, -1, :]
        true_v = logits[:, self._true_id]
        false_v = logits[:, self._false_id]
        return yes_no_score(true_v, false_v, self.temperature)


def create_reranker(model_name: str, temperature: float = DEFAULT_TEMPERATURE) -> Reranker:
    return Qwen3Reranker(model_name, temperature=temperature)
