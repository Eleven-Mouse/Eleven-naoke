# 模型接入踩坑记录

接入选型模型时遇到的两个真实深坑。如果你也要在自己的 RAG 项目里用这两个模型，希望你能少走弯路。

## 坑 1：Qwen3-Reranker 的 checkpoint 不含分类头

### 现象

用最直觉的方式加载重排模型：

```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen3-Reranker-0.6B")
```

**能跑通，不报错**，加载报告里有一行容易被忽略的提示：

```
score.weight | MISSING
```

之后每次 `rerank` 都能返回 [0,1] 的"相关度"，但得分完全不可信——`score.weight` 是**随机初始化**的分类头，输出的是噪声。

### 根因

查证官方 checkpoint（310 个张量里没有任何 `score` 键，`config.json` 的 `architectures` 是 `Qwen3ForCausalLM`）可知：**Qwen3-Reranker 的本体是一个 CausalLM**，官方训练的不是"打分头"，而是让模型在 prompt 末尾输出 "yes" 或 "no" 的倾向。

### 正确做法

按官方模型卡的方法，用 CausalLM 比较最后一个 token 处 "yes"/"no" 的概率：

```python
logits = model(**inputs, logits_to_keep=1).logits[:, -1, :]   # 只取最后位置
true_v = logits[:, tokenizer.encode("yes", add_special_tokens=False)[0]]
false_v = logits[:, tokenizer.encode("no", add_special_tokens=False)[0]]
scores = softmax([false_v, true_v], dim=1)[:, 1]
```

三个配套细节：

1. **`logits_to_keep=1` 必须加**。不加的话 transformers 会算整段序列 × 15 万词表的完整 logits——16 条候选 × 900 token 在 CPU 上要好几 GB 内存。
2. **Qwen tokenizer 默认没有 pad_token**，批量推理前要手动 `tokenizer.pad_token = tokenizer.eos_token`（有的版本还会要求 `model.config.pad_token_id` 也设置）。
3. prompt 必须严格用官方模板（system 判定指令 + `<think></think>` 后缀），模板变了得分就不可比。完整实现见 `src/pkb/reranker.py`。

**教训**：加载报告的 MISSING key 不是警告级别的问题，它意味着这一层是随机的。模型"能跑出结果"和"结果是训练过的"是两回事。

## 坑 2：EmbeddingGemma 在 HuggingFace 是门控模型

### 现象

```python
SentenceTransformer("google/embeddinggemma-300m")
```

报错：

```
Access to model google/embeddinggemma-300m is restricted.
You must have access to it and be authenticated to access it.
```

### 根因

Google 发在 HuggingFace 的 Gemma 系模型都是**门控（gated）仓库**：必须登录 HF 账号、在模型页点击接受许可协议之后才有下载权限。自动化脚本、CI、给新人的一键 setup 都会被这里卡住。企业内部工具链"由负责人单独分发模型文件"的做法，本质上就是在绕这个问题。

### 解法

- **开箱即用**：社区镜像（如 `unsloth/embeddinggemma-300m`）权重相同、无门控。注意：权重相同不等于许可豁免，Gemma 的使用条款依然适用，商用前自己确认。
- **合规用官方源**：在模型页接受许可后 `huggingface-cli login`，把模型 id 改回 `google/embeddinggemma-300m` 即可。

**教训**：给团队/用户做工具链时，"模型能不能被下载"和"模型下载下来能不能跑"是两个独立的风险点，都要在 setup 阶段验证。

## 关联

- 本仓库的模型配置说明见 [README 的模型配置章节](../README.md#模型配置)
- 重排器的完整实现：`src/pkb/reranker.py`
