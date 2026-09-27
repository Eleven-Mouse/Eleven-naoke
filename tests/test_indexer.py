from pkb.config import load_config
from pkb.indexer import build_index, scan_documents
from pkb.retriever import search


def test_scan_documents(kb_root):
    config = load_config(kb_root)
    docs = scan_documents(config)
    assert len(docs) == 2
    assert all(rel.startswith("wiki/demo/") for rel in docs)


def test_incremental_index_and_search(kb_root, stub_embedder):
    config = load_config(kb_root)
    report = build_index(config, stub_embedder)
    assert len(report.added) == 2
    assert report.total_chunks >= 2

    # 无变化时增量索引为空操作
    again = build_index(config, stub_embedder)
    assert again.unchanged

    # 语义命中
    hits = search(config, stub_embedder, "订单超时 CALLBAK_TIMEOUT", top_k=2)
    assert hits
    assert hits[0].path.endswith(".md")
    assert 0 < hits[0].score <= 1.0


def test_update_and_remove_detection(kb_root, stub_embedder):
    config = load_config(kb_root)
    build_index(config, stub_embedder)

    page = config.wiki_dir / "demo" / "concept-重试策略.md"
    page.write_text("# 重试策略\n\n更新后的口径：最多五次重试。\n", encoding="utf-8")
    report = build_index(config, stub_embedder)
    assert report.updated == ["wiki/demo/concept-重试策略.md"]

    page.unlink()
    report = build_index(config, stub_embedder)
    assert report.removed == ["wiki/demo/concept-重试策略.md"]


def test_collection_filter(kb_root, stub_embedder):
    config = load_config(kb_root)
    build_index(config, stub_embedder)

    other = config.wiki_dir / "notes"
    other.mkdir()
    (other / "memo.md").write_text("# 备忘\n\n订单超时关键词出现在另一集合。", encoding="utf-8")
    build_index(config, stub_embedder)

    hits = search(config, stub_embedder, "订单超时", top_k=5, collection="demo")
    assert hits
    assert all(h.path.startswith("wiki/demo/") for h in hits)
