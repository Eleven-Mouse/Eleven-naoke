from pathlib import Path

from pkb.documents import load_document, parse_frontmatter


def test_parse_frontmatter_basic():
    meta, body = parse_frontmatter("---\ntitle: 示例\ntags: [a, b]\n---\n正文内容")
    assert meta["title"] == "示例"
    assert meta["tags"] == ["a", "b"]
    assert body.strip() == "正文内容"


def test_parse_frontmatter_missing():
    meta, body = parse_frontmatter("# 无 frontmatter\n正文")
    assert meta == {}
    assert body.startswith("# 无 frontmatter")


def test_load_document_title_fallback(tmp_path: Path):
    f = tmp_path / "page.md"
    f.write_text("# 一级标题优先\n内容", encoding="utf-8")
    doc = load_document(f, tmp_path, "demo")
    assert doc.title == "一级标题优先"
    assert doc.rel_path == "page.md"
    assert doc.collection == "demo"


def test_load_document_filename_fallback(tmp_path: Path):
    f = tmp_path / "concept-无标题.md"
    f.write_text("没有标题的正文", encoding="utf-8")
    doc = load_document(f, tmp_path, "demo")
    assert doc.title == "concept-无标题"
