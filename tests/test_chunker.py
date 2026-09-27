from pathlib import Path

from pkb.chunker import chunk_document
from pkb.documents import Document


def make_doc(body: str, title: str = "示例页") -> Document:
    return Document(
        path=Path("wiki/demo/x.md"),
        collection="demo",
        title=title,
        body=body,
    )


def test_split_by_headings_with_breadcrumb():
    doc = make_doc("# 主标题\n## 部署\nWindows 使用 zip 包。\n## 升级\n先备份数据。")
    chunks = chunk_document(doc)
    headings = [c.headings for c in chunks]
    assert "主标题 > 部署" in headings
    assert "主标题 > 升级" in headings
    deploy = next(c for c in chunks if c.headings == "主标题 > 部署")
    assert "Windows 使用 zip 包" in deploy.text
    # 文本前缀带完整面包屑，增强检索语义
    assert deploy.text.startswith("[示例页 > 主标题 > 部署]")


def test_title_level_heading_not_duplicated():
    doc = make_doc("# 示例页\n## 部署\nWindows 使用 zip 包。")
    chunks = chunk_document(doc)
    deploy = next(c for c in chunks if "部署" in c.headings)
    assert deploy.headings == "部署"
    assert deploy.text.startswith("[示例页 > 部署]")


def test_long_section_split_by_paragraphs():
    long_para = "很长的段落。" * 120
    doc = make_doc(f"# 标题\n\n{long_para}\n\n第二段内容。\n\n第三段内容。")
    chunks = chunk_document(doc, max_chars=200)
    assert len(chunks) > 1


def test_preamble_content_gets_title_breadcrumb():
    doc = make_doc("开头没有标题的导语。\n\n# 第一章\n正文。")
    chunks = chunk_document(doc)
    assert chunks[0].headings == "示例页"
    assert any(c.headings == "第一章" for c in chunks)
