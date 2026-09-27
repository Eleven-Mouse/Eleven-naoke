"""按 Markdown 标题结构切分文档，保留标题面包屑作为检索上下文。

设计理念来自"知识对象"：一个标题章节通常对应一个可独立理解的结论，
切分时保持章节完整；超长章节再按空行段落二次切分。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pkb.documents import Document

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    text: str
    path: str
    collection: str
    title: str
    headings: str  # 标题面包屑，如 "部署 > Windows"


def _fallback_split(section: str, max_chars: int) -> list[str]:
    """按空行段落聚合切分超长章节，尽量不超过 max_chars。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section) if p.strip()]
    pieces: list[str] = []
    buf = ""
    for para in paragraphs:
        candidate = f"{buf}\n\n{para}" if buf else para
        if buf and len(candidate) > max_chars:
            pieces.append(buf)
            buf = para
        else:
            buf = candidate
    if buf:
        pieces.append(buf)
    return pieces or [section.strip()]


def chunk_document(doc: Document, max_chars: int = 800) -> list[Chunk]:
    """把文档切分为 Chunk 列表；每个 chunk 的文本前缀标题面包屑以增强语义。"""
    chunks: list[Chunk] = []
    # 逐行扫描，按最近一级标题聚合成 section；标题面包屑记录各级最近标题
    trail: dict[int, str] = {}
    current_headings: list[str] = []
    current_lines: list[str] = []
    preamble = True  # 首个标题之前的内容

    def flush() -> None:
        nonlocal current_lines
        section = "\n".join(current_lines).strip()
        current_lines = []
        if not section:
            return
        breadcrumb = " > ".join(current_headings) if current_headings else doc.title
        for piece in _fallback_split(section, max_chars):
            text = f"[{doc.title} > {breadcrumb}]\n{piece}" if current_headings else f"[{doc.title}]\n{piece}"
            chunks.append(
                Chunk(
                    text=text,
                    path=doc.rel_path,
                    collection=doc.collection,
                    title=doc.title,
                    headings=breadcrumb,
                )
            )

    for line in doc.body.splitlines():
        match = _HEADING.match(line)
        if match:
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            flush()  # 首个标题前的导语也要独立成块
            # 更新面包屑：丢弃 >= 当前级别的历史标题；与页面标题相同的一级标题不再重复
            trail = {lv: h for lv, h in trail.items() if lv < level}
            trail[level] = heading_text
            current_headings = [
                h for lv, h in trail.items() if not (lv == 1 and h == doc.title)
            ]
            preamble = False
        current_lines.append(line)
    flush()
    return chunks
