"""Markdown 文档与 frontmatter 解析。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Document:
    path: Path
    collection: str
    title: str
    body: str
    metadata: dict = field(default_factory=dict)

    @property
    def rel_path(self) -> str:
        return self.path.as_posix()


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 `---` 包裹的 YAML frontmatter，返回 (metadata, body)。无 frontmatter 时返回空 dict。"""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            block = "\n".join(lines[1:i])
            body = "\n".join(lines[i + 1 :])
            try:
                meta = yaml.safe_load(block)
            except yaml.YAMLError:
                meta = {}
            return (meta if isinstance(meta, dict) else {}), body
    return {}, text


def load_document(path: Path, root: Path, collection: str) -> Document:
    text = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text)
    # 标题优先级：frontmatter title > 一级标题 > 文件名
    title = str(meta.get("title") or "").strip()
    if not title:
        for line in body.splitlines():
            if line.startswith("# ") and line[2:].strip():
                title = line[2:].strip()
                break
    if not title:
        title = path.stem
    return Document(
        path=path.relative_to(root),
        collection=collection,
        title=title,
        body=body,
        metadata=meta,
    )
