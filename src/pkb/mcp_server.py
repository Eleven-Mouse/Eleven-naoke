"""MCP Server：把知识库检索暴露为标准 MCP 工具，供 Claude Code 等客户端调用。

注册方式（在知识库根目录执行）：
    claude mcp add kb -- uv run kb mcp
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from pkb.cli import search_json
from pkb.config import collections, find_root, load_config

mcp = FastMCP("pkb")


@mcp.tool()
def kb_search(query: str, collection: str = "", top_k: int = 5) -> str:
    """在个人知识库中语义检索知识，返回带出处的 Markdown 片段。

    Args:
        query: 自然语言问题或关键词。
        collection: 限定知识集合（wiki/ 下的子目录名），留空检索全部。
        top_k: 返回条数，默认 5。
    """
    return search_json(query, top_k=top_k, collection=collection or None)


@mcp.tool()
def kb_collections() -> str:
    """列出知识库中所有可用的知识集合。"""
    config = load_config(find_root())
    names = collections(config)
    return "、".join(names) if names else "（知识库为空）"


if __name__ == "__main__":
    mcp.run()
