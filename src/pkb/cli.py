"""kb 命令行入口。

常用命令：
    kb init                  初始化知识库骨架
    kb index                 增量建立/刷新索引
    kb search "问题"          语义检索知识
    kb get <路径>             查看知识页面
    kb status                索引状态与待提交知识提醒
    kb mcp                   启动 MCP Server（供 Claude Code 等调用）
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer

from pkb.config import (
    DEFAULT_COLLECTION,
    KBError,
    Config,
    collections,
    find_root,
    load_config,
    save_config,
)
from pkb.embedder import create_embedder
from pkb.indexer import build_index, scan_documents
from pkb.reranker import create_reranker
from pkb.retriever import format_results
from pkb.retriever import search as semantic_search


def _embedder(config: Config):
    typer.echo(f"加载向量化模型 {config.model}（backend={config.backend}，首次自动下载）…")
    return create_embedder(config.model, config.backend)

app = typer.Typer(help="pkb — 个人本地 RAG 知识库", no_args_is_help=True)


def _root() -> Path:
    return find_root()


def _config() -> Config:
    return load_config(_root())


@app.command()
def init(
    collection: str = typer.Option(DEFAULT_COLLECTION, "--collection", "-c", help="初始知识集合名"),
):
    """在当前目录初始化知识库。"""
    root = Path.cwd()
    config = Config(root=root)
    save_config(config)
    coll_dir = config.wiki_dir / collection
    coll_dir.mkdir(parents=True, exist_ok=True)
    index_page = coll_dir / "index.md"
    if not index_page.exists():
        index_page.write_text(
            f"# {collection} 知识索引\n\n在此登记本集合的知识页面。所有 Markdown 文件都会被 `kb index` 自动索引。\n",
            encoding="utf-8",
        )
    typer.echo(f"知识库已初始化：{root}")
    typer.echo(f"知识目录：wiki/{collection}/（把 Markdown 放进来，然后执行 kb index）")
    typer.echo("下一步：kb index && kb search \"你的问题\"")


@app.command()
def index():
    """增量建立/刷新索引（只处理有变化的文件）。"""
    config = _config()
    docs = scan_documents(config)
    if not docs:
        typer.echo(f"wiki/ 下没有 Markdown 文件。把知识页面放入 {config.wiki_dir}/ 后重试。")
        raise typer.Exit(code=1)
    typer.echo(f"加载向量化模型 {config.model}（backend={config.backend}，首次自动下载，之后离线）…")
    embedder = create_embedder(config.model, config.backend)
    report = build_index(config, embedder)
    typer.echo(report.summary())


@app.command()
def search(
    query: str = typer.Argument(..., help="自然语言问题"),
    top_k: int = typer.Option(None, "--top-k", "-k", help="返回条数"),
    collection: str = typer.Option(None, "--collection", "-c", help="限定知识集合"),
    rerank: bool = typer.Option(
        None, "--rerank/--no-rerank", help="是否用重排模型精排（默认取配置 rerank）"
    ),
):
    """语义检索知识，返回带出处的片段。"""
    config = _config()
    use_rerank = config.rerank if rerank is None else rerank
    embedder = _embedder(config)
    reranker = None
    if use_rerank:
        typer.echo(f"加载重排模型 {config.reranker_model}…")
        reranker = create_reranker(config.reranker_model, config.reranker_temperature)
    hits = semantic_search(
        config, embedder, query, top_k=top_k, collection=collection, reranker=reranker
    )
    typer.echo(format_results(hits))


@app.command()
def get(path: str = typer.Argument(..., help="知识页面相对路径")):
    """查看某个知识页面全文。"""
    config = _config()
    target = (config.root / path).resolve()
    if not target.is_file() or config.wiki_dir not in target.parents:
        typer.echo(f"路径无效或不在 wiki/ 内：{path}")
        raise typer.Exit(code=1)
    typer.echo(target.read_text(encoding="utf-8"))


@app.command()
def status():
    """索引状态 + 待提交知识提醒（借鉴原项目：任务结束先看有没有未沉淀/未提交的知识）。"""
    config = _config()
    docs = scan_documents(config)
    typer.echo(f"知识库根目录：{config.root}")
    typer.echo(f"知识集合：{'、'.join(collections(config)) or '（无）'}")
    typer.echo(f"待索引 Markdown：{len(docs)} 个文件")

    if config.index_dir.is_dir():
        import lancedb

        try:
            table = lancedb.connect(str(config.index_dir)).open_table("chunks")
            rows = table.count_rows()
            if rows:
                typer.echo(f"索引：{rows} 个知识块。如 wiki/ 有改动，执行 kb index 增量刷新。")
            else:
                typer.echo("索引：空。执行 kb index 建立索引。")
        except Exception:
            typer.echo("索引：读取失败，可执行 kb index 重建。")

    # Git 待提交知识提醒
    if (config.root / ".git").exists():
        result = subprocess.run(
            ["git", "status", "--porcelain", "wiki/"],
            cwd=config.root,
            capture_output=True,
            text=True,
            check=False,
        )
        pending = [line for line in result.stdout.splitlines() if line.strip()]
        if pending:
            typer.echo(f"\n⚠ 有 {len(pending)} 个知识文件尚未提交：")
            for line in pending[:10]:
                typer.echo(f"  {line}")
            typer.echo("知识写入、提交、推送请分别确认，避免丢失沉淀。")


def main() -> None:
    try:
        app()
    except KBError as exc:
        typer.echo(f"错误：{exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

# 供 MCP 模块复用的检索便捷函数
def search_json(query: str, top_k: int | None = None, collection: str | None = None) -> str:
    config = _config()
    embedder = create_embedder(config.model, config.backend)
    reranker = (
        create_reranker(config.reranker_model, config.reranker_temperature)
        if config.rerank
        else None
    )
    hits = semantic_search(
        config, embedder, query, top_k=top_k, collection=collection, reranker=reranker
    )
    return json.dumps(
        [
            {"path": h.path, "headings": h.headings, "score": round(h.score, 3), "text": h.text}
            for h in hits
        ],
        ensure_ascii=False,
        indent=2,
    )
