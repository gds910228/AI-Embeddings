from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import uuid
import os
import re

import requests

from services.chunking import chunk_text
from services.vector_store import add_texts


ALLOWED_EXTS = {".md", ".txt"}


def _is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def _read_text_from_url(url: str) -> str:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    # 简单判断编码
    resp.encoding = resp.encoding or "utf-8"
    return resp.text


def _read_text_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def _iter_local_files(path: Path) -> List[Path]:
    files: List[Path] = []
    if path.is_file():
        if path.suffix.lower() in ALLOWED_EXTS:
            files.append(path)
    elif path.is_dir():
        for ext in ALLOWED_EXTS:
            files.extend(path.rglob(f"*{ext}"))
    return files


def _within_whitelist(p: Path, whitelist: List[Path]) -> bool:
    if not whitelist:
        return True
    p_abs = p.resolve()
    for w in whitelist:
        try:
            if p_abs.is_relative_to(w.resolve()):
                return True
        except AttributeError:
            # for Python <3.9: emulate is_relative_to
            try:
                p_abs.relative_to(w.resolve())
                return True
            except Exception:
                pass
    return False


def _load_whitelist_from_env() -> List[Path]:
    raw = os.getenv("ALLOW_INDEX_DIRS", "").strip()
    if not raw:
        return []
    parts = [x.strip() for x in raw.split(";") if x.strip()]
    return [Path(x) for x in parts]


def collect_sources(paths: List[str]) -> List[Tuple[str, str]]:
    """
    收集可索引的源：返回 (source, text)
    支持：
    - 本地文件或文件夹（.md/.txt）
    - URL（http/https）
    """
    sources: List[Tuple[str, str]] = []
    whitelist = _load_whitelist_from_env()

    for raw in paths:
        raw = raw.strip()
        if not raw:
            continue

        if _is_url(raw):
            try:
                text = _read_text_from_url(raw)
                sources.append((raw, text))
            except Exception as e:
                raise RuntimeError(f"获取URL失败: {raw} -> {e}") from e
            continue

        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"路径不存在: {raw}")

        if whitelist and not _within_whitelist(p, whitelist):
            raise PermissionError(f"路径不在白名单中: {raw}")

        for f in _iter_local_files(p):
            try:
                text = _read_text_file(f)
                sources.append((str(f), text))
            except Exception as e:
                raise RuntimeError(f"读取文件失败: {f} -> {e}") from e

    return sources


def index_to_chroma(
    *,
    kb: str,
    sources: List[Tuple[str, str]],
    embed_batch_fn,
    chunk_size: int = 500,
    overlap: int = 50,
    model: str = "embedding-3",
) -> Dict[str, Any]:
    """
    将 sources 写入向量库（自动选择 chroma 或本地 JSONL）：
    - 分块 -> 批量嵌入 -> 添加到集合
    embed_batch_fn: 可调用(texts: List[str], model: str) -> List[List[float]]
    """
    total_files = 0
    total_chunks = 0
    per_file_stats: List[Dict[str, Any]] = []

    for source, text in sources:
        total_files += 1
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            per_file_stats.append(
                {"source": source, "chunks": 0, "status": "empty"}
            )
            continue

        # 生成 ids 与元数据
        base_id = uuid.uuid4().hex[:8]
        ids = [f"{base_id}-{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": source, "kb": kb, "chunk_index": i} for i in range(len(chunks))
        ]

        # 批量嵌入
        embeddings = embed_batch_fn(chunks, model)

        # 写入
        add_texts(kb, documents=chunks, ids=ids, metadatas=metadatas, embeddings=embeddings)

        total_chunks += len(chunks)
        per_file_stats.append(
            {"source": source, "chunks": len(chunks), "status": "ok"}
        )

    return {
        "kb": kb,
        "files_indexed": total_files,
        "chunks_indexed": total_chunks,
        "details": per_file_stats,
        "model": model,
        "chunk_size": chunk_size,
        "overlap": overlap,
    }