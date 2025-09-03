from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.api.models.Collection import Collection
except Exception as e:  # pragma: no cover
    chromadb = None
    Collection = Any  # type: ignore


_PERSIST_DIR = Path("outputs") / "chroma_db"
_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

_client = None  # lazy init


def _ensure_client():
    global _client
    if _client is None:
        if chromadb is None:
            raise RuntimeError("未安装 chromadb，请先安装：pip install chromadb")
        # 使用本地持久化目录，便于演示与复用
        _client = chromadb.PersistentClient(path=str(_PERSIST_DIR))
    return _client


def get_or_create_collection(name: str) -> "Collection":
    client = _ensure_client()
    # 避免重复创建
    existing = None
    try:
        existing = client.get_collection(name)
    except Exception:
        existing = None
    if existing:
        return existing
    return client.create_collection(name=name, metadata={"kb": name})


def add_texts(
    name: str,
    documents: List[str],
    ids: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    embeddings: Optional[List[List[float]]] = None,
):
    """
    向集合写入文档片段（可传入预计算的嵌入向量）
    """
    if not documents:
        return
    col = get_or_create_collection(name)
    col.add(documents=documents, ids=ids, metadatas=metadatas, embeddings=embeddings)


def query_by_embeddings(
    name: str,
    query_embeddings: List[List[float]],
    n_results: int = 5,
) -> Dict[str, Any]:
    col = get_or_create_collection(name)
    res = col.query(query_embeddings=query_embeddings, n_results=n_results)
    # 统一输出结构
    return {
        "ids": res.get("ids", []),
        "documents": res.get("documents", []),
        "metadatas": res.get("metadatas", []),
        "distances": res.get("distances", []),  # 越小越相似（若启用）
    }