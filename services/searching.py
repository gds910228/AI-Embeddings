from __future__ import annotations
from typing import Any, Dict, List

from services.vector_store import query_by_embeddings


def search_kb(
    *,
    kb: str,
    query: str,
    embed_single_fn,
    top_k: int = 5,
    model: str = "embedding-3",
) -> Dict[str, Any]:
    """
    在指定知识库(kb)中进行语义搜索
    - embed_single_fn: 可调用(text: str, model: str) -> List[float]
    """
    if not query:
        return {"success": False, "error": "查询文本不能为空"}

    # 生成查询向量
    query_embedding = embed_single_fn(query, model)

    # 在向量库中检索
    res = query_by_embeddings(kb, [query_embedding], n_results=top_k)

    docs_list = res.get("documents", [[]])
    metas_list = res.get("metadatas", [[]])
    ids_list = res.get("ids", [[]])
    dists_list = res.get("distances", [[]])

    docs = docs_list[0] if docs_list else []
    metas = metas_list[0] if metas_list else []
    ids = ids_list[0] if ids_list else []
    dists = dists_list[0] if dists_list else []

    results: List[Dict[str, Any]] = []
    for i, text in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        item = {
            "rank": i + 1,
            "id": ids[i] if i < len(ids) else None,
            "text": text,
            "source": meta.get("source"),
            "kb": meta.get("kb"),
            "chunk_index": meta.get("chunk_index"),
        }
        if i < len(dists) and dists[i] is not None:
            item["distance"] = dists[i]
        results.append(item)

    return {
        "success": True,
        "kb": kb,
        "query": query,
        "model": model,
        "top_k": top_k,
        "results": results,
        "count": len(results),
    }