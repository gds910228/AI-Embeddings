from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 尝试使用 chromadb；不可用则回退到本地 JSONL 存储
_HAS_CHROMA = False
try:
    import chromadb  # type: ignore
    from chromadb.api.models.Collection import Collection  # type: ignore
    _HAS_CHROMA = True
except Exception:
    _HAS_CHROMA = False

# -------- Chroma 后端（若可用） --------
if _HAS_CHROMA:
    _PERSIST_DIR = Path("outputs") / "chroma_db"
    _PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    _chroma_client = chromadb.PersistentClient(path=str(_PERSIST_DIR))

    def _get_or_create_collection(name: str) -> "Collection":
        try:
            return _chroma_client.get_collection(name)
        except Exception:
            return _chroma_client.create_collection(name=name, metadata={"kb": name})

    def add_texts(
        name: str,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ):
        if not documents:
            return
        col = _get_or_create_collection(name)
        col.add(documents=documents, ids=ids, metadatas=metadatas, embeddings=embeddings)

    def query_by_embeddings(
        name: str,
        query_embeddings: List[List[float]],
        n_results: int = 5,
    ) -> Dict[str, Any]:
        col = _get_or_create_collection(name)
        res = col.query(query_embeddings=query_embeddings, n_results=n_results)
        return {
            "ids": res.get("ids", []),
            "documents": res.get("documents", []),
            "metadatas": res.get("metadatas", []),
            "distances": res.get("distances", []),
        }

# -------- 简易本地 JSONL 后端（默认） --------
else:
    _KB_DIR = Path("outputs") / "simple_kb"
    _KB_DIR.mkdir(parents=True, exist_ok=True)

    def _kb_file(name: str) -> Path:
        p = _KB_DIR / f"{name}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def add_texts(
        name: str,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ):
        """
        以 JSONL 方式持久化，每行一条：
        { "id": str, "text": str, "metadata": {...}, "embedding": [...] }
        """
        if not documents:
            return
        metadatas = metadatas or [{} for _ in documents]
        embeddings = embeddings or [[] for _ in documents]
        path = _kb_file(name)
        with path.open("a", encoding="utf-8") as f:
            for i, text in enumerate(documents):
                rec = {
                    "id": ids[i] if i < len(ids) else f"auto-{i}",
                    "text": text,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "embedding": embeddings[i] if i < len(embeddings) else [],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _load_kb(name: str) -> Tuple[List[str], List[str], List[Dict[str, Any]], np.ndarray]:
        """
        返回 (ids, texts, metas, emb_matrix[np.float32])
        """
        path = _kb_file(name)
        ids: List[str] = []
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        embs: List[List[float]] = []
        if not path.exists():
            return ids, texts, metas, np.zeros((0, 0), dtype=np.float32)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ids.append(obj.get("id"))
                    texts.append(obj.get("text"))
                    metas.append(obj.get("metadata") or {})
                    emb = obj.get("embedding") or []
                    embs.append([float(x) for x in emb])
                except Exception:
                    continue
        emb_mat = np.array(embs, dtype=np.float32) if embs else np.zeros((0, 0), dtype=np.float32)
        return ids, texts, metas, emb_mat

    def _cosine_sim_matrix(q: np.ndarray, m: np.ndarray) -> np.ndarray:
        """
        q: (1, d) or (nq, d); m: (N, d)
        return: (nq, N) 相似度
        """
        if q.ndim == 1:
            q = q[None, :]
        if m.size == 0:
            return np.zeros((q.shape[0], 0), dtype=np.float32)
        q_norm = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
        m_norm = np.linalg.norm(m, axis=1, keepdims=True).T + 1e-12  # (1, N)
        sim = (q @ m.T) / (q_norm @ m_norm)
        return sim.astype(np.float32)

    def query_by_embeddings(
        name: str,
        query_embeddings: List[List[float]],
        n_results: int = 5,
    ) -> Dict[str, Any]:
        ids, texts, metas, emb_mat = _load_kb(name)
        if not len(query_embeddings):
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        q = np.array(query_embeddings, dtype=np.float32)
        sim = _cosine_sim_matrix(q, emb_mat)  # (nq, N)
        results = {"ids": [], "documents": [], "metadatas": [], "distances": []}
        for i in range(sim.shape[0]):
            sims = sim[i]
            topk_idx = np.argsort(-sims)[: min(n_results, sims.shape[0])]
            results["ids"].append([ids[j] for j in topk_idx])
            results["documents"].append([texts[j] for j in topk_idx])
            results["metadatas"].append([metas[j] for j in topk_idx])
            # 用 (1 - 相似度) 作为“距离”，越小越相似（与 chroma 对齐）
            results["distances"].append([float(1.0 - sims[j]) for j in topk_idx])
        return results