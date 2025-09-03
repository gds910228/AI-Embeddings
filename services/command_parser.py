from __future__ import annotations
import re
from typing import Dict, Any, List


def _parse_kv_flags(text: str) -> Dict[str, str]:
    """
    解析形如 key=value 的参数（允许 kb=foo top=5 chunk_size=500 overlap=50）
    """
    flags: Dict[str, str] = {}
    for m in re.finditer(r"(\bkb|top|chunk_size|overlap|model)\s*=\s*([^\s\"']+)", text, flags=re.IGNORECASE):
        key = m.group(1).lower()
        val = m.group(2)
        flags[key] = val.strip()
    return flags


def _extract_quoted(text: str) -> List[str]:
    """
    提取引号中的内容（支持中文/英文引号）
    """
    items: List[str] = []
    for pattern in [r"'([^']+)'", r"\"([^\"]+)\""]:
        items.extend(re.findall(pattern, text))
    return [s.strip() for s in items if s.strip()]


def parse_command(command: str) -> Dict[str, Any]:
    """
    解析自然语言指令，支持：
    - 索引:
      例: 索引 docs/policies kb=kb_policies chunk_size=500 overlap=50
          索引 "https://example.com/page" kb=kb_web
          index ./notes kb=kb_notes
    - 搜索:
      例: 搜索 '差旅报销怎么走' kb=kb_policies top=5
          search "请假申请" top=3
    返回统一结构:
    {
      "action": "index"|"search",
      # index:
      "paths": [...], "kb": str, "chunk_size": int, "overlap": int, "model": str
      # search:
      "query": str, "kb": str, "top": int, "model": str
    }
    """
    text = command.strip()

    # 统一英文关键词
    lower = text.lower()
    is_index = lower.startswith("索引") or lower.startswith("index")
    is_search = lower.startswith("搜索") or lower.startswith("search")

    flags = _parse_kv_flags(text)

    # 通用参数
    kb = flags.get("kb", "kb_default")
    model = flags.get("model", "embedding-3")

    if is_index:
        # 路径解析：优先取引号内的，若无则取“索引”后的剩余，剔除 kv 段
        quoted = _extract_quoted(text)
        if quoted:
            paths = quoted
        else:
            # 索引 后的主体（去掉 flags）
            body = re.sub(r"^(\s*索引|\s*index)\s*", "", text, flags=re.IGNORECASE).strip()
            # 删去所有 k=v 片段
            body = re.sub(r"(\bkb|top|chunk_size|overlap|model)\s*=\s*[^\s\"']+", "", body, flags=re.IGNORECASE).strip()
            # 以空格或逗号分隔多路径
            if body:
                paths = [p.strip() for p in re.split(r"[,\s]+", body) if p.strip()]
            else:
                paths = []

        chunk_size = int(flags.get("chunk_size", "500")) if flags.get("chunk_size") else 500
        overlap = int(flags.get("overlap", "50")) if flags.get("overlap") else 50

        return {
            "action": "index",
            "paths": paths,
            "kb": kb,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "model": model,
        }

    if is_search:
        # 查询解析：优先引号内内容，否则去掉前缀后的剩余（再剔除 k=v）
        quoted = _extract_quoted(text)
        if quoted:
            query = quoted[0]
        else:
            body = re.sub(r"^(\s*搜索|\s*search)\s*", "", text, flags=re.IGNORECASE).strip()
            body = re.sub(r"(\bkb|top|chunk_size|overlap|model)\s*=\s*[^\s\"']+", "", body, flags=re.IGNORECASE).strip()
            query = body

        top = int(flags.get("top", "5")) if flags.get("top") else 5

        return {
            "action": "search",
            "query": query,
            "kb": kb,
            "top": top,
            "model": model,
        }

    # 未识别
    return {
      "action": "unknown",
      "error": "无法识别的指令。示例：'索引 docs kb=kb_docs' 或 '搜索 \"请假申请\" kb=kb_docs top=5'"
    }