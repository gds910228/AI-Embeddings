from typing import List
import re


def _normalize(text: str) -> str:
    # 规范化空白符，保留换行用于段落切分
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _split_sentences(text: str) -> List[str]:
    # 按中英文标点与换行做句/段切分
    # 保留分隔符，便于重构上下文
    parts: List[str] = []
    buf: List[str] = []
    for ch in text:
        buf.append(ch)
        if ch in "。！？!?;\n":
            parts.append("".join(buf).strip())
            buf = []
    if buf:
        parts.append("".join(buf).strip())
    # 过滤空串
    return [p for p in parts if p]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    将文本切分为长度受控的片段，尽量按句子边界切分。
    - chunk_size: 目标片段最大长度（字符数）
    - overlap: 片段之间的重叠字符数（用于保持语义连续）
    """
    text = _normalize(text)
    if not text:
        return []

    sentences = _split_sentences(text)
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    for s in sentences:
        s_len = len(s)
        if cur_len + s_len <= chunk_size or not cur:
            cur.append(s)
            cur_len += s_len
        else:
            chunks.append("".join(cur).strip())
            # 构造重叠窗口
            if overlap > 0 and chunks[-1]:
                tail = chunks[-1][-overlap:]
            else:
                tail = ""
            cur = [tail, s] if tail else [s]
            cur_len = sum(len(x) for x in cur)

    if cur:
        chunks.append("".join(cur).strip())

    # 若仍有超长，做硬切
    final_chunks: List[str] = []
    for c in chunks:
        if len(c) <= chunk_size:
            final_chunks.append(c)
        else:
            for i in range(0, len(c), chunk_size - overlap if chunk_size > overlap else chunk_size):
                final_chunks.append(c[i : i + chunk_size])
    return [c for c in final_chunks if c]