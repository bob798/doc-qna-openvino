"""
表格感知切片器

策略：
  1. 解析 ParsedDocument 的每页 Markdown
  2. 识别 Markdown 表格块（连续 `| ... |` 行 + 至少一行分隔符 `|---|---|`）
  3. 表格切片：表头 + 单行 → 一个 chunk（保留列名上下文）
  4. 非表格切片：按二级标题 / 段落分组，单个 chunk 控制在 [MIN_CHUNK, MAX_CHUNK] 字符范围
     超长段落做二次切分，保留 OVERLAP 字符重叠
  5. 元数据：{doc_name, page, section_title, kind: "text"|"table"|"table_header"}
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional

from .doc_parser import ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)

MIN_CHUNK = 200
MAX_CHUNK = 500
OVERLAP = 50

TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"text": self.text, "metadata": self.metadata}


# ── 表格块识别 ────────────────────────────────────────────────────────────────


@dataclass
class TableBlock:
    header: str
    separator: str
    rows: List[str]


def _detect_table_blocks(lines: List[str]) -> List[tuple[int, int, TableBlock]]:
    """
    返回 [(start_idx, end_idx_exclusive, TableBlock)]，索引基于 lines。
    """
    out = []
    i = 0
    n = len(lines)
    while i < n - 1:
        if TABLE_LINE_RE.match(lines[i]) and TABLE_SEP_RE.match(lines[i + 1] or ""):
            header = lines[i]
            sep = lines[i + 1]
            j = i + 2
            rows = []
            while j < n and TABLE_LINE_RE.match(lines[j]):
                rows.append(lines[j])
                j += 1
            if rows:
                out.append((i, j, TableBlock(header=header, separator=sep, rows=rows)))
                i = j
                continue
        i += 1
    return out


# ── 切片：表格 ────────────────────────────────────────────────────────────────


def _table_to_chunks(
    block: TableBlock, base_meta: dict
) -> List[Chunk]:
    """每行 + 表头 → 一个 chunk；同时附加一个表头总览 chunk 便于召回"""
    chunks: List[Chunk] = []

    overview = "\n".join([block.header, block.separator])
    chunks.append(
        Chunk(
            text=overview,
            metadata={**base_meta, "kind": "table_header", "row_count": len(block.rows)},
        )
    )
    for idx, row in enumerate(block.rows, start=1):
        text = "\n".join([block.header, block.separator, row])
        chunks.append(
            Chunk(
                text=text,
                metadata={**base_meta, "kind": "table", "row_index": idx},
            )
        )
    return chunks


# ── 切片：文字段 ──────────────────────────────────────────────────────────────


def _split_long_paragraph(text: str, max_chars: int = MAX_CHUNK, overlap: int = OVERLAP) -> List[str]:
    """超长段落按字符滑窗切分，保留 overlap 字符重叠"""
    if len(text) <= max_chars:
        return [text]
    pieces = []
    step = max_chars - overlap
    if step <= 0:
        step = max_chars
    i = 0
    while i < len(text):
        pieces.append(text[i : i + max_chars])
        if i + max_chars >= len(text):
            break
        i += step
    return pieces


def _flush_buffer(buf: List[str], base_meta: dict) -> List[Chunk]:
    """把累积的段落 buffer 输出成 chunks（合并到 ≥ MIN_CHUNK 后再切）"""
    if not buf:
        return []
    text = "\n\n".join(s for s in buf if s.strip())
    if not text.strip():
        return []
    chunks: List[Chunk] = []
    if len(text) <= MAX_CHUNK:
        chunks.append(Chunk(text=text, metadata={**base_meta, "kind": "text"}))
        return chunks
    for piece in _split_long_paragraph(text):
        chunks.append(Chunk(text=piece, metadata={**base_meta, "kind": "text"}))
    return chunks


# ── 主流程 ────────────────────────────────────────────────────────────────────


def chunk_page(page: ParsedPage, doc_name: str) -> List[Chunk]:
    md = page.markdown or ""
    if not md.strip():
        return []

    lines = md.split("\n")
    table_spans = _detect_table_blocks(lines)
    table_idx_set = set()
    for s, e, _ in table_spans:
        for k in range(s, e):
            table_idx_set.add(k)

    chunks: List[Chunk] = []
    section_title: Optional[str] = None
    buf: List[str] = []
    cur_paragraph: List[str] = []

    base_meta = {
        "doc_name": doc_name,
        "page": page.page_no,
    }

    def base_with_section() -> dict:
        m = dict(base_meta)
        if section_title:
            m["section_title"] = section_title
        return m

    i = 0
    n = len(lines)
    table_iter = iter(table_spans)
    next_table = next(table_iter, None)

    while i < n:
        # 命中表格起点 → flush 当前 buffer，输出表格 chunk
        if next_table and i == next_table[0]:
            if cur_paragraph:
                buf.append(" ".join(cur_paragraph).strip())
                cur_paragraph = []
            chunks.extend(_flush_buffer(buf, base_with_section()))
            buf = []

            _, end, block = next_table
            chunks.extend(_table_to_chunks(block, base_with_section()))
            i = end
            next_table = next(table_iter, None)
            continue

        line = lines[i]
        stripped = line.strip()

        m = HEADING_RE.match(stripped)
        if m:
            # 标题：先 flush 之前的内容，再切换 section_title
            if cur_paragraph:
                buf.append(" ".join(cur_paragraph).strip())
                cur_paragraph = []
            chunks.extend(_flush_buffer(buf, base_with_section()))
            buf = []
            section_title = m.group(2).strip()
            # 标题本身也作为 chunk 的一部分（拼到下一个段落）
            buf.append(stripped)
            i += 1
            continue

        if stripped == "":
            if cur_paragraph:
                buf.append(" ".join(cur_paragraph).strip())
                cur_paragraph = []
                # 检查 buffer 是否已经够大
                cur_text_len = sum(len(s) for s in buf)
                if cur_text_len >= MIN_CHUNK:
                    chunks.extend(_flush_buffer(buf, base_with_section()))
                    buf = []
            i += 1
            continue

        cur_paragraph.append(stripped)
        i += 1

    if cur_paragraph:
        buf.append(" ".join(cur_paragraph).strip())
    chunks.extend(_flush_buffer(buf, base_with_section()))

    return chunks


def chunk_document(doc: ParsedDocument) -> List[Chunk]:
    out: List[Chunk] = []
    for page in doc.pages:
        out.extend(chunk_page(page, doc_name=doc.doc_name))
    return out


# ── 序列化 ────────────────────────────────────────────────────────────────────


def chunks_to_jsonl(chunks: Iterable[Chunk]) -> str:
    import json

    return "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in chunks)
