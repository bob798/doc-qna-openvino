#!/usr/bin/env python3
"""
Phase 2 端到端管线 CLI

输入：单个 PDF 或 PDF 目录
输出：
    <out>/<name>.markdown        全文档 Markdown
    <out>/<name>.chunks.jsonl    切片 jsonl（带元数据）
    <out>/<name>.summary.json    解析时长 / 切片统计

示例：
    python scripts/run_phase2_pipeline.py --pdf data/test_documents/spec_with_tables.pdf
    python scripts/run_phase2_pipeline.py --pdf_dir data/test_documents --out results/phase2
    python scripts/run_phase2_pipeline.py --pdf foo.pdf --force_ocr  # 用于 Tesseract 对照实验
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import chunks_to_jsonl  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def write_outputs(out_dir: Path, name: str, result):
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{name}.markdown"
    jsonl_path = out_dir / f"{name}.chunks.jsonl"
    summary_path = out_dir / f"{name}.summary.json"

    md_path.write_text(result.document.to_markdown(), encoding="utf-8")
    jsonl_path.write_text(chunks_to_jsonl(result.chunks), encoding="utf-8")

    summary = {
        "doc_name": result.document.doc_name,
        "timings": result.document.timings,
        "chunk_stats": {
            "total": len(result.chunks),
            "by_kind": _counter([c.metadata.get("kind", "?") for c in result.chunks]),
            "avg_chars": (
                sum(len(c.text) for c in result.chunks) / max(len(result.chunks), 1)
            ),
        },
        "outputs": {
            "markdown": md_path.name,
            "chunks_jsonl": jsonl_path.name,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        f"  写入: {md_path.name} / {jsonl_path.name} / {summary_path.name} "
        f"(chunks={summary['chunk_stats']['total']})"
    )
    return summary


def _counter(items):
    out = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser(description="Phase 2 端到端管线 CLI")
    ap.add_argument("--pdf", help="单个 PDF 路径")
    ap.add_argument("--pdf_dir", help="PDF 目录（批量模式）")
    ap.add_argument("--out", default="results/phase2", help="输出目录")
    ap.add_argument("--ir_dir", default="./models/paddleocr_vl_ov")
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--force_ocr", action="store_true", help="跳过 text-layer，全部走 OCR")
    args = ap.parse_args()

    if not args.pdf and not args.pdf_dir:
        ap.error("必须指定 --pdf 或 --pdf_dir")

    out_dir = Path(args.out)
    pdfs = []
    if args.pdf:
        pdfs.append(Path(args.pdf))
    if args.pdf_dir:
        pdfs.extend(sorted(Path(args.pdf_dir).glob("*.pdf")))

    if not pdfs:
        logger.error("未找到 PDF 文件")
        sys.exit(1)

    summaries = []
    engine = None  # lazy 初始化（多 PDF 时复用）
    for pdf in pdfs:
        result = run_pipeline(
            pdf,
            ir_dir=args.ir_dir,
            device=args.device,
            dpi=args.dpi,
            force_ocr=args.force_ocr,
            engine=engine,
        )
        # 复用 engine
        if engine is None:
            engine = getattr(result, "_engine", None)
        s = write_outputs(out_dir, pdf.stem, result)
        summaries.append(s)

    # 全局汇总
    if len(summaries) > 1:
        overall = out_dir / "phase2_overview.json"
        overall.write_text(
            json.dumps({"docs": summaries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"全局汇总: {overall}")


if __name__ == "__main__":
    main()
