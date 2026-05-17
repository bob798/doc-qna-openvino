"""
Phase 3 · 把 Phase 2 输出的 chunks.jsonl 灌入 ChromaDB

用法：
    # 灌入 results/phase2 下所有 *.chunks.jsonl
    python scripts/build_index.py \
        --chunks_dir results/phase2 \
        --persist_dir chroma_db \
        --device CPU --reset

    # 仅灌入指定文件
    python scripts/build_index.py \
        --chunks_files results/phase2/spec_with_tables.chunks.jsonl \
        --persist_dir chroma_db
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# 允许从 openvino/ 直接运行：把当前包根目录加进 sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.embedding import OpenVINOEmbedder, EmbedTiming  # noqa: E402
from src.vector_store import ChromaStore, iter_chunks_jsonl  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_index")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks_dir", type=str, default="results/phase2",
                   help="包含 *.chunks.jsonl 的目录")
    p.add_argument("--chunks_files", type=str, nargs="*", default=None,
                   help="指定具体 chunks.jsonl 文件（与 --chunks_dir 二选一）")
    p.add_argument("--persist_dir", type=str, default="chroma_db",
                   help="ChromaDB 持久化目录")
    p.add_argument("--collection", type=str, default="doc_chunks")
    p.add_argument("--device", type=str, default="CPU",
                   help="Embedder OpenVINO 设备：CPU / GPU / AUTO")
    p.add_argument("--model_id", type=str,
                   default="OpenVINO/Qwen3-Embedding-0.6B-int8-ov")
    p.add_argument("--local_dir", type=str, default=None,
                   help="若指定，直接用本地 IR 目录，不走 HF 下载")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--min_chars", type=int, default=0,
                   help="丢掉文本短于此阈值的 chunk（消除页眉页脚噪声，常用 60）")
    p.add_argument("--reset", action="store_true", help="灌入前清空集合")
    return p.parse_args()


def main():
    args = parse_args()

    # 1) 收集 chunks 文件
    files = []
    if args.chunks_files:
        files = [Path(p) for p in args.chunks_files]
    else:
        d = Path(args.chunks_dir)
        if not d.exists():
            logger.error(f"chunks 目录不存在: {d}")
            sys.exit(2)
        files = sorted(d.glob("*.chunks.jsonl"))
    if not files:
        logger.error("未找到任何 *.chunks.jsonl")
        sys.exit(2)
    logger.info(f"待灌入文件: {len(files)} 个")
    for f in files:
        logger.info(f"  - {f}")

    chunks = iter_chunks_jsonl(files, min_chars=args.min_chars)
    if args.min_chars > 0:
        logger.info(f"chunks 总数: {len(chunks)}（已过滤短于 {args.min_chars} 字符的噪声 chunk）")
    else:
        logger.info(f"chunks 总数: {len(chunks)}")
    if not chunks:
        logger.error("chunks 为空，退出")
        sys.exit(2)

    # 2) Embedder
    embedder = OpenVINOEmbedder(
        model_id=args.model_id,
        device=args.device,
        local_dir=args.local_dir,
        max_length=args.max_length,
    )

    # 3) Store
    store = ChromaStore(persist_dir=args.persist_dir, collection_name=args.collection)
    if args.reset:
        store.reset_collection()
        logger.info("集合已清空")

    # 4) 编码 + 入库
    texts = [c["text"] for c in chunks]
    timing = EmbedTiming()
    t0 = time.perf_counter()
    embs = embedder.encode(texts, batch_size=args.batch_size, timing=timing)
    encode_total = (time.perf_counter() - t0) * 1000
    logger.info(
        f"encode 完成: N={len(texts)}, dim={embs.shape[1]}, "
        f"tokenize={timing.tokenize_ms:.0f}ms, infer={timing.infer_ms:.0f}ms, "
        f"total={encode_total:.0f}ms ({encode_total/len(texts):.1f} ms/chunk)"
    )

    store.add_chunks(chunks, embs)
    logger.info(f"集合最终条数: {store.count()}")

    # 5) 输出 summary
    out_path = Path(args.persist_dir).parent / "build_index.summary.json"
    summary = {
        "embedder": {
            "model_id": args.model_id,
            "device": args.device,
            "dim": int(embs.shape[1]),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        },
        "chunks": {
            "files": [str(f) for f in files],
            "total": len(chunks),
        },
        "timing_ms": {
            "tokenize": round(timing.tokenize_ms, 1),
            "infer": round(timing.infer_ms, 1),
            "encode_total": round(encode_total, 1),
            "per_chunk": round(encode_total / max(1, len(chunks)), 2),
        },
        "store": {
            "persist_dir": args.persist_dir,
            "collection": args.collection,
            "count": store.count(),
        },
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"summary → {out_path}")


if __name__ == "__main__":
    main()
