"""
Phase 3 · 端到端问答 Demo

支持三种问题输入：
  1. --question "问题文本"           单条
  2. --questions_file q.txt           每行一个问题
  3. --eval_jsonl data/eval_questions.jsonl --question_ids q003,q008,q011
                                       从评测集挑题

输出：
  - 控制台打印每条问答 + 命中段 + 耗时分解
  - --out results/phase3/qa_run.json   汇总 JSON
  - --out_md results/phase3/qa_run.md  Markdown 报告

如果 --auto_build_index 且持久库为空，会自动从 --chunks_dir 灌入。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows cmd 默认 cp1252 编码会让中文 print 直接崩溃，强制 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.embedding import OpenVINOEmbedder  # noqa: E402
from src.llm import QwenLLM  # noqa: E402
from src.rag import RAGPipeline  # noqa: E402
from src.vector_store import ChromaStore, iter_chunks_jsonl  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_qa")


def parse_args():
    p = argparse.ArgumentParser()
    # 问题来源
    p.add_argument("--question", type=str, default=None)
    p.add_argument("--questions_file", type=str, default=None)
    p.add_argument("--eval_jsonl", type=str, default=None,
                   help="评测集 jsonl，每行 {id, question, ...}")
    p.add_argument("--question_ids", type=str, default=None,
                   help="逗号分隔的 question id 列表（配 --eval_jsonl 用）")
    p.add_argument("--limit", type=int, default=None,
                   help="评测集模式下取前 N 条")

    # 检索 / 库
    p.add_argument("--persist_dir", type=str, default="chroma_db")
    p.add_argument("--collection", type=str, default="doc_chunks")
    p.add_argument("--top_k", type=int, default=5)

    # 自动建库（若库为空）
    p.add_argument("--auto_build_index", action="store_true")
    p.add_argument("--chunks_dir", type=str, default="results/phase2")

    # 模型
    p.add_argument("--embed_model_id", type=str,
                   default="OpenVINO/Qwen3-Embedding-0.6B-int8-ov")
    p.add_argument("--embed_device", type=str, default="CPU")
    p.add_argument("--embed_local_dir", type=str, default=None)

    p.add_argument("--llm_model_id", type=str,
                   default="OpenVINO/Qwen3-1.7B-int4-ov")
    p.add_argument("--llm_device", type=str, default="CPU")
    p.add_argument("--llm_local_dir", type=str, default=None)
    p.add_argument("--max_new_tokens", type=int, default=384)

    # 输出
    p.add_argument("--out", type=str, default="results/phase3/qa_run.json")
    p.add_argument("--out_md", type=str, default="results/phase3/qa_run.md")
    return p.parse_args()


def collect_questions(args) -> List[dict]:
    """返回 [{id, question}] 列表"""
    out: List[dict] = []

    if args.eval_jsonl:
        path = Path(args.eval_jsonl)
        if not path.exists():
            logger.error(f"eval_jsonl 不存在: {path}")
            sys.exit(2)
        ids_filter = set(s.strip() for s in args.question_ids.split(",")) if args.question_ids else None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if ids_filter and obj.get("id") not in ids_filter:
                    continue
                out.append({
                    "id": obj.get("id", f"q{len(out)+1:03d}"),
                    "question": obj["question"],
                    "expected_keywords": obj.get("expected_keywords"),
                    "must_cite_doc": obj.get("must_cite_doc"),
                    "type": obj.get("type"),
                })
                if args.limit and len(out) >= args.limit:
                    break

    if args.questions_file:
        with open(args.questions_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                out.append({"id": f"q{len(out)+1:03d}", "question": line})

    if args.question:
        out.append({"id": "q_cli", "question": args.question})

    return out


def maybe_build_index(args, store: ChromaStore):
    """库为空且开了 --auto_build_index 时，自动灌入"""
    if store.count() > 0:
        logger.info(f"chroma_db 已有 {store.count()} 条，跳过自动建库")
        return
    if not args.auto_build_index:
        logger.error(
            f"chroma_db 为空且未开 --auto_build_index；"
            f"请先运行 scripts/build_index.py 或加 --auto_build_index"
        )
        sys.exit(2)
    d = Path(args.chunks_dir)
    files = sorted(d.glob("*.chunks.jsonl"))
    if not files:
        logger.error(f"自动建库失败：{d} 下没有 *.chunks.jsonl")
        sys.exit(2)
    logger.info(f"自动建库：{len(files)} 份 chunks.jsonl → {args.persist_dir}")
    chunks = iter_chunks_jsonl(files)
    embedder_tmp = OpenVINOEmbedder(
        model_id=args.embed_model_id,
        device=args.embed_device,
        local_dir=args.embed_local_dir,
    )
    embs = embedder_tmp.encode([c["text"] for c in chunks], batch_size=8)
    store.add_chunks(chunks, embs)
    logger.info(f"自动建库完成：共 {store.count()} 条")


def fmt_ms(x):
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}"


def main():
    args = parse_args()
    questions = collect_questions(args)
    if not questions:
        logger.error("没有可跑的问题，请指定 --question / --questions_file / --eval_jsonl")
        sys.exit(2)
    logger.info(f"待跑问题：{len(questions)} 条")

    # 1) 准备 store（先建/连）
    store = ChromaStore(persist_dir=args.persist_dir, collection_name=args.collection)
    maybe_build_index(args, store)

    # 2) Embedder + LLM
    embedder = OpenVINOEmbedder(
        model_id=args.embed_model_id,
        device=args.embed_device,
        local_dir=args.embed_local_dir,
    )
    llm = QwenLLM(
        model_id=args.llm_model_id,
        device=args.llm_device,
        local_dir=args.llm_local_dir,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=False,
    )
    rag = RAGPipeline(
        embedder=embedder,
        store=store,
        llm=llm,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )

    # 3) 跑问答
    results = []
    t_run = time.perf_counter()
    for i, q in enumerate(questions, start=1):
        logger.info(f"[{i}/{len(questions)}] {q['id']}: {q['question']}")
        ans = rag.answer(q["question"])

        print("\n" + "=" * 78)
        print(f"Q{i} ({q['id']}): {q['question']}")
        print(f"A: {ans.answer}")
        print(
            f"⏱  embed={fmt_ms(ans.timing.embed_query_ms)}ms  "
            f"retrieve={fmt_ms(ans.timing.retrieve_ms)}ms  "
            f"llm={fmt_ms(ans.timing.llm_ms)}ms  "
            f"total={fmt_ms(ans.timing.total_ms)}ms  "
            f"new_tokens={ans.timing.new_tokens}  "
            f"tps={ans.timing.tokens_per_second:.1f}"
        )
        print("Top-K 命中:")
        for h in ans.hits:
            print(f"  - {h.cite()} score={h.score:.3f}  {h.text[:100].replace(chr(10),' ')}")

        results.append({**q, **ans.to_dict()})

    total_run = (time.perf_counter() - t_run) * 1000
    logger.info(f"全部完成，共 {len(results)} 题，{total_run:.0f} ms")

    # 4) 输出 JSON + Markdown
    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "config": {
            "embed_model_id": args.embed_model_id,
            "embed_device": args.embed_device,
            "llm_model_id": args.llm_model_id,
            "llm_device": args.llm_device,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "persist_dir": args.persist_dir,
            "collection": args.collection,
        },
        "summary": {
            "n_questions": len(results),
            "total_run_ms": round(total_run, 1),
            "avg_total_ms": round(sum(r["timing"]["total_ms"] for r in results) / max(1, len(results)), 1),
            "avg_embed_ms": round(sum(r["timing"]["embed_query_ms"] for r in results) / max(1, len(results)), 1),
            "avg_retrieve_ms": round(sum(r["timing"]["retrieve_ms"] for r in results) / max(1, len(results)), 1),
            "avg_llm_ms": round(sum(r["timing"]["llm_ms"] for r in results) / max(1, len(results)), 1),
            "avg_tps": round(sum(r["timing"]["tokens_per_second"] for r in results) / max(1, len(results)), 1),
        },
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"JSON → {out_json}")

    # Markdown 报告
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 3 · RAG 端到端问答结果",
        "",
        f"- 嵌入模型: `{args.embed_model_id}` on `{args.embed_device}`",
        f"- 大语言模型: `{args.llm_model_id}` on `{args.llm_device}`",
        f"- Top-K: `{args.top_k}` | max_new_tokens: `{args.max_new_tokens}`",
        f"- 向量库: `{args.persist_dir}/{args.collection}` (chunks={store.count()})",
        "",
        "## 性能汇总",
        "",
        "| 阶段 | 平均耗时 (ms) |",
        "|------|---------------|",
        f"| embed query | {sum(r['timing']['embed_query_ms'] for r in results)/max(1,len(results)):.1f} |",
        f"| retrieve | {sum(r['timing']['retrieve_ms'] for r in results)/max(1,len(results)):.1f} |",
        f"| LLM | {sum(r['timing']['llm_ms'] for r in results)/max(1,len(results)):.1f} |",
        f"| **total** | **{sum(r['timing']['total_ms'] for r in results)/max(1,len(results)):.1f}** |",
        f"| LLM tps | {sum(r['timing']['tokens_per_second'] for r in results)/max(1,len(results)):.1f} tok/s |",
        "",
        "## 详细问答",
        "",
    ]
    for i, r in enumerate(results, start=1):
        lines += [
            f"### Q{i} ({r.get('id')})",
            "",
            f"**问题**：{r['question']}",
            "",
            f"**回答**：",
            "",
            r["answer"].strip(),
            "",
            "**Top-K 命中**：",
            "",
        ]
        for c in r["citations"]:
            lines.append(
                f"- `[{c['doc_name']} p.{c['page']}]` score={c['score']:.3f} kind={c.get('kind')} — {c['preview']}"
            )
        t = r["timing"]
        lines += [
            "",
            f"⏱ embed={t['embed_query_ms']}ms  retrieve={t['retrieve_ms']}ms  "
            f"llm={t['llm_ms']}ms  total={t['total_ms']}ms  "
            f"new_tokens={t['new_tokens']}  tps={t['tokens_per_second']}",
            "",
            "---",
            "",
        ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Markdown → {out_md}")


if __name__ == "__main__":
    main()
