#!/usr/bin/env python3
"""
Benchmark 2：Tesseract vs PaddleOCR-VL 解析质量对比

输入：data/test_images/ 下 10 张图片
输出：results/phase1/quality_compare.md   （并列展示，便于人工评估）
      results/phase1/quality_compare/<name>/  （单图对比详情，含原图复制）
      results/phase1/quality_compare.json    （结构化数据）

逐张人工打分填入 results/phase1/quality_compare.md 末尾的评分表即可。

运行：
    python scripts/benchmark_ocr_quality.py
    python scripts/benchmark_ocr_quality.py --image_dir data/test_images --ir_dir ./models/paddleocr_vl_ov
"""

import argparse
import difflib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import OpenVINOPaddleOCRVL, TesseractEngine, iter_images  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def categorize(name: str) -> str:
    n = name.lower()
    if n.startswith("text"):
        return "纯文字"
    if n.startswith("table"):
        return "表格"
    if n.startswith("formula"):
        return "公式"
    if n.startswith("mix"):
        return "图文混排"
    return "其他"


def char_similarity(a: str, b: str) -> float:
    """字符级 SequenceMatcher 相似度，仅作粗略指标"""
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def write_summary(out_md: Path, items: list, env_note: str):
    lines = []
    lines.append("# Benchmark 2 · Tesseract vs PaddleOCR-VL 解析质量对比\n")
    lines.append(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> {env_note}\n")
    lines.append("## 总览\n")
    lines.append("| 图片 | 类型 | Tesseract 字数 | PaddleOCR-VL 字数 | 字符相似度 | Tesseract 耗时 | PaddleOCR-VL 耗时 |")
    lines.append("|------|------|----------------|-------------------|------------|----------------|-------------------|")
    for it in items:
        t = it["tesseract"]
        p = it["paddleocr_vl"]
        lines.append(
            f"| {it['name']} | {it['category']} | {t['len']} | {p['len']} | "
            f"{it['similarity']:.2%} | {t['ms']:.0f} ms | {p['ms']:.0f} ms |"
        )
    lines.append("")

    lines.append("## 单图详情\n")
    for it in items:
        lines.append(f"### {it['name']} · {it['category']}\n")
        lines.append("**Tesseract 输出**\n")
        lines.append("```text")
        lines.append(it["tesseract"]["text"].strip() or "（空）")
        lines.append("```\n")
        lines.append("**PaddleOCR-VL 输出**\n")
        lines.append("```markdown")
        lines.append(it["paddleocr_vl"]["text"].strip() or "（空）")
        lines.append("```\n")
        lines.append(f"详情目录：[`{it['detail_dir']}`]({it['detail_dir']})\n")

    lines.append("## 人工评分（请逐项填入）\n")
    lines.append("评分维度：1=很差，5=很好；表格结构与公式列对纯文字图填 N/A。\n")
    lines.append("| 图片 | 文字识别 (T) | 文字识别 (P) | 表格结构 (T) | 表格结构 (P) | 公式 (T) | 公式 (P) | 备注 |")
    lines.append("|------|--------------|--------------|--------------|--------------|----------|----------|------|")
    for it in items:
        lines.append(f"| {it['name']} |  |  |  |  |  |  |  |")
    lines.append("")
    lines.append("> 评分填好后将本表截图存入 `assets/`，并在 Notebook 中引用作为对比演示。")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Tesseract vs PaddleOCR-VL 解析质量对比")
    ap.add_argument("--image_dir", default="data/test_images")
    ap.add_argument("--ir_dir", default="./models/paddleocr_vl_ov")
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--tesseract_lang", default="chi_sim+eng")
    ap.add_argument("--output_dir", default="results/phase1")
    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    detail_root = out_dir / "quality_compare"
    detail_root.mkdir(parents=True, exist_ok=True)

    images = iter_images(image_dir)
    if not images:
        logger.error(f"目录中无图片：{image_dir}")
        sys.exit(1)
    logger.info(f"将对比 {len(images)} 张图片")

    tess = TesseractEngine(lang=args.tesseract_lang)
    ov_engine = OpenVINOPaddleOCRVL(args.ir_dir, device=args.device)

    items = []
    for img in images:
        category = categorize(img.stem)
        logger.info(f"→ {img.name} [{category}]")
        detail_dir = detail_root / img.stem
        detail_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, detail_dir / img.name)

        try:
            t_res = tess.infer(img)
        except Exception as e:
            logger.error(f"  Tesseract 失败: {e}")
            t_res = type("R", (), dict(text=f"[ERROR] {e}", elapsed_ms=0.0))()

        try:
            p_res = ov_engine.infer(img)
        except Exception as e:
            logger.error(f"  PaddleOCR-VL 失败: {e}")
            p_res = type("R", (), dict(text=f"[ERROR] {e}", elapsed_ms=0.0))()

        sim = char_similarity(t_res.text, p_res.text)

        (detail_dir / "tesseract.txt").write_text(t_res.text, encoding="utf-8")
        (detail_dir / "paddleocr_vl.md").write_text(p_res.text, encoding="utf-8")
        diff = "\n".join(
            difflib.unified_diff(
                t_res.text.splitlines(),
                p_res.text.splitlines(),
                fromfile="tesseract",
                tofile="paddleocr_vl",
                lineterm="",
            )
        )
        (detail_dir / "diff.txt").write_text(diff, encoding="utf-8")

        items.append(
            {
                "name": img.name,
                "category": category,
                "detail_dir": str(detail_dir.relative_to(out_dir.parent)).replace("\\", "/"),
                "tesseract": {"text": t_res.text, "len": len(t_res.text), "ms": t_res.elapsed_ms},
                "paddleocr_vl": {"text": p_res.text, "len": len(p_res.text), "ms": p_res.elapsed_ms},
                "similarity": sim,
            }
        )

    summary_md = out_dir / "quality_compare.md"
    summary_json = out_dir / "quality_compare.json"
    summary_json.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(
        summary_md,
        items,
        env_note=f"Tesseract lang={args.tesseract_lang}; OpenVINO IR={args.ir_dir} on {args.device}",
    )
    logger.info(f"完成 → {summary_md}")


if __name__ == "__main__":
    main()
