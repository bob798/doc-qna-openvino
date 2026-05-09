#!/usr/bin/env python3
"""
Benchmark 1：PyTorch vs OpenVINO 推理速度

输入：data/test_images/ 下的 10 张图片
输出：results/phase1/benchmark_inference.md  (markdown 表格)
      results/phase1/benchmark_inference.json (原始数据)

运行：
    # 同时跑两端（默认）
    python scripts/benchmark_inference.py

    # 只跑 OpenVINO
    python scripts/benchmark_inference.py --no_pytorch

    # 自定义路径
    python scripts/benchmark_inference.py \
        --image_dir data/test_images \
        --ir_dir ./models/paddleocr_vl_ov \
        --output_dir results/phase1 \
        --warmup 1 --runs 3
"""

import argparse
import json
import logging
import platform
import statistics
import sys
import time
from pathlib import Path

# 让 src/ 可被导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import (  # noqa: E402
    OpenVINOPaddleOCRVL,
    PyTorchPaddleOCRVL,
    iter_images,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def env_info() -> dict:
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
    }
    try:
        import openvino as ov

        info["openvino"] = ov.__version__
    except Exception:
        info["openvino"] = "n/a"
    try:
        import torch

        info["torch"] = torch.__version__
    except Exception:
        info["torch"] = "n/a"
    return info


def run_engine(engine, image_path: Path, warmup: int, runs: int) -> dict:
    """对单张图执行 warmup + runs 次，返回耗时统计"""
    for _ in range(warmup):
        engine.infer(image_path)

    elapsed = []
    last_text = ""
    for _ in range(runs):
        res = engine.infer(image_path)
        elapsed.append(res.elapsed_ms)
        last_text = res.text

    return {
        "mean_ms": statistics.mean(elapsed),
        "median_ms": statistics.median(elapsed),
        "min_ms": min(elapsed),
        "max_ms": max(elapsed),
        "raw_ms": elapsed,
        "text_len": len(last_text),
    }


def write_markdown(out_path: Path, env: dict, rows: list, totals: dict, runs: int):
    lines = []
    lines.append("# Benchmark 1 · PyTorch vs OpenVINO 推理速度\n")
    lines.append(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 每张图运行 {runs} 次取均值，前置 warmup 1 次\n")
    lines.append("## 环境信息\n")
    for k, v in env.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## 单张图片耗时（ms · mean ± min/max over runs）\n")
    has_torch = any("pytorch" in r for r in rows)
    if has_torch:
        lines.append("| 图片 | 类型 | PyTorch (ms) | OpenVINO (ms) | 加速比 |")
        lines.append("|------|------|--------------|---------------|--------|")
        for r in rows:
            pt = r.get("pytorch", {})
            ov_ = r.get("openvino", {})
            speedup = (
                f"{pt.get('mean_ms', 0) / ov_['mean_ms']:.2f}×"
                if pt.get("mean_ms") and ov_.get("mean_ms")
                else "—"
            )
            pt_str = (
                f"{pt['mean_ms']:.1f} (min {pt['min_ms']:.1f} / max {pt['max_ms']:.1f})"
                if pt
                else "—"
            )
            ov_str = (
                f"{ov_['mean_ms']:.1f} (min {ov_['min_ms']:.1f} / max {ov_['max_ms']:.1f})"
                if ov_
                else "—"
            )
            lines.append(f"| {r['name']} | {r['category']} | {pt_str} | {ov_str} | {speedup} |")
    else:
        lines.append("| 图片 | 类型 | OpenVINO (ms) |")
        lines.append("|------|------|---------------|")
        for r in rows:
            ov_ = r.get("openvino", {})
            ov_str = (
                f"{ov_['mean_ms']:.1f} (min {ov_['min_ms']:.1f} / max {ov_['max_ms']:.1f})"
                if ov_
                else "—"
            )
            lines.append(f"| {r['name']} | {r['category']} | {ov_str} |")
    lines.append("")

    lines.append("## 全部图片汇总\n")
    if "pytorch" in totals:
        lines.append(f"- PyTorch 平均: **{totals['pytorch']:.1f} ms**")
    if "openvino" in totals:
        lines.append(f"- OpenVINO 平均: **{totals['openvino']:.1f} ms**")
    if "speedup" in totals:
        lines.append(f"- 加速比: **{totals['speedup']:.2f}×**")
    lines.append("")
    lines.append("> 该数据用于替换 `docs/进阶方案.md` 中的 ≥ 2× 预估值。")
    out_path.write_text("\n".join(lines), encoding="utf-8")


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


def main():
    ap = argparse.ArgumentParser(description="PyTorch vs OpenVINO 推理速度 Benchmark")
    ap.add_argument("--image_dir", default="data/test_images")
    ap.add_argument("--ir_dir", default="./models/paddleocr_vl_ov", help="OpenVINO IR 目录")
    ap.add_argument("--hf_repo", default="PaddlePaddle/PaddleOCR-VL")
    ap.add_argument("--device", default="CPU", help="OpenVINO device: CPU/GPU/NPU/AUTO")
    ap.add_argument("--torch_device", default="cpu")
    ap.add_argument("--output_dir", default="results/phase1")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--no_pytorch", action="store_true", help="跳过 PyTorch 推理")
    ap.add_argument("--no_openvino", action="store_true", help="跳过 OpenVINO 推理")
    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = iter_images(image_dir)
    if not images:
        logger.error(f"目录中无图片：{image_dir}（请按 README 准备 10 张测试图片）")
        sys.exit(1)
    logger.info(f"将测试 {len(images)} 张图片")

    ov_engine = None
    pt_engine = None
    if not args.no_openvino:
        ov_engine = OpenVINOPaddleOCRVL(args.ir_dir, device=args.device)
    if not args.no_pytorch:
        try:
            pt_engine = PyTorchPaddleOCRVL(args.hf_repo, device=args.torch_device)
        except Exception as e:
            logger.warning(f"PyTorch 初始化失败（继续仅跑 OpenVINO）: {e}")

    rows = []
    pt_totals, ov_totals = [], []
    for img in images:
        row = {"name": img.name, "category": categorize(img.stem)}
        logger.info(f"→ {img.name} [{row['category']}]")

        if pt_engine is not None:
            try:
                row["pytorch"] = run_engine(pt_engine, img, args.warmup, args.runs)
                pt_totals.append(row["pytorch"]["mean_ms"])
                logger.info(f"  PyTorch  mean={row['pytorch']['mean_ms']:.1f} ms")
            except Exception as e:
                logger.error(f"  PyTorch 推理失败: {e}")

        if ov_engine is not None:
            try:
                row["openvino"] = run_engine(ov_engine, img, args.warmup, args.runs)
                ov_totals.append(row["openvino"]["mean_ms"])
                logger.info(f"  OpenVINO mean={row['openvino']['mean_ms']:.1f} ms")
            except Exception as e:
                logger.error(f"  OpenVINO 推理失败: {e}")

        rows.append(row)

    totals = {}
    if pt_totals:
        totals["pytorch"] = statistics.mean(pt_totals)
    if ov_totals:
        totals["openvino"] = statistics.mean(ov_totals)
    if "pytorch" in totals and "openvino" in totals:
        totals["speedup"] = totals["pytorch"] / totals["openvino"]

    env = env_info()
    json_path = out_dir / "benchmark_inference.json"
    md_path = out_dir / "benchmark_inference.md"
    json_path.write_text(
        json.dumps({"env": env, "rows": rows, "totals": totals, "runs": args.runs}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(md_path, env, rows, totals, args.runs)
    logger.info(f"完成 → {md_path} / {json_path}")


if __name__ == "__main__":
    main()
