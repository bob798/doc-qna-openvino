#!/usr/bin/env python3
"""
单页文档解析性能及耗时测试

测试维度：
- 各阶段分步耗时（布局检测 / 视觉编码 / Prefill / Decode）
- 端到端总耗时
- 吞吐量（pages/sec）
- 不同文档类型的性能差异

使用方式：
    python benchmark_performance.py --image_dir ./data/test_images --model_dir ./models/qnn/context_binaries
    python benchmark_performance.py --image single_page.png --warmup 3 --repeat 10
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── 性能测试 ──────────────────────────────────────────────────────────────────

class PerformanceBenchmark:
    """性能基准测试"""

    def __init__(self, model_dir: str, backend: str = "htp"):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from doc_parser import DocParser
        self.parser = DocParser(model_dir=model_dir, backend=backend)

    def benchmark_single(
        self,
        image_path: str,
        warmup: int = 3,
        repeat: int = 10,
    ) -> Dict:
        """对单张图片进行性能测试"""
        logger.info(f"测试图片: {image_path}")
        logger.info(f"  预热: {warmup} 次, 正式: {repeat} 次")

        # 预热
        for _ in range(warmup):
            self.parser.parse(image_path)

        # 正式测试
        timings = []
        for i in range(repeat):
            result = self.parser.parse(image_path)
            timings.append(result.timing)

        # 统计
        stats = self._compute_stats(timings)
        stats["image"] = os.path.basename(image_path)
        stats["warmup"] = warmup
        stats["repeat"] = repeat

        return stats

    def benchmark_batch(
        self,
        image_dir: str,
        warmup: int = 2,
        repeat: int = 5,
    ) -> Dict:
        """批量性能测试"""
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        images = sorted(
            str(f) for f in Path(image_dir).iterdir()
            if f.suffix.lower() in image_extensions
        )

        if not images:
            logger.error(f"未找到测试图片: {image_dir}")
            return {}

        logger.info(f"批量测试: {len(images)} 张图片")

        results = []
        for img in images:
            stats = self.benchmark_single(img, warmup, repeat)
            results.append(stats)

        # 汇总
        summary = self._compute_batch_summary(results)
        return {
            "per_image": results,
            "summary": summary,
        }

    def _compute_stats(self, timings: List[Dict]) -> Dict:
        """计算时间统计"""
        stages = {}
        for timing in timings:
            for stage, t in timing.items():
                if stage not in stages:
                    stages[stage] = []
                stages[stage].append(t * 1000)  # 转为毫秒

        stats = {}
        for stage, values in stages.items():
            arr = np.array(values)
            stats[stage] = {
                "mean_ms": round(float(np.mean(arr)), 1),
                "std_ms": round(float(np.std(arr)), 1),
                "min_ms": round(float(np.min(arr)), 1),
                "max_ms": round(float(np.max(arr)), 1),
                "p50_ms": round(float(np.percentile(arr, 50)), 1),
                "p95_ms": round(float(np.percentile(arr, 95)), 1),
            }

        return stats

    def _compute_batch_summary(self, results: List[Dict]) -> Dict:
        """计算批量测试汇总"""
        total_times = []
        for r in results:
            if "total" in r:
                total_times.append(r["total"]["mean_ms"])

        if not total_times:
            return {}

        arr = np.array(total_times)
        return {
            "num_images": len(results),
            "total_latency": {
                "mean_ms": round(float(np.mean(arr)), 1),
                "std_ms": round(float(np.std(arr)), 1),
                "min_ms": round(float(np.min(arr)), 1),
                "max_ms": round(float(np.max(arr)), 1),
            },
            "throughput_pages_per_sec": round(1000.0 / float(np.mean(arr)), 3),
        }


# ── 报告生成 ──────────────────────────────────────────────────────────────────

def generate_performance_report(results: Dict, output_path: str):
    """生成性能测试报告"""
    summary = results.get("summary", {})
    per_image = results.get("per_image", [])

    report = """# 单页文档解析性能测试报告

## 测试概览

| 指标 | 值 |
|------|-----|
| 测试图片数 | {num_images} |
| 平均端到端耗时 | {mean_ms:.1f} ms |
| 最小耗时 | {min_ms:.1f} ms |
| 最大耗时 | {max_ms:.1f} ms |
| 吞吐量 | {throughput:.3f} pages/sec |

## 各阶段耗时分析

| 阶段 | 平均耗时 (ms) | 标准差 (ms) | P95 (ms) | 占比 |
|------|:------------:|:-----------:|:--------:|:----:|
""".format(
        num_images=summary.get("num_images", 0),
        mean_ms=summary.get("total_latency", {}).get("mean_ms", 0),
        min_ms=summary.get("total_latency", {}).get("min_ms", 0),
        max_ms=summary.get("total_latency", {}).get("max_ms", 0),
        throughput=summary.get("throughput_pages_per_sec", 0),
    )

    # 阶段耗时（取第一张图的数据作为典型代表）
    if per_image:
        typical = per_image[0]
        total_mean = typical.get("total", {}).get("mean_ms", 1)
        stage_order = ["layout_detection", "vl_recognition", "total"]

        for stage in stage_order:
            if stage in typical and isinstance(typical[stage], dict):
                s = typical[stage]
                pct = s["mean_ms"] / total_mean * 100 if stage != "total" else 100
                report += "| {stage} | {mean:.1f} | {std:.1f} | {p95:.1f} | {pct:.1f}% |\n".format(
                    stage=stage,
                    mean=s["mean_ms"],
                    std=s["std_ms"],
                    p95=s.get("p95_ms", s["mean_ms"]),
                    pct=pct,
                )

    report += """
## 逐图耗时

| 图片 | 布局检测 (ms) | VL 识别 (ms) | 总耗时 (ms) |
|------|:------------:|:-----------:|:-----------:|
"""

    for r in per_image:
        layout = r.get("layout_detection", {}).get("mean_ms", "-")
        vl = r.get("vl_recognition", {}).get("mean_ms", "-")
        total = r.get("total", {}).get("mean_ms", "-")
        report += f"| {r.get('image', '?')} | {layout} | {vl} | {total} |\n"

    report += """
## 测试环境

| 项目 | 值 |
|------|-----|
| 推理后端 | QNN HTP (Hexagon NPU) |
| 量化策略 | 布局检测: INT8, VL 模型: FP16 |
| SDK 版本 | QNN SDK (待填) |
| 设备 / Simulator | (待填) |
| 预热次数 | {warmup} |
| 测试次数 | {repeat} |

## 性能优化建议

1. **布局检测**：INT8 量化后在 HTP 上通常可达 10-50ms，若超出可尝试降低输入分辨率
2. **视觉编码**：SigLip-400M 在 HTP FP16 约 100-300ms，是主要瓶颈之一
3. **文本解码**：3B 模型的自回归生成是最大耗时项，可通过：
   - 减少 max_new_tokens
   - 使用 speculative decoding
   - 尝试 INT8 量化（需验证精度）
4. **整体优化**：对简单文档可跳过布局检测，直接全图 VL 识别

---

*报告由自动化性能测试脚本生成*
""".format(
        warmup=per_image[0].get("warmup", 3) if per_image else 3,
        repeat=per_image[0].get("repeat", 10) if per_image else 10,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"性能报告已生成: {output_path}")
    return report


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="文档解析性能测试")
    parser.add_argument("--image", type=str, help="单张测试图片")
    parser.add_argument("--image_dir", type=str, help="批量测试图片目录")
    parser.add_argument("--model_dir", type=str, default="./models/qnn/context_binaries",
                        help="QNN 模型目录")
    parser.add_argument("--backend", choices=["htp", "htp_simulator", "cpu"], default="htp")
    parser.add_argument("--warmup", type=int, default=3, help="预热次数")
    parser.add_argument("--repeat", type=int, default=10, help="正式测试次数")
    parser.add_argument("--output", type=str, default="./reports/性能测试报告.md",
                        help="报告输出路径")

    args = parser.parse_args()

    bench = PerformanceBenchmark(args.model_dir, args.backend)

    if args.image:
        results = {
            "per_image": [bench.benchmark_single(args.image, args.warmup, args.repeat)],
            "summary": {},
        }
    elif args.image_dir:
        results = bench.benchmark_batch(args.image_dir, args.warmup, args.repeat)
    else:
        parser.print_help()
        return

    # 保存 JSON
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json_path = args.output.replace(".md", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 生成报告
    report = generate_performance_report(results, args.output)
    print(report)


if __name__ == "__main__":
    main()
