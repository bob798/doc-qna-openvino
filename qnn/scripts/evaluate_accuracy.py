#!/usr/bin/env python3
"""
精度对比评测脚本

对比 QNN 端侧推理结果 vs 原始 PaddlePaddle 推理结果：
- 布局检测 mAP
- 文本识别准确率（编辑距离 / CER / WER）
- 表格结构保留率
- 端到端文档解析质量

使用方式：
    # 运行 PaddlePaddle 基线推理
    python evaluate_accuracy.py baseline --image_dir ./data/test_images --output_dir ./results/baseline

    # 运行 QNN 推理
    python evaluate_accuracy.py qnn --image_dir ./data/test_images --output_dir ./results/qnn

    # 对比评测
    python evaluate_accuracy.py compare --baseline_dir ./results/baseline --qnn_dir ./results/qnn

    # 生成报告
    python evaluate_accuracy.py report --results_dir ./results
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── 评测指标 ──────────────────────────────────────────────────────────────────

def edit_distance(s1: str, s2: str) -> int:
    """计算编辑距离 (Levenshtein distance)"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """字符错误率 (CER)"""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """词错误率 (WER)"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(" ".join(ref_words), " ".join(hyp_words)) / len(ref_words)


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """计算两个框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_layout_map(
    pred_boxes: List[List[float]],
    pred_labels: List[int],
    gt_boxes: List[List[float]],
    gt_labels: List[int],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """计算布局检测 mAP"""
    label_names = {0: "text", 1: "title", 2: "table", 3: "figure", 4: "formula"}
    results = {}

    for label_id, label_name in label_names.items():
        pred_idx = [i for i, l in enumerate(pred_labels) if l == label_id]
        gt_idx = [i for i, l in enumerate(gt_labels) if l == label_id]

        if not gt_idx:
            continue

        tp = 0
        matched_gt = set()
        for pi in pred_idx:
            best_iou = 0
            best_gi = -1
            for gi in gt_idx:
                if gi in matched_gt:
                    continue
                iou = compute_iou(pred_boxes[pi], gt_boxes[gi])
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= iou_threshold and best_gi >= 0:
                tp += 1
                matched_gt.add(best_gi)

        precision = tp / max(len(pred_idx), 1)
        recall = tp / len(gt_idx)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        results[label_name] = {"precision": precision, "recall": recall, "f1": f1}

    # 总体 mAP
    if results:
        results["mAP"] = np.mean([v["f1"] for v in results.values()])
    else:
        results["mAP"] = 0.0

    return results


def compute_table_accuracy(pred_table: str, gt_table: str) -> Dict[str, float]:
    """评估表格结构保留率"""
    # 解析 Markdown 表格行列
    def parse_table(md: str) -> List[List[str]]:
        rows = []
        for line in md.strip().split("\n"):
            if line.startswith("|") and not all(c in "|-: " for c in line):
                cells = [c.strip() for c in line.split("|")[1:-1]]
                rows.append(cells)
        return rows

    pred_rows = parse_table(pred_table)
    gt_rows = parse_table(gt_table)

    if not gt_rows:
        return {"row_accuracy": 1.0, "cell_accuracy": 1.0, "structure_score": 1.0}

    # 行数匹配度
    row_accuracy = 1.0 - abs(len(pred_rows) - len(gt_rows)) / max(len(gt_rows), 1)
    row_accuracy = max(0.0, row_accuracy)

    # 单元格内容匹配
    total_cells = 0
    correct_cells = 0
    for i in range(min(len(pred_rows), len(gt_rows))):
        for j in range(min(len(pred_rows[i]), len(gt_rows[i]))):
            total_cells += 1
            if pred_rows[i][j].strip() == gt_rows[i][j].strip():
                correct_cells += 1
        total_cells += abs(len(pred_rows[i]) - len(gt_rows[i]))

    cell_accuracy = correct_cells / max(total_cells, 1)

    # 结构分数（列数一致性）
    pred_cols = set(len(r) for r in pred_rows) if pred_rows else {0}
    gt_cols = set(len(r) for r in gt_rows) if gt_rows else {0}
    structure_score = 1.0 if pred_cols == gt_cols else 0.5

    return {
        "row_accuracy": round(row_accuracy, 4),
        "cell_accuracy": round(cell_accuracy, 4),
        "structure_score": round(structure_score, 4),
    }


# ── 基线推理 ──────────────────────────────────────────────────────────────────

def run_baseline(image_dir: str, output_dir: str):
    """使用原始 PaddlePaddle 推理作为基线"""
    os.makedirs(output_dir, exist_ok=True)

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        logger.error("请安装 PaddleOCR: pip install paddleocr")
        sys.exit(1)

    # 初始化 PaddleOCR（doc_parser 模式）
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = sorted(
        f for f in Path(image_dir).iterdir()
        if f.suffix.lower() in image_extensions
    )

    logger.info(f"PaddlePaddle 基线推理: {len(images)} 张图片")

    for img_path in images:
        logger.info(f"  处理: {img_path.name}")

        # 布局检测 + OCR
        result = ocr.ocr(str(img_path), cls=True)

        # 保存结果
        output = {
            "image": str(img_path),
            "ocr_result": _serialize_paddle_result(result),
        }

        # 使用 doc_parser 获取结构化输出
        try:
            from paddleocr import doc_parser
            parsed = doc_parser(str(img_path))
            output["markdown"] = parsed.get("markdown", "")
            output["layout"] = parsed.get("layout", [])
        except (ImportError, AttributeError):
            # doc_parser 可能不在所有版本中
            output["markdown"] = _build_markdown_from_ocr(result)
            output["layout"] = []

        json_path = os.path.join(output_dir, f"{img_path.stem}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        # 保存 Markdown
        md_path = os.path.join(output_dir, f"{img_path.stem}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(output["markdown"])

    logger.info(f"基线结果已保存: {output_dir}")


def _serialize_paddle_result(result) -> list:
    """序列化 PaddleOCR 结果"""
    if not result:
        return []
    serialized = []
    for page in result:
        if not page:
            continue
        for item in page:
            box, (text, conf) = item
            serialized.append({
                "box": [[float(p[0]), float(p[1])] for p in box],
                "text": text,
                "confidence": float(conf),
            })
    return serialized


def _build_markdown_from_ocr(result) -> str:
    """从 OCR 结果构建简单 Markdown"""
    lines = []
    if result:
        for page in result:
            if not page:
                continue
            for item in page:
                _, (text, _) = item
                lines.append(text)
    return "\n".join(lines)


# ── QNN 推理 ──────────────────────────────────────────────────────────────────

def run_qnn(image_dir: str, output_dir: str, model_dir: str, backend: str):
    """使用 QNN 端侧推理"""
    os.makedirs(output_dir, exist_ok=True)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from doc_parser import DocParser

    parser = DocParser(model_dir=model_dir, backend=backend)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = sorted(
        f for f in Path(image_dir).iterdir()
        if f.suffix.lower() in image_extensions
    )

    logger.info(f"QNN 推理: {len(images)} 张图片")

    for img_path in images:
        result = parser.parse_with_fallback(str(img_path))

        # 保存结果
        json_path = os.path.join(output_dir, f"{img_path.stem}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        md_path = os.path.join(output_dir, f"{img_path.stem}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(result.markdown)

    logger.info(f"QNN 结果已保存: {output_dir}")


# ── 对比评测 ──────────────────────────────────────────────────────────────────

def compare_results(baseline_dir: str, qnn_dir: str) -> Dict:
    """对比 baseline 和 QNN 的推理结果"""
    metrics = {
        "per_image": [],
        "summary": {},
    }

    baseline_files = sorted(Path(baseline_dir).glob("*.json"))
    logger.info(f"对比 {len(baseline_files)} 个文件的结果")

    all_cer = []
    all_wer = []
    table_metrics = []

    for bl_json in baseline_files:
        stem = bl_json.stem
        qnn_json = Path(qnn_dir) / f"{stem}.json"

        if not qnn_json.exists():
            logger.warning(f"  QNN 结果缺失: {stem}")
            continue

        with open(bl_json) as f:
            bl_data = json.load(f)
        with open(qnn_json) as f:
            qnn_data = json.load(f)

        # 对比 Markdown 输出
        bl_md = bl_data.get("markdown", "")
        qnn_md = qnn_data.get("markdown", "")

        cer = character_error_rate(bl_md, qnn_md)
        wer = word_error_rate(bl_md, qnn_md)
        all_cer.append(cer)
        all_wer.append(wer)

        image_metric = {
            "image": stem,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "accuracy": round(1 - cer, 4),
        }

        # 表格对比（如果有）
        bl_regions = bl_data.get("layout", []) or bl_data.get("regions", [])
        qnn_regions = qnn_data.get("regions", [])

        bl_tables = [r for r in bl_regions if r.get("label") == "table"]
        qnn_tables = [r for r in qnn_regions if r.get("label") == "table"]

        if bl_tables and qnn_tables:
            for bt, qt in zip(bl_tables, qnn_tables):
                bl_content = bt.get("content", "")
                qnn_content = qt.get("content", "")
                tm = compute_table_accuracy(qnn_content, bl_content)
                table_metrics.append(tm)
                image_metric["table_accuracy"] = tm

        metrics["per_image"].append(image_metric)
        logger.info(f"  {stem}: CER={cer:.4f}, 准确率={1-cer:.4f}")

    # 汇总
    if all_cer:
        avg_cer = np.mean(all_cer)
        avg_wer = np.mean(all_wer)
        accuracy = 1 - avg_cer

        metrics["summary"] = {
            "num_images": len(all_cer),
            "avg_cer": round(float(avg_cer), 4),
            "avg_wer": round(float(avg_wer), 4),
            "text_accuracy": round(float(accuracy), 4),
            "accuracy_loss": round(float(avg_cer) * 100, 2),  # 百分比
            "pass_threshold": float(avg_cer) <= 0.05,  # ≤5% 损失
        }

        if table_metrics:
            metrics["summary"]["table"] = {
                "avg_row_accuracy": round(float(np.mean([t["row_accuracy"] for t in table_metrics])), 4),
                "avg_cell_accuracy": round(float(np.mean([t["cell_accuracy"] for t in table_metrics])), 4),
                "avg_structure_score": round(float(np.mean([t["structure_score"] for t in table_metrics])), 4),
            }

    return metrics


# ── 报告生成 ──────────────────────────────────────────────────────────────────

def generate_report(results_dir: str):
    """生成精度对比报告"""
    baseline_dir = os.path.join(results_dir, "baseline")
    qnn_dir = os.path.join(results_dir, "qnn")

    if not os.path.exists(baseline_dir) or not os.path.exists(qnn_dir):
        logger.error("请先运行 baseline 和 qnn 推理")
        return

    metrics = compare_results(baseline_dir, qnn_dir)

    # 保存详细结果
    metrics_path = os.path.join(results_dir, "accuracy_comparison.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 生成 Markdown 报告
    report = _format_report(metrics)
    report_path = os.path.join(results_dir, "精度对比报告.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"报告已生成: {report_path}")
    print(report)


def _format_report(metrics: Dict) -> str:
    """格式化精度对比报告"""
    summary = metrics.get("summary", {})
    per_image = metrics.get("per_image", [])

    report = """# 精度对比报告：QNN 端侧推理 vs PaddlePaddle 原始推理

## 测试概览

| 指标 | 值 |
|------|-----|
| 测试图片数 | {num_images} |
| 平均字符错误率 (CER) | {avg_cer:.2%} |
| 平均词错误率 (WER) | {avg_wer:.2%} |
| **文本识别准确率** | **{text_accuracy:.2%}** |
| **精度损失** | **{accuracy_loss:.2f}%** |
| 是否满足 ≤5% 阈值 | {pass_status} |

## 验收结论

{conclusion}

## 逐图详细结果

| 图片 | CER | WER | 准确率 |
|------|-----|-----|--------|
""".format(
        num_images=summary.get("num_images", 0),
        avg_cer=summary.get("avg_cer", 0),
        avg_wer=summary.get("avg_wer", 0),
        text_accuracy=summary.get("text_accuracy", 0),
        accuracy_loss=summary.get("accuracy_loss", 0),
        pass_status="✓ 通过" if summary.get("pass_threshold") else "✗ 未通过",
        conclusion=(
            "端侧推理精度损失在 5% 以内，满足验收要求。"
            if summary.get("pass_threshold")
            else "精度损失超过 5%，需进一步优化量化策略。\n\n建议：\n"
            "1. VL 模型从 INT8 切换到 FP16\n"
            "2. 增加校准数据集样本数\n"
            "3. 对敏感层使用混合精度"
        ),
    )

    for item in per_image:
        report += "| {image} | {cer:.4f} | {wer:.4f} | {accuracy:.2%} |\n".format(**item)

    # 表格评测
    if "table" in summary:
        table = summary["table"]
        report += f"""
## 表格结构保留率

| 指标 | 值 |
|------|-----|
| 行数准确率 | {table['avg_row_accuracy']:.2%} |
| 单元格准确率 | {table['avg_cell_accuracy']:.2%} |
| 结构一致性 | {table['avg_structure_score']:.2%} |
"""

    report += """
## 测试环境

| 项目 | 值 |
|------|-----|
| 基线环境 | PaddlePaddle + PaddleOCR-VL (CPU/GPU) |
| 端侧环境 | QNN SDK + HTP (Hexagon NPU) |
| 量化策略 | 布局检测: INT8, VL 模型: FP16 |
| 校准数据 | 100 张文档图片 |

---

*报告由自动化评测脚本生成*
"""
    return report


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PaddleOCR-VL 精度对比评测")
    subparsers = parser.add_subparsers(dest="command")

    # baseline
    bl_parser = subparsers.add_parser("baseline", help="运行 PaddlePaddle 基线推理")
    bl_parser.add_argument("--image_dir", required=True)
    bl_parser.add_argument("--output_dir", default="./results/baseline")

    # qnn
    qnn_parser = subparsers.add_parser("qnn", help="运行 QNN 端侧推理")
    qnn_parser.add_argument("--image_dir", required=True)
    qnn_parser.add_argument("--output_dir", default="./results/qnn")
    qnn_parser.add_argument("--model_dir", default="./models/qnn/context_binaries")
    qnn_parser.add_argument("--backend", default="htp")

    # compare
    cmp_parser = subparsers.add_parser("compare", help="对比评测")
    cmp_parser.add_argument("--baseline_dir", default="./results/baseline")
    cmp_parser.add_argument("--qnn_dir", default="./results/qnn")

    # report
    rpt_parser = subparsers.add_parser("report", help="生成报告")
    rpt_parser.add_argument("--results_dir", default="./results")

    args = parser.parse_args()

    if args.command == "baseline":
        run_baseline(args.image_dir, args.output_dir)
    elif args.command == "qnn":
        run_qnn(args.image_dir, args.output_dir, args.model_dir, args.backend)
    elif args.command == "compare":
        metrics = compare_results(args.baseline_dir, args.qnn_dir)
        print(json.dumps(metrics["summary"], indent=2, ensure_ascii=False))
    elif args.command == "report":
        generate_report(args.results_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
