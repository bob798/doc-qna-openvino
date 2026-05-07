#!/usr/bin/env python3
"""
基于 QNN 端侧推理的页面级文档解析 Pipeline

参考 PaddleOCR CLI 的 doc_parser，实现完整的文档解析流程：
    输入：单页文档图片
    输出：结构化 Markdown 解析结果

Pipeline 流程：
    1. 布局检测 → 识别文本块、表格、公式、图表区域
    2. 区域裁剪 → 按检测框裁剪子图
    3. VL 识别 → 对每个区域执行视觉语言识别
    4. 结果组装 → 按阅读顺序拼装 Markdown

使用方式：
    python doc_parser.py --image document.png --output result.md
    python doc_parser.py --image_dir ./test_images --output_dir ./results
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 版面元素类型 ──────────────────────────────────────────────────────────────

LAYOUT_LABELS = {
    0: "text",
    1: "title",
    2: "table",
    3: "figure",
    4: "formula",
    5: "list",
    6: "header",
    7: "footer",
    8: "caption",
}

# 各类型对应的 VL 识别 prompt
VL_PROMPTS = {
    "text": "识别图中所有文字内容，保持段落格式",
    "title": "识别图中的标题文字",
    "table": "识别图中的表格，输出为 Markdown 表格格式，保持行列结构",
    "figure": "描述图中的图表内容",
    "formula": "识别图中的数学公式，输出为 LaTeX 格式",
    "list": "识别图中的列表内容，保持列表格式",
    "header": "识别页眉内容",
    "footer": "识别页脚内容",
    "caption": "识别图表标题",
}


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class LayoutRegion:
    """版面检测区域"""

    def __init__(self, box: np.ndarray, score: float, label: int):
        self.box = box          # [x1, y1, x2, y2]
        self.score = score
        self.label = label
        self.label_name = LAYOUT_LABELS.get(label, f"unknown_{label}")
        self.content = ""       # VL 识别结果

    @property
    def y_center(self) -> float:
        return (self.box[1] + self.box[3]) / 2

    @property
    def x_center(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def area(self) -> float:
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])


class ParseResult:
    """文档解析结果"""

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.regions: List[LayoutRegion] = []
        self.markdown = ""
        self.timing = {}  # 各阶段耗时

    def to_markdown(self) -> str:
        return self.markdown

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "num_regions": len(self.regions),
            "regions": [
                {
                    "label": r.label_name,
                    "score": round(r.score, 4),
                    "box": r.box.tolist(),
                    "content": r.content,
                }
                for r in self.regions
            ],
            "markdown": self.markdown,
            "timing": self.timing,
        }


# ── 文档解析 Pipeline ─────────────────────────────────────────────────────────

class DocParser:
    """
    页面级文档解析器

    串联布局检测和 VL 识别模型，实现完整的文档解析功能。
    """

    def __init__(
        self,
        model_dir: str = "./models/qnn/context_binaries",
        sdk_path: Optional[str] = None,
        backend: str = "htp",
        confidence_threshold: float = 0.5,
        use_vl_for_all: bool = False,
    ):
        """
        Args:
            model_dir: QNN context binary 目录
            sdk_path: QNN SDK 路径
            backend: 推理后端 (htp / htp_simulator / cpu)
            confidence_threshold: 布局检测置信度阈值
            use_vl_for_all: 是否对所有区域使用 VL 识别（False 时图表仅标记）
        """
        self.confidence_threshold = confidence_threshold
        self.use_vl_for_all = use_vl_for_all

        # 加载推理服务
        from qnn_inference import QNNInferenceService
        self.service = QNNInferenceService(
            model_dir=model_dir,
            sdk_path=sdk_path,
            backend=backend,
        )

    def parse(self, image_path: str) -> ParseResult:
        """
        解析单页文档图片

        Args:
            image_path: 图片路径

        Returns:
            ParseResult 对象，包含版面区域、识别内容和 Markdown
        """
        from PIL import Image

        result = ParseResult(image_path)
        total_t0 = time.time()

        # 加载图片
        image = np.array(Image.open(image_path).convert("RGB"))
        h, w = image.shape[:2]
        logger.info(f"解析文档: {image_path} ({w}x{h})")

        # ── Step 1: 布局检测 ──
        t0 = time.time()
        boxes, scores, labels = self.service.detect_layout(image)
        result.timing["layout_detection"] = time.time() - t0

        # 过滤低置信度检测
        regions = []
        for i in range(len(scores)):
            if scores[i] >= self.confidence_threshold:
                region = LayoutRegion(
                    box=boxes[i],
                    score=float(scores[i]),
                    label=int(labels[i]),
                )
                regions.append(region)

        logger.info(f"  检测到 {len(regions)} 个版面区域 (阈值={self.confidence_threshold})")
        for r in regions:
            logger.info(f"    [{r.label_name}] score={r.score:.3f}")

        # ── Step 2: 按阅读顺序排序 ──
        regions = self._sort_reading_order(regions, w)

        # ── Step 3: VL 识别 ──
        t0 = time.time()
        for region in regions:
            # 裁剪子图
            x1, y1, x2, y2 = [int(v) for v in region.box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # 选择识别 prompt
            prompt = VL_PROMPTS.get(region.label_name, VL_PROMPTS["text"])

            # 图表可选跳过 VL
            if region.label_name == "figure" and not self.use_vl_for_all:
                region.content = f"[图片: {x2-x1}x{y2-y1}]"
                continue

            # 执行 VL 识别
            try:
                region.content = self.service.recognize_vl(crop, prompt)
            except Exception as e:
                logger.warning(f"    VL 识别失败 ({region.label_name}): {e}")
                region.content = f"[识别失败: {region.label_name}]"

        result.timing["vl_recognition"] = time.time() - t0

        result.regions = regions

        # ── Step 4: 组装 Markdown ──
        result.markdown = self._assemble_markdown(regions)
        result.timing["total"] = time.time() - total_t0

        logger.info(f"  解析完成，总耗时: {result.timing['total']*1000:.0f} ms")
        return result

    def parse_with_fallback(self, image_path: str) -> ParseResult:
        """
        带 fallback 的解析：如果布局检测失败，直接用 VL 模型全图识别

        适用于简单页面或布局检测模型不可用的场景。
        """
        result = self.parse(image_path)

        if not result.regions:
            logger.info("  布局检测无结果，fallback 到全图 VL 识别")
            from PIL import Image
            image = np.array(Image.open(image_path).convert("RGB"))

            t0 = time.time()
            content = self.service.recognize_vl(
                image,
                "请完整识别这个文档页面的所有内容（包括文字、表格、公式），"
                "输出为结构化 Markdown 格式。",
            )
            result.timing["vl_fullpage"] = time.time() - t0

            region = LayoutRegion(
                box=np.array([0, 0, image.shape[1], image.shape[0]]),
                score=1.0,
                label=0,
            )
            region.content = content
            result.regions = [region]
            result.markdown = content

        return result

    # ── 阅读顺序排序 ─────────────────────────────────────────────────────────

    def _sort_reading_order(
        self, regions: List[LayoutRegion], page_width: int
    ) -> List[LayoutRegion]:
        """
        按阅读顺序排序版面区域

        策略：
        1. 按 y 坐标分行（y 差距 < 行高阈值 → 同一行）
        2. 同一行内按 x 坐标从左到右排序
        """
        if not regions:
            return regions

        # 按 y_center 排序
        regions.sort(key=lambda r: r.y_center)

        # 分行：相邻区域 y 差距 < 阈值则同行
        lines = []
        current_line = [regions[0]]
        line_threshold = 30  # 像素

        for r in regions[1:]:
            if abs(r.y_center - current_line[-1].y_center) < line_threshold:
                current_line.append(r)
            else:
                lines.append(current_line)
                current_line = [r]
        lines.append(current_line)

        # 每行内按 x 排序
        sorted_regions = []
        for line in lines:
            line.sort(key=lambda r: r.x_center)
            sorted_regions.extend(line)

        return sorted_regions

    # ── Markdown 组装 ─────────────────────────────────────────────────────────

    def _assemble_markdown(self, regions: List[LayoutRegion]) -> str:
        """将各区域识别结果组装为完整 Markdown"""
        parts = []

        for region in regions:
            content = region.content.strip()
            if not content:
                continue

            if region.label_name == "title":
                # 标题：根据位置推断层级（简单启发式）
                level = self._infer_title_level(region, regions)
                parts.append(f"{'#' * level} {content}")
                parts.append("")

            elif region.label_name == "table":
                # 表格：内容应已是 Markdown 表格格式
                parts.append(content)
                parts.append("")

            elif region.label_name == "formula":
                # 公式：包裹在 $$ 中
                if not content.startswith("$$"):
                    content = f"$$ {content} $$"
                parts.append(content)
                parts.append("")

            elif region.label_name == "figure":
                parts.append(content)
                parts.append("")

            elif region.label_name in ("header", "footer"):
                # 页眉页脚：注释形式
                parts.append(f"<!-- {region.label_name}: {content} -->")

            elif region.label_name == "caption":
                parts.append(f"*{content}*")
                parts.append("")

            else:
                # 文本块
                parts.append(content)
                parts.append("")

        return "\n".join(parts)

    def _infer_title_level(
        self, title_region: LayoutRegion, all_regions: List[LayoutRegion]
    ) -> int:
        """根据字体大小（区域高度）推断标题层级"""
        title_height = title_region.box[3] - title_region.box[1]

        # 收集所有标题的高度
        title_heights = []
        for r in all_regions:
            if r.label_name == "title":
                title_heights.append(r.box[3] - r.box[1])

        if not title_heights:
            return 2

        max_h = max(title_heights)
        min_h = min(title_heights)
        range_h = max_h - min_h

        if range_h < 5:
            return 2

        # 归一化高度 → 层级
        ratio = (title_height - min_h) / range_h
        if ratio > 0.7:
            return 1
        elif ratio > 0.3:
            return 2
        else:
            return 3


# ── 批量解析 ──────────────────────────────────────────────────────────────────

def parse_directory(
    parser: DocParser,
    image_dir: str,
    output_dir: str,
):
    """批量解析目录中的所有文档图片"""
    import json

    os.makedirs(output_dir, exist_ok=True)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    images = sorted(
        f for f in Path(image_dir).iterdir()
        if f.suffix.lower() in image_extensions
    )

    logger.info(f"批量解析: {len(images)} 张图片")
    results = []

    for img_path in images:
        result = parser.parse_with_fallback(str(img_path))

        # 保存 Markdown
        md_path = os.path.join(output_dir, f"{img_path.stem}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(result.markdown)

        # 保存详细结果
        json_path = os.path.join(output_dir, f"{img_path.stem}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        results.append(result)
        logger.info(f"  ✓ {img_path.name} → {md_path}")

    # 汇总报告
    _print_summary(results)
    return results


def _print_summary(results: List[ParseResult]):
    """打印批量解析汇总"""
    logger.info("=" * 60)
    logger.info("解析汇总")
    logger.info("=" * 60)

    total_regions = sum(len(r.regions) for r in results)
    total_time = sum(r.timing.get("total", 0) for r in results)

    label_counts = {}
    for r in results:
        for region in r.regions:
            label_counts[region.label_name] = label_counts.get(region.label_name, 0) + 1

    logger.info(f"  解析页数: {len(results)}")
    logger.info(f"  总区域数: {total_regions}")
    logger.info(f"  区域分布:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        logger.info(f"    {label}: {count}")
    logger.info(f"  总耗时: {total_time*1000:.0f} ms")
    logger.info(f"  平均每页: {total_time/max(len(results),1)*1000:.0f} ms")


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="基于 QNN 的页面级文档解析")
    parser.add_argument("--image", type=str, help="单张文档图片路径")
    parser.add_argument("--image_dir", type=str, help="批量文档图片目录")
    parser.add_argument("--output", type=str, default=None, help="输出 Markdown 文件路径")
    parser.add_argument("--output_dir", type=str, default="./results", help="批量输出目录")
    parser.add_argument("--model_dir", type=str, default="./models/qnn/context_binaries",
                        help="QNN 模型目录")
    parser.add_argument("--backend", choices=["htp", "htp_simulator", "cpu"], default="htp",
                        help="推理后端")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="布局检测置信度阈值")
    parser.add_argument("--fullpage", action="store_true",
                        help="跳过布局检测，全图 VL 识别")

    args = parser.parse_args()

    # 创建解析器
    doc_parser = DocParser(
        model_dir=args.model_dir,
        backend=args.backend,
        confidence_threshold=args.confidence,
    )

    if args.image:
        # 单页解析
        if args.fullpage:
            result = doc_parser.parse_with_fallback(args.image)
        else:
            result = doc_parser.parse(args.image)

        # 输出结果
        print("\n" + "=" * 60)
        print("文档解析结果")
        print("=" * 60)
        print(result.markdown)

        # 保存文件
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result.markdown)
            logger.info(f"结果已保存: {args.output}")

        # 打印耗时
        print("\n--- 耗时统计 ---")
        for stage, t in result.timing.items():
            print(f"  {stage}: {t*1000:.0f} ms")

    elif args.image_dir:
        # 批量解析
        parse_directory(doc_parser, args.image_dir, args.output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
