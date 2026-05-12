#!/usr/bin/env python3
"""
OmniDocBench 子集对比骨架：Tesseract vs PaddleOCR-VL

流程：
  1. 加载 `data/omnidocbench/OmniDocBench.json`（由 `download_eval_materials.py` 拉取）
  2. 按 doc_type 过滤（默认 academic_literature / financial_report / textbook）
  3. 随机采样 N 页（可复现 seed）
  4. 逐页跑 Tesseract + PaddleOCR-VL，导出 `results/omnidocbench/predictions.json`
  5. **指标计算交给官方**：clone `https://github.com/opendatalab/OmniDocBench` 后用其
     `pdf_validation/` 工具跑 Edit Distance / TEDS / CDM，本脚本只产 predictions

用法：
    python scripts/run_omnidocbench_subset.py --n 20 --seed 42
    python scripts/run_omnidocbench_subset.py --n 5 --tesseract-only   # 冒烟
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ANNOTATION = PROJECT_ROOT / "data" / "omnidocbench" / "OmniDocBench.json"
DEFAULT_OUT = PROJECT_ROOT / "results" / "omnidocbench"
DEFAULT_DOC_TYPES = ("academic_literature", "financial_report", "textbook")


def load_annotation(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(f"未找到标注文件 {path}，先跑 `python scripts/download_eval_materials.py`")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("data", [])


def filter_and_sample(items: list, doc_types: tuple[str, ...], n: int, seed: int) -> list:
    """按 page_info.page_attribute.data_source 过滤；列表为空则不过滤。"""
    def _doc_type(item: dict) -> str | None:
        pa = ((item.get("page_info") or {}).get("page_attribute")) or {}
        return pa.get("data_source")

    filtered = [it for it in items if _doc_type(it) in doc_types] if doc_types else items
    if not filtered:
        logger.warning("过滤后 0 条 (doc_types=%s)，回退到无过滤", doc_types)
        filtered = items
    rng = random.Random(seed)
    rng.shuffle(filtered)
    return filtered[:n]


def run_tesseract(image_path: Path, lang: str = "chi_sim+eng") -> dict:
    """单页 Tesseract OCR，返回 {"text": str, "elapsed_ms": float}。"""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        return {"text": None, "error": f"pytesseract/Pillow 缺失: {exc}", "elapsed_ms": 0.0}

    t0 = time.perf_counter()
    text = pytesseract.image_to_string(Image.open(image_path), lang=lang)
    return {"text": text, "elapsed_ms": (time.perf_counter() - t0) * 1000}


def run_paddleocr_vl(image_path: Path) -> dict:
    """单页 PaddleOCR-VL OpenVINO 推理，复用 src/inference.py。"""
    try:
        from src.inference import get_default_engine  # type: ignore
    except ImportError as exc:
        return {"text": None, "error": f"src.inference 不可用: {exc}", "elapsed_ms": 0.0}

    engine = get_default_engine()  # 单例，第一次调用加载 IR
    t0 = time.perf_counter()
    result = engine.infer(str(image_path), prompt="OCR")
    return {
        "text": result.text if hasattr(result, "text") else str(result),
        "elapsed_ms": (time.perf_counter() - t0) * 1000,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_ANNOTATION.parent,
                        help="OmniDocBench 数据根目录（用于解析 image 相对路径）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=20, help="采样数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--doc-types", nargs="*", default=list(DEFAULT_DOC_TYPES),
                        help="过滤的文档类型；传空字符串表示不过滤")
    parser.add_argument("--tesseract-only", action="store_true", help="只跑 Tesseract（冒烟用）")
    parser.add_argument("--paddleocr-only", action="store_true", help="只跑 PaddleOCR-VL")
    args = parser.parse_args()

    items = load_annotation(args.annotation)
    sampled = filter_and_sample(items, tuple(args.doc_types), args.n, args.seed)
    logger.info("采样 %d / %d 条 (doc_types=%s, seed=%d)", len(sampled), len(items), args.doc_types, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    predictions = []

    for idx, item in enumerate(sampled, 1):
        pi = item.get("page_info") or {}
        img_rel = pi.get("image_path") or pi.get("img_path") or pi.get("image")
        if not isinstance(img_rel, str):
            logger.warning("[%d/%d] 缺少 page_info.image_path，跳过", idx, len(sampled))
            continue
        if not img_rel.startswith("images/"):
            img_rel = f"images/{img_rel}"
        img_path = args.image_root / img_rel
        if not img_path.exists():
            logger.warning("[%d/%d] 文件不存在 %s，跳过", idx, len(sampled), img_path)
            continue

        record = {"id": item.get("id") or img_rel, "image_path": str(img_rel)}
        if not args.paddleocr_only:
            logger.info("[%d/%d] Tesseract %s", idx, len(sampled), img_rel)
            record["tesseract"] = run_tesseract(img_path)
        if not args.tesseract_only:
            logger.info("[%d/%d] PaddleOCR-VL %s", idx, len(sampled), img_rel)
            record["paddleocr_vl"] = run_paddleocr_vl(img_path)
        predictions.append(record)

    out_file = args.out / "predictions.json"
    out_file.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("写入 %d 条 predictions → %s", len(predictions), out_file)

    print()
    print("=" * 70)
    print("下一步：用 OmniDocBench 官方脚本计算指标")
    print("=" * 70)
    print("""
  git clone https://github.com/opendatalab/OmniDocBench
  cd OmniDocBench/pdf_validation
  # 按 README 调整 config，将本脚本输出的 predictions.json 作为输入
  python pdf_validation.py --predictions {out_file} --annotation {annotation}
""".format(out_file=out_file, annotation=args.annotation))
    return 0


if __name__ == "__main__":
    sys.exit(main())
