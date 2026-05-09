"""
PDF 预处理：可读性判断 + 按页渲染

策略：
  1. 用 pdfplumber 抽取每页文字层
  2. 若 90% 以上页面的字符数 ≥ MIN_CHARS_PER_PAGE，判定为「文字 PDF」，走 text-layer 路径
  3. 否则走 OCR 路径：用 pdf2image 把页面渲染成 PIL.Image，交给 PaddleOCR-VL
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

MIN_CHARS_PER_PAGE = 50          # 单页文字层字数下限
MIN_TEXT_PAGE_RATIO = 0.9        # 命中文字层的页占比阈值
DEFAULT_DPI = 200                # 渲染 DPI（与 PaddleOCR-VL 推荐输入相符）


@dataclass
class PageContent:
    page_no: int                  # 1-indexed
    source: str                   # "text_layer" | "ocr"
    text: Optional[str] = None    # text_layer 命中时的文字
    image_path: Optional[Path] = None  # OCR 路径下的图片缓存


@dataclass
class PreprocessResult:
    pdf_path: Path
    mode: str                     # "text" | "ocr" | "mixed"
    pages: List[PageContent]

    @property
    def num_pages(self) -> int:
        return len(self.pages)


def _extract_text_pages(pdf_path: Path) -> List[Optional[str]]:
    """对每页尝试抽取文字层。失败的页返回 None。"""
    try:
        import pdfplumber
    except ImportError as e:
        raise ImportError("需要 pdfplumber：`pip install pdfplumber`") from e

    pages: List[Optional[str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            try:
                txt = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"  page {page.page_number} extract_text 失败: {e}")
                txt = ""
            pages.append(txt)
    return pages


def _render_pages_to_images(
    pdf_path: Path, out_dir: Path, dpi: int, pages: Optional[List[int]] = None
) -> List[Path]:
    """优先用 pypdfium2（自带二进制），不可用时回退到 pdf2image+poppler"""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(pdf_path))
        n = len(pdf)
        target_pages = pages if pages else list(range(1, n + 1))
        scale = dpi / 72.0
        out_paths = []
        for p in target_pages:
            page = pdf[p - 1]
            pil = page.render(scale=scale).to_pil()
            out_path = out_dir / f"page_{p:04d}.png"
            pil.save(out_path)
            out_paths.append(out_path)
        return out_paths
    except ImportError:
        pass

    try:
        from pdf2image import convert_from_path
    except ImportError as e:
        raise ImportError(
            "需要 pypdfium2 或 pdf2image：`pip install pypdfium2`"
        ) from e

    kwargs = {"dpi": dpi, "fmt": "png", "output_folder": str(out_dir), "paths_only": True}
    if pages:
        kwargs["first_page"] = min(pages)
        kwargs["last_page"] = max(pages)
    image_paths = convert_from_path(str(pdf_path), **kwargs)
    return [Path(p) for p in image_paths]


def preprocess_pdf(
    pdf_path: str | Path,
    cache_dir: Optional[str | Path] = None,
    dpi: int = DEFAULT_DPI,
    force_ocr: bool = False,
) -> PreprocessResult:
    """
    执行 PDF 可读性判断 + 必要的页面渲染。

    Args:
        pdf_path: PDF 路径
        cache_dir: 渲染图片缓存目录，默认 `<pdf>.cache/`
        dpi: 渲染 DPI
        force_ocr: 强制走 OCR 路径（用于 Benchmark / 对比实验）
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    cache_dir = Path(cache_dir) if cache_dir else pdf_path.with_suffix("").with_name(
        pdf_path.stem + ".cache"
    )

    text_pages = _extract_text_pages(pdf_path)
    n = len(text_pages)
    text_hit = sum(1 for t in text_pages if t and len(t.strip()) >= MIN_CHARS_PER_PAGE)
    text_ratio = text_hit / max(n, 1)
    logger.info(
        f"  {pdf_path.name}: {n} 页, 文字层命中 {text_hit}/{n} ({text_ratio:.1%})"
    )

    use_text_layer = (not force_ocr) and (text_ratio >= MIN_TEXT_PAGE_RATIO)
    mode = "text" if use_text_layer else ("ocr" if force_ocr or text_ratio == 0 else "mixed")

    pages: List[PageContent] = []
    if use_text_layer:
        for i, t in enumerate(text_pages, start=1):
            pages.append(PageContent(page_no=i, source="text_layer", text=t or ""))
        logger.info(f"  → 走 text-layer 路径")
        return PreprocessResult(pdf_path=pdf_path, mode=mode, pages=pages)

    # 否则需要渲染图片：mixed 时只渲染缺失页，ocr 时渲染所有页
    if mode == "mixed" and not force_ocr:
        ocr_pages = [
            i + 1 for i, t in enumerate(text_pages)
            if not t or len(t.strip()) < MIN_CHARS_PER_PAGE
        ]
    else:
        ocr_pages = list(range(1, n + 1))

    logger.info(f"  → 渲染 {len(ocr_pages)} 页用于 OCR @ {dpi} DPI")
    image_paths = _render_pages_to_images(pdf_path, cache_dir, dpi, pages=ocr_pages)
    # pdf2image 的输出顺序和 first_page..last_page 一致
    img_iter = iter(image_paths)

    for i, t in enumerate(text_pages, start=1):
        if i in ocr_pages:
            try:
                img = next(img_iter)
            except StopIteration:
                img = None
            pages.append(PageContent(page_no=i, source="ocr", image_path=img))
        else:
            pages.append(PageContent(page_no=i, source="text_layer", text=t or ""))

    return PreprocessResult(pdf_path=pdf_path, mode=mode, pages=pages)
