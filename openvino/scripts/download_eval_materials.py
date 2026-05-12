#!/usr/bin/env python3
"""
下载 Phase 3 / Phase 4 评测所需的公开材料

自动下载（直链 PDF，无需登录）：
  - PaddleOCR-VL 论文 (arXiv 2510.14528)
  - PaddleOCR-VL-1.5 论文 (arXiv 2601.21957)
  - NVIDIA H100 PCIe Product Brief (PB-11133-001)
  - NVIDIA H100 NVL Product Brief (PB-11773-001)

可选自动下载（需 `pip install huggingface_hub`）：
  - OmniDocBench 标注 + N 个样本（HF 数据集 opendatalab/OmniDocBench）

人工下载（站点限制，README 留地址）：
  - 昆仑芯 P800 / R200 规格书 - 主题贴合文心赛道
  - 小米路由器 / 手环说明书 - 消费级中文
  - GB/T 国标 PDF - 扫描件路径补强

用法：
    python scripts/download_eval_materials.py
    python scripts/download_eval_materials.py --skip omnidocbench
    python scripts/download_eval_materials.py --omnidoc-samples 20
"""

import argparse
import json
import logging
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "data" / "eval_documents"
OMNIDOC_OUT = PROJECT_ROOT / "data" / "omnidocbench"

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 doc-qna-openvino/0.1"

DIRECT_PDFS = [
    {
        "id": "paddleocr_vl_paper",
        "url": "https://arxiv.org/pdf/2510.14528",
        "filename": "paddleocr_vl_2510.14528.pdf",
        "min_size_kb": 500,
    },
    {
        "id": "paddleocr_vl_15_paper",
        "url": "https://arxiv.org/pdf/2601.21957",
        "filename": "paddleocr_vl_15_2601.21957.pdf",
        "min_size_kb": 500,
    },
    {
        "id": "h100_pcie_pb",
        "url": "https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/data-center/h100/PB-11133-001_v01.pdf",
        "filename": "h100_pcie_PB-11133-001.pdf",
        "min_size_kb": 200,
    },
    {
        "id": "h100_nvl_pb",
        "url": "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/h100/PB-11773-001_v01.pdf",
        "filename": "h100_nvl_PB-11773-001.pdf",
        "min_size_kb": 200,
    },
    {
        "id": "xiaomi_band9_manual",
        "url": "https://object.pscloud.io/cms/cms/Uploads/file_0_912_157_0_0_4Vg1op.pdf",
        "filename": "xiaomi_band9_user_manual.pdf",
        "min_size_kb": 100,
    },
]

MANUAL_DOCS = [
    {
        "id": "kunlunxin_product_brief",
        "filename": "kunlunxin_product_brief.pdf",
        "url": "https://www.paddlepaddle.org.cn/documentation/docs/zh/hardware_support/xpu/xpu-p800_install_cn.html",
        "site": "https://www.kunlunxin.com/",
        "note": "昆仑芯 Product Brief（实际为 K100/K200 一代）- 主题贴合文心赛道。官方无公开 P800 datasheet，可在浏览器打开上面 PaddlePaddle XPU-P800 文档 → 打印为 PDF。当前样本是手工拿到的 K100/K200 PB，功能上等价（中文规格 + 表格），保存为 kunlunxin_product_brief.pdf",
    },
    {
        "id": "gb_t_2423_1",
        "filename": "gb_t_2423_1.pdf",
        "url": "https://openstd.samr.gov.cn/bzgk/gb/newGbInfo?hcno=4B30041DEFB4D9283C1DC9592735F67E",
        "site": "https://openstd.samr.gov.cn/bzgk/gb/",
        "note": "GB/T 2423.1-2008 环境试验 - 扫描+文字层混合，补 scanned 路径证据。浏览器打开上面 URL，点页面 \"PDF\" 按钮下载（cookie 鉴权，命令行不行），保存为 gb_t_2423_1.pdf",
    },
]


def download_pdf(url: str, dest: Path, min_size_kb: int = 100) -> bool:
    """单文件下载，带 UA + 大小校验。已存在且非空则跳过。"""
    if dest.exists() and dest.stat().st_size > min_size_kb * 1024:
        logger.info("[skip] %s (已存在, %d KB)", dest.name, dest.stat().st_size // 1024)
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp:
            data = resp.read()
        dest.write_bytes(data)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        logger.error("[fail] %s -> %s: %s", url, dest.name, exc)
        logger.error("       若提示 SSL 错误，可 `pip install certifi` 后重跑")
        return False

    size_kb = dest.stat().st_size // 1024
    if size_kb < min_size_kb:
        logger.warning("[warn] %s 大小 %d KB 低于阈值 %d KB（可能是错误页 HTML）", dest.name, size_kb, min_size_kb)
        return False
    logger.info("[ok]   %s (%d KB)", dest.name, size_kb)
    return True


def download_direct_pdfs(out_dir: Path) -> dict:
    """下载所有 DIRECT_PDFS 条目，返回 {id: local_path or None}。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for item in DIRECT_PDFS:
        dest = out_dir / item["filename"]
        ok = download_pdf(item["url"], dest, min_size_kb=item["min_size_kb"])
        result[item["id"]] = str(dest.relative_to(PROJECT_ROOT)) if ok else None
    return result


def download_omnidocbench(out_dir: Path, n_samples: int) -> dict:
    """下载 OmniDocBench 标注 + n_samples 个样本图像。

    - 默认只拉 annotation JSON（小）+ README。
    - n_samples > 0 时，再 hf_hub_download 拉具体图像。
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        logger.warning("[skip] OmniDocBench: 缺少 huggingface_hub，跳过。`pip install huggingface_hub` 后重跑。")
        return {"annotation": None, "samples": []}

    out_dir.mkdir(parents=True, exist_ok=True)
    repo_id = "opendatalab/OmniDocBench"

    logger.info("[hf]   拉取 OmniDocBench 标注 / README ...")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(out_dir),
            allow_patterns=["OmniDocBench.json", "README*.md", "*.md"],
        )
    except Exception as exc:
        logger.error("[fail] OmniDocBench annotation 下载失败: %s", exc)
        return {"annotation": None, "samples": []}

    ann_path = out_dir / "OmniDocBench.json"
    if not ann_path.exists():
        logger.warning("[warn] OmniDocBench.json 未找到，仓库结构可能已变更，检查 %s", out_dir)
        return {"annotation": None, "samples": []}

    if n_samples <= 0:
        logger.info("[ok]   OmniDocBench annotation only (n_samples=0)")
        return {"annotation": str(ann_path.relative_to(PROJECT_ROOT)), "samples": []}

    try:
        annotation = json.loads(ann_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("[fail] 解析 OmniDocBench.json: %s", exc)
        return {"annotation": str(ann_path.relative_to(PROJECT_ROOT)), "samples": []}

    items = annotation if isinstance(annotation, list) else annotation.get("data", [])
    if not items:
        logger.warning("[warn] OmniDocBench.json 结构未识别，跳过采样")
        return {"annotation": str(ann_path.relative_to(PROJECT_ROOT)), "samples": []}

    import random as _random
    rng = _random.Random(42)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    sampled = [items[i] for i in indices[:n_samples]]
    logger.info("[hf]   开始下载 %d 个样本图像 (seed=42, 共 %d 候选)", len(sampled), len(items))
    samples = []
    for idx, item in enumerate(sampled, 1):
        img_rel = _extract_image_path(item)
        if not img_rel:
            logger.debug("跳过样本 %d: 没有 image 字段 %s", idx, list(item.keys())[:5])
            continue
        try:
            local = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=img_rel,
                local_dir=str(out_dir),
            )
            samples.append(str(Path(local).relative_to(PROJECT_ROOT)))
            if idx % 5 == 0:
                logger.info("[hf]   %d/%d 完成", idx, len(sampled))
        except Exception as exc:
            logger.warning("跳过 %s: %s", img_rel, exc)

    logger.info("[ok]   OmniDocBench: annotation + %d 样本图像", len(samples))
    return {"annotation": str(ann_path.relative_to(PROJECT_ROOT)), "samples": samples}


def _extract_image_path(item: dict) -> str | None:
    """从 annotation 条目里拿图像在 HF 仓库的相对路径。

    OmniDocBench `page_info.image_path` 是裸文件名（如 `page-xxx.png`），
    仓库实际路径是 `images/<filename>`。
    """
    pi = item.get("page_info") or {}
    name = pi.get("image_path") or pi.get("img_path") or pi.get("image")
    if isinstance(name, str):
        return name if name.startswith("images/") else f"images/{name}"
    # 顶层 fallback（极少数 annotation 版本）
    for key in ("image_path", "image", "img_path"):
        v = item.get(key)
        if isinstance(v, str):
            return v if v.startswith("images/") else f"images/{v}"
    return None


def write_manifest(out_dir: Path, direct: dict | None, omnidoc: dict | None) -> None:
    """写 manifest，对人工材料 (MANUAL_DOCS) 检测本地是否到位。

    `direct` / `omnidoc` 传 None 表示本次未跑该阶段——保留 manifest 中已有值，避免被空 stub 覆盖。
    """
    manifest_path = out_dir / "manifest.json"
    prev = {}
    if manifest_path.exists():
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev = {}

    manual = []
    for item in MANUAL_DOCS:
        local = out_dir / item["filename"]
        item_out = dict(item)
        item_out["present"] = local.exists() and local.stat().st_size > 10_000
        item_out["local_path"] = str(local.relative_to(PROJECT_ROOT)) if item_out["present"] else None
        manual.append(item_out)

    manifest = {
        "direct_pdfs": direct if direct is not None else prev.get("direct_pdfs", {}),
        "omnidocbench": omnidoc if omnidoc is not None else prev.get("omnidocbench", {"annotation": None, "samples": []}),
        "manual": manual,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[ok]   manifest 写入: %s", manifest_path.relative_to(PROJECT_ROOT))


def print_manual_instructions(out_dir: Path) -> None:
    pending = [m for m in MANUAL_DOCS if not (out_dir / m["filename"]).exists()]
    if not pending:
        print()
        print("=" * 70)
        print("✓ 所有人工材料已到位")
        print("=" * 70)
        for item in MANUAL_DOCS:
            print(f"  [{item['id']}] {out_dir.joinpath(item['filename']).relative_to(PROJECT_ROOT)}")
        return

    print()
    print("=" * 70)
    print(f"以下 {len(pending)} 份材料站点限制，需手动下载到 openvino/data/eval_documents/：")
    print("=" * 70)
    for item in pending:
        print(f"\n  [{item['id']}]  → {item['filename']}")
        print(f"    url:  {item['url']}")
        print(f"    site: {item['site']}")
        print(f"    note: {item['note']}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="PDF 输出目录")
    parser.add_argument("--omnidoc-out", type=Path, default=OMNIDOC_OUT, help="OmniDocBench 输出目录")
    parser.add_argument("--omnidoc-samples", type=int, default=20, help="OmniDocBench 采样数（0=只拉标注，-1=全部）")
    parser.add_argument("--skip", choices=["pdfs", "omnidocbench"], action="append", default=[], help="跳过某个阶段，可多次")
    args = parser.parse_args()

    direct_result = None
    if "pdfs" not in args.skip:
        logger.info("=== 直链 PDF 下载 ===")
        direct_result = download_direct_pdfs(args.out)
    else:
        logger.info("[skip] 直链 PDF 阶段（保留 manifest 已有状态）")

    omnidoc_result = None
    if "omnidocbench" not in args.skip:
        logger.info("=== OmniDocBench 子集 ===")
        omnidoc_result = download_omnidocbench(args.omnidoc_out, args.omnidoc_samples)
    else:
        logger.info("[skip] OmniDocBench 阶段（保留 manifest 已有状态）")

    write_manifest(args.out, direct_result, omnidoc_result)
    print_manual_instructions(args.out)

    failed = [k for k, v in (direct_result or {}).items() if v is None]
    if failed:
        logger.warning("以下文档下载失败，请手动检查: %s", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
