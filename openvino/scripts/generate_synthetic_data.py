#!/usr/bin/env python3
"""
生成合成测试数据（用于在没有真实样本时跑通管线）：

- data/test_images/text_01..03.png         纯文字
- data/test_images/table_01..03.png        表格
- data/test_images/formula_01..02.png      公式
- data/test_images/mix_01..02.png          图文混排

- data/test_documents/text_pdf.pdf         纯文本 PDF（含文字层）
- data/test_documents/scanned.pdf          扫描件（无文字层，由图片合成）
- data/test_documents/spec_with_tables.pdf 含表格规格书

合成图为简单点阵渲染，仅用于走通端到端管线；真实评测请替换为产品文档。

运行：
    python scripts/generate_synthetic_data.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "test_images"
PDF_DIR = ROOT / "data" / "test_documents"
IMG_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

# Windows 系统中文字体优先
FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            try:
                return ImageFont.truetype(f, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


# ── 图片合成 ──────────────────────────────────────────────────────────────────


def draw_text_image(path: Path, lines: list[str], size=(1024, 720)):
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    font = load_font(28)
    y = 60
    for line in lines:
        draw.text((60, y), line, fill="black", font=font)
        y += 48
    img.save(path)
    print(f"  写入 {path.name}")


def draw_table_image(
    path: Path, header: list[str], rows: list[list[str]], title: str = "", size=(1024, 720)
):
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    font = load_font(28)
    title_font = load_font(34)
    if title:
        draw.text((60, 30), title, fill="black", font=title_font)

    # 表格区域
    x0, y0 = 60, 110 if title else 60
    col_w = (size[0] - 120) // len(header)
    row_h = 60

    # 表头
    for i, h in enumerate(header):
        rect = [x0 + i * col_w, y0, x0 + (i + 1) * col_w, y0 + row_h]
        draw.rectangle(rect, outline="black", width=2, fill="#eef")
        draw.text((rect[0] + 12, rect[1] + 12), h, fill="black", font=font)

    # 数据行
    for r_idx, row in enumerate(rows, start=1):
        y = y0 + r_idx * row_h
        for c_idx, cell in enumerate(row):
            rect = [x0 + c_idx * col_w, y, x0 + (c_idx + 1) * col_w, y + row_h]
            draw.rectangle(rect, outline="black", width=2)
            draw.text((rect[0] + 12, rect[1] + 12), cell, fill="black", font=font)

    img.save(path)
    print(f"  写入 {path.name}")


def draw_formula_image(path: Path, formula: str, caption: str, size=(1024, 480)):
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    font_big = load_font(60)
    font_small = load_font(28)
    draw.text((60, 80), formula, fill="black", font=font_big)
    draw.text((60, 240), caption, fill="black", font=font_small)
    img.save(path)
    print(f"  写入 {path.name}")


def draw_mix_image(
    path: Path, title: str, paragraph: str, table_rows: list[list[str]], size=(1024, 900)
):
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(36)
    body_font = load_font(26)
    draw.text((60, 30), title, fill="black", font=title_font)

    # 段落
    y = 100
    for line in paragraph.split("\n"):
        draw.text((60, y), line, fill="black", font=body_font)
        y += 38

    # 简单图示
    box = [60, y + 20, 320, y + 200]
    draw.rectangle(box, outline="black", width=3, fill="#fec")
    draw.text((box[0] + 30, box[1] + 70), "[示意图]", fill="black", font=body_font)

    # 表格
    x0, y0 = 360, y + 20
    header = table_rows[0]
    rows = table_rows[1:]
    col_w = 180
    row_h = 50
    for i, h in enumerate(header):
        rect = [x0 + i * col_w, y0, x0 + (i + 1) * col_w, y0 + row_h]
        draw.rectangle(rect, outline="black", width=2, fill="#eef")
        draw.text((rect[0] + 12, rect[1] + 10), h, fill="black", font=body_font)
    for r_idx, row in enumerate(rows, start=1):
        ry = y0 + r_idx * row_h
        for c_idx, cell in enumerate(row):
            rect = [x0 + c_idx * col_w, ry, x0 + (c_idx + 1) * col_w, ry + row_h]
            draw.rectangle(rect, outline="black", width=2)
            draw.text((rect[0] + 12, rect[1] + 10), cell, fill="black", font=body_font)

    img.save(path)
    print(f"  写入 {path.name}")


# ── PDF 合成 ──────────────────────────────────────────────────────────────────


def make_text_pdf(path: Path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    pdfmetrics.registerFont(TTFont("MSYH", "C:/Windows/Fonts/msyh.ttc"))
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setFont("MSYH", 18)
    c.drawString(60, 800, "文字 PDF 测试样本")
    c.setFont("MSYH", 13)
    body = [
        "本节用于验证 pdfplumber 文字层抽取路径。",
        "",
        "## 工作原理",
        "PaddleOCR-VL 接收图片输入并输出结构化 Markdown，",
        "包括标题、段落、表格、公式等区块的语义信息。",
        "",
        "## 注意事项",
        "若 PDF 含可抽取文字层，则应直接复用文字层，",
        "避免不必要的 OCR 开销。",
    ]
    y = 760
    for line in body:
        c.drawString(60, y, line)
        y -= 24
    c.showPage()
    c.setFont("MSYH", 13)
    c.drawString(60, 800, "## 第二页")
    c.drawString(60, 770, "本页用于验证多页解析。")
    c.save()
    print(f"  写入 {path.name}")


def make_scanned_pdf(path: Path, source_images: list[Path]):
    """用合成图片合成无文字层 PDF（用 reportlab 嵌图）"""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    page_w, page_h = A4
    c = canvas.Canvas(str(path), pagesize=A4)
    for img_path in source_images:
        with Image.open(img_path) as im:
            iw, ih = im.size
        scale = min((page_w - 80) / iw, (page_h - 80) / ih)
        w, h = iw * scale, ih * scale
        x = (page_w - w) / 2
        y = (page_h - h) / 2
        c.drawImage(str(img_path), x, y, w, h, preserveAspectRatio=True)
        c.showPage()
    c.save()
    print(f"  写入 {path.name} ({len(source_images)} 页)")


def make_spec_pdf(path: Path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    pdfmetrics.registerFont(TTFont("MSYH", "C:/Windows/Fonts/msyh.ttc"))
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setFont("MSYH", 18)
    c.drawString(60, 800, "产品规格书（含表格）")
    c.setFont("MSYH", 13)
    c.drawString(60, 760, "## 主要参数")

    headers = ["型号", "功率", "工作温度"]
    rows = [
        ["A100", "300W", "-20~70℃"],
        ["A200", "400W", "-10~60℃"],
        ["A300", "500W", "0~50℃"],
    ]
    x0, y0 = 60, 720
    col_w = 150
    row_h = 28
    for i, h in enumerate(headers):
        c.rect(x0 + i * col_w, y0 - row_h, col_w, row_h, stroke=1)
        c.drawString(x0 + i * col_w + 8, y0 - row_h + 8, h)
    for r_idx, row in enumerate(rows, start=1):
        ry = y0 - (r_idx + 1) * row_h
        for ci, cell in enumerate(row):
            c.rect(x0 + ci * col_w, ry, col_w, row_h, stroke=1)
            c.drawString(x0 + ci * col_w + 8, ry + 8, cell)

    y = y0 - (len(rows) + 2) * row_h - 20
    c.drawString(60, y, "## 故障代码")
    y -= 24
    for line in [
        "请参考下表常见故障代码与处理方式。",
        "如遇代码 E01/E02 请立即联系售后。",
    ]:
        c.drawString(60, y, line)
        y -= 24

    c.save()
    print(f"  写入 {path.name}")


# ── 主流程 ────────────────────────────────────────────────────────────────────


def main():
    print("生成测试图片 ...")
    # 纯文字 ×3
    draw_text_image(
        IMG_DIR / "text_01.png",
        [
            "OpenVINO 是 Intel 推出的开源推理工具套件，",
            "支持跨 CPU/GPU/NPU 部署。",
            "PaddleOCR-VL 是 0.9B 视觉语言模型，",
            "用于文档结构化理解。",
            "本图用于纯文字识别场景验证。",
        ],
    )
    draw_text_image(
        IMG_DIR / "text_02.png",
        [
            "Section 1: Background",
            "Document understanding requires layout-aware parsing.",
            "OpenVINO accelerates inference on Intel hardware.",
            "Multi-language support is essential for global products.",
        ],
    )
    draw_text_image(
        IMG_DIR / "text_03.png",
        [
            "## 注意事项",
            "1. 请妥善保管设备序列号；",
            "2. 工作环境温度需在规格范围内；",
            "3. 定期备份关键配置文件；",
            "4. 异常断电后请检查日志。",
        ],
    )
    # 表格 ×3
    draw_table_image(
        IMG_DIR / "table_01.png",
        title="主要参数",
        header=["型号", "功率", "温度"],
        rows=[
            ["A100", "300W", "-20~70℃"],
            ["A200", "400W", "-10~60℃"],
            ["A300", "500W", "0~50℃"],
        ],
    )
    draw_table_image(
        IMG_DIR / "table_02.png",
        title="财务对比",
        header=["季度", "营收", "利润", "增长率"],
        rows=[
            ["Q1", "120M", "20M", "+8%"],
            ["Q2", "135M", "24M", "+12%"],
            ["Q3", "150M", "30M", "+11%"],
        ],
    )
    draw_table_image(
        IMG_DIR / "table_03.png",
        title="故障代码",
        header=["代码", "含义", "处理"],
        rows=[
            ["E01", "电源异常", "检查电源"],
            ["E02", "过温保护", "等待降温"],
            ["E03", "通信失败", "重启设备"],
            ["E04", "传感器故障", "联系售后"],
        ],
    )
    # 公式 ×2
    draw_formula_image(
        IMG_DIR / "formula_01.png",
        formula="E = mc^2",
        caption="质能方程，描述质量与能量的等价关系",
    )
    draw_formula_image(
        IMG_DIR / "formula_02.png",
        formula="∫ f(x) dx = F(b) - F(a)",
        caption="牛顿-莱布尼茨公式",
    )
    # 图文混排 ×2
    draw_mix_image(
        IMG_DIR / "mix_01.png",
        title="A100 产品概览",
        paragraph=(
            "A100 是面向工业控制场景的高性能控制器，\n"
            "支持多协议通信与冗余电源。\n"
            "可用于自动化产线、楼宇控制等领域。"
        ),
        table_rows=[
            ["参数", "值"],
            ["功率", "300W"],
            ["温度", "-20~70℃"],
        ],
    )
    draw_mix_image(
        IMG_DIR / "mix_02.png",
        title="季度财报摘要",
        paragraph="本季度营收同比增长 12%，主要驱动来自 AI 业务。",
        table_rows=[
            ["指标", "本季", "同比"],
            ["营收", "135M", "+12%"],
            ["利润", "24M", "+18%"],
        ],
    )

    print("\n生成测试 PDF ...")
    make_text_pdf(PDF_DIR / "text_pdf.pdf")
    make_scanned_pdf(
        PDF_DIR / "scanned.pdf",
        source_images=[IMG_DIR / "text_01.png", IMG_DIR / "text_03.png"],
    )
    make_spec_pdf(PDF_DIR / "spec_with_tables.pdf")
    print("\n完成。")


if __name__ == "__main__":
    main()
