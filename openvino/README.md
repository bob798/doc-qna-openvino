# 进阶任务 #13 · OpenVINO 子项目代码

> 飞桨黑客松第 10 期 · 文心伙伴赛道
> 项目：基于 PaddleOCR-VL + OpenVINO 的产品文档智能入库与客服问答系统

本目录是 OpenVINO 路径的全部代码和实验产物，**Phase 1（推理 Benchmark）** 与 **Phase 2（文档解析 + 切片）** 已完成代码实现，运行所需的模型 / 测试数据由开发者本机准备。

```
openvino/
├── README.md                       本文档
├── requirements.txt                依赖
├── configs/models.json             模型/路径配置
├── data/
│   ├── test_images/README.md       Phase 1 的 10 张测试图片说明
│   └── test_documents/README.md    Phase 2 的 3 份测试 PDF 说明
├── src/
│   ├── inference.py                OpenVINO / PyTorch / Tesseract 推理封装
│   ├── pdf_preprocessor.py         pdfplumber 可读性判断 + pdf2image 渲染
│   ├── doc_parser.py               PaddleOCR-VL 全图解析 → 结构化 Markdown
│   ├── chunker.py                  表格感知切片 + 元数据
│   └── pipeline.py                 Phase 2 端到端管线
├── scripts/
│   ├── benchmark_inference.py      Phase 1 · PyTorch vs OpenVINO 速度
│   ├── benchmark_ocr_quality.py    Phase 1 · Tesseract vs PaddleOCR-VL 质量
│   └── run_phase2_pipeline.py      Phase 2 · CLI 入口
└── results/
    ├── phase1/                     Benchmark 输出（脚本运行后自动生成）
    └── phase2/                     切片输出（脚本运行后自动生成）
```

---

## 准备工作

### 1. 安装依赖

```bash
cd openvino
python -m venv .venv && source .venv/bin/activate    # 或 conda
pip install -r requirements.txt
# Tesseract 需另装二进制（macOS：`brew install tesseract tesseract-lang`）
```

### 2. 准备 PaddleOCR-VL 的 OpenVINO IR

参考 [openvino_notebooks/paddleocr_vl](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/paddleocr_vl)
的官方流程将模型下载并转为 IR，输出到：

```
openvino/models/paddleocr_vl_ov/
├── openvino_model.xml
├── openvino_model.bin
├── tokenizer.xml
├── tokenizer.bin
└── ...
```

可在 `configs/models.json` 中调整 IR 目录与设备。

### 3. 准备测试数据

- `data/test_images/`：按 [README](data/test_images/README.md) 放入 10 张图片
- `data/test_documents/`：按 [README](data/test_documents/README.md) 放入 3 份 PDF

---

## Phase 1 · 推理 Benchmark

### Benchmark 1：PyTorch vs OpenVINO 速度

```bash
python scripts/benchmark_inference.py \
    --image_dir data/test_images \
    --ir_dir ./models/paddleocr_vl_ov \
    --device CPU \
    --runs 3
```

输出：

- `results/phase1/benchmark_inference.md` —— Markdown 表格（含每张图 mean/min/max 与加速比）
- `results/phase1/benchmark_inference.json` —— 原始数据

> 用该 markdown 中的实测均值替换 `docs/进阶方案.md` 里的 "≥ 2×" 预估。

如果当前环境装不了 PyTorch（如 macOS x86_64 PyTorch 版本受限），加 `--no_pytorch` 仅跑 OpenVINO，并在最终交付物中说明环境差异。

### Benchmark 2：Tesseract vs PaddleOCR-VL 解析质量

```bash
python scripts/benchmark_ocr_quality.py \
    --image_dir data/test_images \
    --ir_dir ./models/paddleocr_vl_ov \
    --tesseract_lang chi_sim+eng
```

输出：

- `results/phase1/quality_compare.md` —— 并列对比 + 末尾人工评分表（手动填写）
- `results/phase1/quality_compare/<img>/` —— 每张图的 `tesseract.txt`、`paddleocr_vl.md`、`diff.txt`、原图副本
- `results/phase1/quality_compare.json` —— 结构化数据

> 评分填好后，把 `quality_compare.md` 末尾评分表的截图存入 `assets/`，作为 Notebook 演示证据。

---

## Phase 2 · 文档解析 + 切片

### 单文档

```bash
python scripts/run_phase2_pipeline.py \
    --pdf data/test_documents/spec_with_tables.pdf \
    --out results/phase2 \
    --ir_dir ./models/paddleocr_vl_ov \
    --dpi 200
```

输出（每份 PDF）：

- `<name>.markdown` —— 全文 Markdown（保留页码注释）
- `<name>.chunks.jsonl` —— 表格感知切片，每行 `{ "text": "...", "metadata": {...} }`
- `<name>.summary.json` —— 解析时长 + 切片统计

### 批量

```bash
python scripts/run_phase2_pipeline.py --pdf_dir data/test_documents --out results/phase2
```

### 验证清单（来自 docs/开发手册.md）

- [ ] 三类 PDF 均能跑通：`text_pdf` / `scanned` / `spec_with_tables`
- [ ] 检查 `text_pdf` 是否走了 text-layer 路径（看 summary 中 `num_ocr_pages`）
- [ ] 检查 `spec_with_tables` 的 chunks 是否每个表格行都附带表头
- [ ] 检查 chunks 元数据：`{doc_name, page, section_title, kind}` 是否完整

### 切片规则简述

| Chunk 类型 | 触发 | 内容 | 元数据 |
|------------|------|------|--------|
| `text` | 普通段落 / 标题 | 200-500 字符语义段，超长按 50 字符 overlap 切分 | `doc_name, page, section_title, kind=text` |
| `table_header` | Markdown 表格命中 | 表头 + 分隔符 | `+ row_count` |
| `table` | 每个表格行 | 表头 + 分隔符 + 单行 | `+ row_index` |

表格行 chunk 携带表头是为了让 Embedding 阶段保留列上下文（解决"300W 是哪一列"的歧义）。

---

## 后续 Phase（W3+）

- **W3 (Phase 3)**：BGE-small embedding + ChromaDB 入库 + Qwen3-1.7B INT4 问答
- **W4 (Phase 4)**：Tesseract 全链路 vs PaddleOCR-VL 全链路对比评测
- **W5 (Phase 5)**：Notebook 整理 + 提交

详见 [`../docs/项目计划.md`](../docs/项目计划.md) 与 [`../docs/开发手册.md`](../docs/开发手册.md)。
