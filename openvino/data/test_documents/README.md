# Phase 2 测试文档清单

放入本目录的 PDF 用于验证完整解析与切片管线，共 **3 份**：

| 文档 | 类型 | 验证目标 |
|------|------|----------|
| `text_pdf.pdf` | 纯文本 PDF（pdfplumber 可直接抽出） | 走文字层路径，跳过 OCR |
| `scanned.pdf` | 扫描件（无文字层） | 走 PaddleOCR-VL OCR 路径 |
| `spec_with_tables.pdf` | 含表格规格书 | 验证表格感知切片 |

运行：

```bash
python scripts/run_phase2_pipeline.py --pdf data/test_documents/text_pdf.pdf --out results/phase2/text_pdf.chunks.jsonl
python scripts/run_phase2_pipeline.py --pdf data/test_documents/scanned.pdf --out results/phase2/scanned.chunks.jsonl
python scripts/run_phase2_pipeline.py --pdf data/test_documents/spec_with_tables.pdf --out results/phase2/spec.chunks.jsonl
```
