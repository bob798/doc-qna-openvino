# `openvino/data/` 目录

| 子目录 / 文件 | 内容 | 入库策略 |
|--------------|------|---------|
| `test_images/` | Phase 1 Benchmark 用 10 张测试图（纯文字 / 表格 / 公式 / 图文混排） | git 跟踪 |
| `test_documents/` | Phase 2 端到端 3 份 PDF（scanned / spec_with_tables / text_pdf） | git 跟踪 |
| `eval_questions.jsonl` | Phase 3 / 4 问答评测集（18+ 题，5 类） | git 跟踪 |
| `eval_questions.schema.md` | 评测集 schema | git 跟踪 |
| `eval_documents/` | Phase 3 / 4 真实评测 PDF（公开下载）| **`.gitignore` 排除 PDF**，只跟踪 `manifest.json` |
| `omnidocbench/` | OmniDocBench 标注 + 采样图像 | **`.gitignore` 排除**，体量大 |

## 自动下载

```bash
cd openvino
python scripts/download_eval_materials.py             # 默认 20 个 OmniDocBench 样本
python scripts/download_eval_materials.py --omnidoc-samples 0       # 只拉直链 PDF + OmniDocBench 标注
python scripts/download_eval_materials.py --skip omnidocbench       # 完全跳过 OmniDocBench
```

直链 PDF（无需登录）：
- PaddleOCR-VL 论文 — [arXiv 2510.14528](https://arxiv.org/abs/2510.14528)
- PaddleOCR-VL-1.5 论文 — [arXiv 2601.21957](https://arxiv.org/abs/2601.21957)
- NVIDIA H100 PCIe Product Brief — `PB-11133-001`
- NVIDIA H100 NVL Product Brief — `PB-11773-001`
- 小米手环 9 用户手册 — `xiaomi_band9_user_manual.pdf`

OmniDocBench：HF 数据集 [`opendatalab/OmniDocBench`](https://huggingface.co/datasets/opendatalab/OmniDocBench)（CVPR 2025），脚本默认只拉标注 + N 个样本，避免下满 6 GB。

## 人工下载（站点限制）

下列材料需手动放进 `eval_documents/`，文件名要与 `eval_questions.jsonl` 中 `doc_ids` 一致：

| ID / 文件名 | 来源 URL | 操作 | 用途 |
|------|---------|------|------|
| `kunlunxin_product_brief.pdf` | [PaddlePaddle XPU-P800 安装文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/hardware_support/xpu/xpu-p800_install_cn.html)（替代源）| 浏览器打开 → 打印为 PDF；或自备昆仑芯 PB | 主题贴合文心赛道；昆仑芯无公开 P800 datasheet，当前样本为 K100/K200 一代 Product Brief，功能等价 |
| `gb_t_2423_1.pdf` | [GB/T 2423.1-2008](https://openstd.samr.gov.cn/bzgk/gb/newGbInfo?hcno=4B30041DEFB4D9283C1DC9592735F67E) | 点页面"PDF"按钮（cookie 鉴权） | 扫描+文字层混合，补 scanned 路径证据 |

## 评测流程

```
download_eval_materials.py   →   eval_documents/*.pdf + omnidocbench/
                              ↓
run_phase2_pipeline.py        →   每份 PDF 切片 jsonl
                              ↓
[Phase 3.1] embed + ChromaDB  →   入库
                              ↓
[Phase 3.3] run_qa.py         →   按 eval_questions.jsonl 跑问答
                              ↓
[Phase 4]   ragas + 人工      →   faithfulness / precision / recall
       +    run_omnidocbench_subset.py + OmniDocBench 官方 eval
                              →   Edit Distance / TEDS / CDM
```
