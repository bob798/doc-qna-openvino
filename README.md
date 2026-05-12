# doc-qna-openvino

> 飞桨黑客松第 10 期 · 文心合作伙伴赛道
> 进阶任务 #13：基于 OpenVINO 的多模态文档理解与智能应用开发
> GitHub ID：[bob798](https://github.com/bob798) · 截止日期：2026-06-05

基于 **PaddleOCR-VL + OpenVINO** 的产品文档智能入库与客服问答系统。

---

## 是什么

把产品手册/规格书/扫描件这类异构文档，端到端转成可被自然语言提问的知识库：

```
PDF / 图片
   │
   ▼
[ PDF 预处理 ]      pdfplumber 文字层判断 + pypdfium2 渲染
   │
   ▼
[ PaddleOCR-VL ]    OpenVINO 加速推理 → 结构化 Markdown
   │
   ▼
[ 表格感知切片 ]    表头 + 行 chunk · 段落 200~500 字符 · 元数据 {doc, page, section}
   │
   ▼
[ Embedding + 检索 ]  BGE-small (OpenVINO) → ChromaDB
   │
   ▼
[ LLM 生成 ]        Qwen3 INT4 (OpenVINO GenAI) → 带来源引用的回答
```

---

## 目录结构

```
doc-qna-openvino/
├── README.md                  本文档
├── TODO.md                    任务清单 + 关联仓库索引
├── docs/                      项目文档
│   ├── 进阶方案.md            已提交并通过的 RFC 方案
│   ├── 项目计划.md            按 Phase 1~5 拆周推进
│   ├── 开发手册.md            每个 Phase 的目标 / 步骤 / 验证标准
│   ├── 技术介绍.md            技术科普（OpenVINO / PaddleOCR-VL）
│   ├── 差异化分析.md          与现有 notebooks 的差异论证
│   ├── 打卡记录.md            打卡任务 #1 提交记录
│   └── 周报/                  每周周报存档
├── openvino/                  代码主体（详见子目录 README）
│   ├── README.md              Phase 1+2 运行指南
│   ├── requirements.txt
│   ├── configs/models.json
│   ├── src/                   推理封装 / PDF 预处理 / 解析 / 切片 / 管线
│   ├── scripts/               Benchmark + 端到端 CLI
│   ├── data/                  10 张测试图 + 3 份测试 PDF
│   └── results/               脚本运行产出
└── assets/                    截图等静态资源
```

---

## 快速开始

```bash
git clone https://github.com/bob798/doc-qna-openvino
cd doc-qna-openvino/openvino
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. 准备 PaddleOCR-VL OpenVINO IR（参考 openvino/README.md）
# 2. 跑 Phase 1 Benchmark
python scripts/benchmark_inference.py
python scripts/benchmark_ocr_quality.py
# 3. 跑 Phase 2 端到端管线
python scripts/run_phase2_pipeline.py --pdf_dir data/test_documents --out results/phase2
```

详细说明见 [`openvino/README.md`](openvino/README.md)。

---

## 进度

| Phase | 内容 | 状态 |
|-------|------|------|
| 1 | 环境搭建 + 模型验证 + 推理 Benchmark | 🔄 代码完成，待 IR 落地后跑实测数据 |
| 2 | 文档解析 + 表格感知切片模块 | 🔄 代码完成，本地烟囱测试通过 |
| 3 | RAG 问答链路（Embedding + ChromaDB + LLM） | ⬜ 未开始 |
| 4 | Tesseract vs PaddleOCR-VL 对比评测 + 优化 | ⬜ 未开始 |
| 5 | 整理 Notebook + README + requirements + 提交 | ⬜ 未开始 |

完整时间表见 [`docs/项目计划.md`](docs/项目计划.md)。

---

## 关联仓库

> 本仓库 `doc-qna-openvino` 仅承载 OpenVINO 路径（进阶任务 #13）。其他赛题 / 提交材料拆分到独立仓库：

| 内容 | 仓库 |
|------|------|
| 高通赛题（基于 QNN 部署 PaddleOCR-VL） | [`bob798/paddleocr-vl-qnn`](https://github.com/bob798/paddleocr-vl-qnn) |
| 进阶任务 #26 小伴（提交材料） | [`bob798/xiaoban`](https://github.com/bob798/xiaoban)（`submission/` 子目录）|

---

## 周报

提交至 [PFCCLab/Camp](https://github.com/PFCCLab/Camp)，路径 `WeeklyReports/Hackathon_10th/ERNIEPartner/ERNIEPartner_13_bob798/`。

| 周次 | 时间窗口 | PR | 本地存档 |
|------|----------|----|----------|
| W1 | 2026.04.25 ~ 2026.05.08 | [#598](https://github.com/PFCCLab/Camp/pull/598) | [`docs/周报/W1_2026-05-08.md`](docs/周报/W1_2026-05-08.md) |
| W2 | 2026.05.09 ~ 2026.05.15 | _草稿_ | [`docs/周报/W2_2026-05-15.md`](docs/周报/W2_2026-05-15.md) |

---

## 链接

- 赛道总览 Issue：https://github.com/PaddlePaddle/Paddle/issues/78485
- PaddleOCR-VL：https://huggingface.co/PaddlePaddle/PaddleOCR-VL
- OpenVINO Notebook（PaddleOCR-VL）：https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/paddleocr_vl
