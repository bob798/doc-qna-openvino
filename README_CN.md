# doc-qna-openvino（中文文档）

> [English README](./README.md)

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
[ Embedding + 检索 ]  Qwen3-Embedding-0.6B-int8 (OpenVINO) → ChromaDB
   │
   ▼
[ LLM 生成 ]        Qwen3 INT4 (OpenVINO GenAI) → 带来源引用的回答
```

---

## 目录结构

```
doc-qna-openvino/
├── README.md                  英文文档（主入口）
├── README_CN.md               本文档
├── TODO.md                    任务清单 + 文件索引
├── docs/                      项目文档
│   ├── 进阶方案.md            已提交并通过的 RFC 方案
│   ├── 项目计划.md            按 Phase 1~5 拆周推进
│   ├── 开发手册.md            每个 Phase 的目标 / 步骤 / 验证标准
│   ├── 技术介绍.md            技术科普（OpenVINO / PaddleOCR-VL）
│   ├── 差异化分析.md          与现有 notebooks 的差异论证
│   ├── 打卡记录.md            打卡任务 #1 提交记录
│   └── 周报/                  每周周报存档
├── openvino/                  代码主体（详见子目录 README）
│   ├── README.md              Phase 1~3 运行指南
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

# Windows 用户需先设置环境变量（详见 openvino/README.md § 一次性准备）
# $env:PYTHONIOENCODING = "utf-8"
# $env:HF_HUB_DISABLE_SYMLINKS = "1"

# 1. 准备 PaddleOCR-VL OpenVINO IR（参考 openvino/README.md § 准备工作）
# 2. Phase 1 — 推理 Benchmark
python scripts/benchmark_inference.py --image_dir data/test_images --device CPU
# 3. Phase 2 — 文档解析 + 表格感知切片
python scripts/run_phase2_pipeline.py --pdf_dir data/test_documents --out results/phase2
# 4. Phase 3 — RAG 端到端问答
python scripts/build_index.py --chunks_dir results/phase2 --persist_dir chroma_db --device CPU --reset
python scripts/run_qa.py --questions_file data/demo_questions.txt --persist_dir chroma_db --out results/phase3/qa_run.json
```

详细说明见 [`openvino/README.md`](openvino/README.md)。

---

## 进度

| Phase | 内容 | 状态 |
|-------|------|------|
| 1 | 环境搭建 + 模型验证 + 推理 Benchmark | ✅ OV CPU/GPU 实测落地（GPU/CPU 1.47×） |
| 2 | 文档解析 + 表格感知切片模块 | ✅ 3 处切片器漏洞修复 + GB/T 2423.1 真实国标 14 页 87 chunks |
| 3 | RAG 问答链路（Embedding + ChromaDB + LLM） | ✅ 端到端跑通，5 题 CPU ~3.3s/题，带 [doc p.页] 引用 |
| 4 | Tesseract vs PaddleOCR-VL 对比评测 | ✅ 3 页代表性对比 + 5 题 eval 集业务验证 |
| 5 | 整理 README + 演示视频 + 提交 PR | 🔄 README 终稿 + 演示视频 + PFCCLab PR |

完整时间表见 [`docs/项目计划.md`](docs/项目计划.md)。

---

## 周报

提交至 [PFCCLab/Camp](https://github.com/PFCCLab/Camp)，路径 `WeeklyReports/Hackathon_10th/ERNIEPartner/ERNIEPartner_13_bob798/`。

| 周次 | 时间窗口 | PR | 本地存档 |
|------|----------|----|----------|
| W1 | 2026.04.25 ~ 2026.05.08 | [#598](https://github.com/PFCCLab/Camp/pull/598) | [`docs/周报/W1_2026-05-08.md`](docs/周报/W1_2026-05-08.md) |
| W2 | 2026.05.09 ~ 2026.05.22 | [#609](https://github.com/PFCCLab/Camp/pull/609) | [`docs/周报/W2_2026-05-22.md`](docs/周报/W2_2026-05-22.md) |
| W3 | 2026.05.23 ~ 2026.06.05 | [#617](https://github.com/PFCCLab/Camp/pull/617) | [`docs/周报/W3_2026-06-05.md`](docs/周报/W3_2026-06-05.md) |

---

## 提交

- OpenVINO demo 仓库 PR：[openvinotoolkit/openvino_build_deploy#552](https://github.com/openvinotoolkit/openvino_build_deploy/pull/552)
- 任务完成提交邮件：[`docs/任务完成提交.email.md`](docs/任务完成提交.email.md)

---

## 链接

- 赛道总览 Issue：https://github.com/PaddlePaddle/Paddle/issues/78485
- PaddleOCR-VL：https://huggingface.co/PaddlePaddle/PaddleOCR-VL
- OpenVINO Notebook（PaddleOCR-VL）：https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/paddleocr_vl
