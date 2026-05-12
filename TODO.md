# 飞桨黑客松第10期 · TODO

> 活动时间：2026-03-27 至 2026-06-05
> 活动链接：https://github.com/PaddlePaddle/Paddle/issues/78485
> GitHub ID：bob798

---

## 打卡任务 #1｜OpenVINO Notebook 快速上手

- [x] 在 issue 评论区报名（`【报名】：1`）
- [x] macOS 安装 OpenVINO Notebooks
- [x] 运行 `paddleocr_vl` notebook 并记录结果
- [x] 发送打卡邮件
  - **收件人**（To）：`ext_paddle_oss@baidu.com`
  - **抄送**（Cc）：`zhuo.wu@intel.com`、`ethan.yang@intel.com`
  - **主题**：`文心伙伴赛道-intel-打卡-bob798`
  - **正文模板**：
    ```
    飞桨团队你好，

    【GitHub ID】：bob798（仓库地址）
    【运行 Notebook】：PaddleOCR-VL Notebook
      https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/paddleocr_vl
    【环境信息】：macOS / <CPU> / <GPU> / OpenVINO <版本>
    【打卡截图】：见附件 / 链接
    ```
- [x] 审核通过

---

## 进阶任务 #13｜基于 OpenVINO 的多模态文档理解与智能应用开发

> 需打卡任务 #1 审核通过后方可认领
> 奖励：¥2000/人 × 2 名

### 报名提交

- [x] 打卡审核通过后，评论区报名（`【报名】：13`）
- [x] 整理项目方案（见 `docs/进阶方案.md`）
- [x] 准备简历（已提交）
- [x] 发送进阶方案邮件（方案已确认通过）

### 开发阶段 ← 当前阶段

> 详细执行手册见 [`docs/开发手册.md`](docs/开发手册.md)
> 截止日期：2026-06-05

- [x] **Phase 1 (Week 1)**: 环境搭建 + 模型验证 + 推理 Benchmark — 代码完成，待跑通实测数据（见 `openvino/scripts/benchmark_inference.py` 和 `benchmark_ocr_quality.py`）
- [x] **Phase 2 (Week 2)**: 文档解析 + 表格感知切片模块 — 代码完成（见 `openvino/src/{pdf_preprocessor,doc_parser,chunker,pipeline}.py`）
- [ ] **Phase 3 (Week 3)**: RAG 问答链路（Embedding + ChromaDB + LLM）
- [ ] **Phase 4 (Week 4)**: Tesseract vs PaddleOCR-VL 对比评测 + 优化
- [ ] **Phase 5 (Week 5)**: 整理 Notebook + README + requirements + 提交

### Phase 3 启动前的收尾（今晚）

> 来自 W2 周报 §3 "风险与待办（非阻塞）" + 5/15 截止事项。所有项都不阻塞 Phase 3.1，可平行做。

- [ ] **W2 周报定稿提交** PFCCLab/Camp PR（截止 2026-05-15）—— `docs/周报/W2_2026-05-15.md`
- [ ] **`docs/进阶方案.md` 占位话术替换**：用 Phase 1 实测数据替换 "≥ 2×"、"显著优于传统 OCR"（数据源 `openvino/results/phase1/benchmark_inference.md`、`quality_compare.md`）
- [ ] **PyTorch baseline 补跑**：验 transformers 4.54 + trust_remote_code 兼容性，补 Benchmark 1 加速比对照
- [ ] **修 pdfplumber 表格 section 元数据**：按 bbox 排序合并，避免表格 chunk `section_title` 跟随页末标题（Phase 4 评测前必须）
- [ ] **`scanned.pdf` 漏识别复测**：等真实业务样本到位后跑 OCR 路径（可推到 Phase 4，与上一项一起做）

### 周报

> 在 [PFCCLab/Camp](https://github.com/PFCCLab/Camp/pull/584) 提交周报
> 目录：`WeeklyReports/Hackathon_10th/ERNIEPartner/`

- [x] Week 1 周报（已提交 [PFCCLab/Camp #598](https://github.com/PFCCLab/Camp/pull/598)）
- [ ] Week 2 周报
- [ ] Week 3 周报
- [ ] Week 4 周报
- [ ] Week 5 周报

---

## 关联仓库

> 本仓库 `doc-qna-openvino` 仅承载 OpenVINO 路径（进阶任务 #13）。其他赛题 / 提交材料已拆分到独立仓库：

| 内容 | 仓库 |
|------|------|
| 高通赛题（基于 QNN 部署 PaddleOCR-VL） | https://github.com/bob798/paddleocr-vl-qnn |
| 进阶任务 #26 小伴（提交材料） | https://github.com/bob798/xiaoban（`submission/` 子目录）|

---

## 文件索引

| 文件 | 用途 |
|------|------|
| `docs/进阶方案.md` | 已提交的进阶任务方案（OpenVINO） |
| `docs/开发手册.md` | Phase 1-5 执行手册（按周推进） |
| `docs/项目计划.md` | 对齐 6/5 截止日期的项目开发计划 |
| `docs/技术介绍.md` | 技术科普（OpenVINO / PaddleOCR-VL） |
| `docs/差异化分析.md` | 与现有 notebooks 的差异论证 |
| `docs/打卡记录.md` | 打卡任务 #1 提交记录 |
| `docs/周报/` | 每周周报存档 |
| `assets/` | 截图（打卡运行结果） |
| `openvino/README.md` | Phase 1+2 运行指南 |
| `openvino/scripts/benchmark_inference.py` | PyTorch vs OpenVINO 速度 Benchmark |
| `openvino/scripts/benchmark_ocr_quality.py` | Tesseract vs PaddleOCR-VL 质量对比 |
| `openvino/scripts/run_phase2_pipeline.py` | PDF → 解析 → 表格感知切片 CLI |
| `openvino/src/` | 共享推理封装 + 解析 + 切片模块 |
