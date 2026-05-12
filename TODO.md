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

### 今晚 Windows 执行清单（按顺序）

> 5/15 是 W2 周报硬截止。`git pull` 拉今天 3 个 commit 后按下方顺序跑。

**0. 同步代码 + 装依赖**（≈ 3 min）

```powershell
git pull
python -m pip install --user -r openvino\requirements.txt
```

**1. 验证 pdfplumber bbox 修复**（≈ 1 min · 代码已修，跑一遍冒烟）

```powershell
cd openvino
python scripts\run_phase2_pipeline.py --pdf data\test_documents\spec_with_tables.pdf --out results\phase2
# 期望：spec_with_tables.chunks.jsonl 里表格行的 section_title=主要参数（不再是页末"故障代码"）
```

**2. 拉新评测材料 + OmniDocBench 子集**（≈ 5 min · 网络）

```powershell
python scripts\download_eval_materials.py --omnidoc-samples 20
# 期望：data\eval_documents\ 下 5 PDF + manifest.json，data\omnidocbench\ 下标注 + 20 张图
```

**3. PyTorch baseline 补跑**（≈ 30~60 min · CPU 慢）

```powershell
python scripts\benchmark_inference.py --warmup 1 --runs 3
# 期望：results\phase1\benchmark_inference.md 含 PyTorch 列 + 加速比；
# 若 PyTorch 加载失败 → 看 stderr，必要时降 transformers 版本或加 trust_remote_code 调试
```

**4. 真实扫描件复测**（≈ 5 min · 接替原 `scanned.pdf`）

```powershell
python scripts\run_phase2_pipeline.py --pdf data\eval_documents\gb_t_2423_1.pdf --out results\phase2
# 期望：OCR 路径吐出 ≥ 10 个 chunk（W1 合成图只有 1 个），证明真实样本下 PaddleOCR-VL 能拿到结构
```

**5. `docs/进阶方案.md` §五 占位话术替换**（≈ 10 min · 编辑）

把 §五 性能指标表改成"预期 → 实测"两列对照，填入：
- PaddleOCR-VL 加速比 → #3 跑完后的真实数字
- 扫描件识别 → #4 真实样本结果（替换 "≥ 95%" 占位）
- LLM 延迟 / 端到端响应 → 标记 "Phase 3 后填" 保留

**6. W2 周报定稿提交**（≈ 15 min）

周报草稿已加入 §1.4 Qwen3 升级 / §1.5 评测体系定稿 / §1.6 pdfplumber 修复。
拿 #3 #4 的最新数据回填 §1.1 表格底部"PyTorch 对照"行后：

```powershell
# Fork PFCCLab/Camp，复制周报到 WeeklyReports/Hackathon_10th/ERNIEPartner/ERNIEPartner_13_bob798/
gh repo fork PFCCLab/Camp --clone --remote
cd Camp
# 把 docs\周报\W2_2026-05-15.md 复制到目标目录
# 提交 PR
```

完成后回来勾这 6 项 + 上面"评测体系准备"的 [ ] 项。

### 评测体系准备（Phase 3 → 4 过渡）

- [x] 评测体系三层架构定稿（见 [`docs/进阶方案.md` §七](docs/进阶方案.md)）
- [x] `eval_questions.jsonl` 骨架（18 题，5 类）+ schema 文档
- [x] `scripts/download_eval_materials.py` 自动下载脚本（5 份直链 + OmniDocBench 子集）
- [x] `scripts/run_omnidocbench_subset.py` OmniDocBench 子集对比骨架
- [x] **人工下载 2 份中文材料**（脚本检测到位后自动打勾，见 `manifest.json` 的 `manual[].present`）
  - [x] 昆仑芯 Product Brief → `K100_K200_spec.pdf` (204 KB)（实际为 K100/K200 一代 PB；官方无公开 P800 datasheet，功能等价用于评测）
  - [x] [GB/T 2423.1-2008](https://openstd.samr.gov.cn/bzgk/gb/newGbInfo?hcno=4B30041DEFB4D9283C1DC9592735F67E) → `gb_t_2423_1.pdf` (834 KB)
- [ ] `eval_questions.jsonl` 扩到 30~50 题（Phase 3.3 跑通端到端后再扩，避免无效题）
- [ ] 接入 RAGAS（`pip install ragas`，Phase 3.3 端到端跑通后）

### 周报

> 在 [PFCCLab/Camp](https://github.com/PFCCLab/Camp/pull/584) 提交周报
> 目录：`WeeklyReports/Hackathon_10th/ERNIEPartner/`

- [x] Week 1 周报（已提交 [PFCCLab/Camp #598](https://github.com/PFCCLab/Camp/pull/598)）
- [ ] Week 2 周报
- [ ] Week 3 周报
- [ ] Week 4 周报
- [ ] Week 5 周报

---

## 文件索引

### 文档

| 文件 | 用途 |
|------|------|
| `docs/进阶方案.md` | 已提交的进阶任务方案 + §七 评测体系 |
| `docs/开发手册.md` | Phase 1-5 执行手册（按周推进） |
| `docs/项目计划.md` | 对齐 6/5 截止日期的项目开发计划 |
| `docs/技术介绍.md` | 技术科普（OpenVINO / PaddleOCR-VL） |
| `docs/差异化分析.md` | 与现有 notebooks 的差异论证 |
| `docs/打卡记录.md` | 打卡任务 #1 提交记录 |
| `docs/周报/` | 每周周报存档 |
| `assets/` | 截图（打卡运行结果） |

### 代码

| 文件 | 用途 |
|------|------|
| `openvino/README.md` | Phase 1+2 运行指南 |
| `openvino/src/` | 推理封装 + PDF 预处理 + 解析 + 表格感知切片 |
| `openvino/scripts/benchmark_inference.py` | Phase 1 · PyTorch vs OpenVINO 速度 Benchmark |
| `openvino/scripts/benchmark_ocr_quality.py` | Phase 1 · Tesseract vs PaddleOCR-VL 质量对比 |
| `openvino/scripts/run_phase2_pipeline.py` | Phase 2 · PDF → 解析 → 表格感知切片 CLI |
| `openvino/scripts/download_eval_materials.py` | Phase 3/4 · 评测材料自动下载（PDF + OmniDocBench） |
| `openvino/scripts/run_omnidocbench_subset.py` | Phase 4 · OmniDocBench 子集对比骨架 |

### 数据

| 路径 | 用途 |
|------|------|
| `openvino/data/test_images/` | Phase 1 测试图 10 张（git 跟踪） |
| `openvino/data/test_documents/` | Phase 2 测试 PDF 3 份（git 跟踪） |
| `openvino/data/eval_questions.jsonl` | Phase 3/4 评测题集（git 跟踪） |
| `openvino/data/eval_questions.schema.md` | 评测题集 schema |
| `openvino/data/eval_documents/` | Phase 3/4 真实评测 PDF（`.gitignore`，自动下载） |
| `openvino/data/omnidocbench/` | OmniDocBench 标注 + 采样图（`.gitignore`） |
