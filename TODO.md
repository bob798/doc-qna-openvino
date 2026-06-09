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
> **2026-05-22 导师反馈方向校准**：5 周周期下停止扩展评测/功能，立刻收敛到端到端 Demo。详情见 [`docs/决策记录/2026-05-22_导师反馈方向校准.md`](docs/决策记录/2026-05-22_导师反馈方向校准.md)。

- [x] **Phase 1 (Week 1)**: 环境搭建 + 模型验证 + 推理 Benchmark — 实测数据落地（OV CPU/GPU 1.47×，PT baseline 弃测有书面说明）
- [x] **Phase 2 (Week 2)**: 文档解析 + 表格感知切片模块 — 切片器 3 处真实漏洞修复 + GB/T 2423.1 14 页实测 87 chunks
- [x] **Phase 3 (Week 3)**: RAG 端到端最小闭环 + 业务问题精选 + OCR 路径选型对比 — 全部已落地：Embedder (Qwen3-Embedding-0.6B-int8) + ChromaDB + Qwen3-1.7B-int4 LLM 跑通，Demo 集 5 题 ~3.3s/题，eval 集 5 题（H100 PB）+ Tesseract vs Paddle 3 页对比，证据完整
- [ ] **Phase 4 (Week 4 ← 现在)**: 终交付物 — 演示视频 + README 终稿 + PR 提交（详见下方"W4 计划"）
- [ ] **Phase 5 (Week 5)**: 兜底周（W4 完成后用于 review 反馈 / 补救；若 W4 直接交付到位则无需 W5）

### W3 已完成（2026-05-23 ~ 2026-06-05）— 主线 + 必做评测全部交付

> 目标已达成：可复现的端到端 Demo（运行命令 + 样例问题 + 检索结果 + 带 (doc,page) 引用的回答） + Tesseract vs PaddleOCR-VL 选型证据。
> W3 周报草稿已写（`docs/周报/W3_2026-06-05.md`），待开 PR + 发邮件——见下方"周报"段。

#### W3 主线（必做 P0）

- [x] **P0-1 Embedding + ChromaDB**：用 `OpenVINO/Qwen3-Embedding-0.6B-int8-ov`（官方预转 INT8 IR，多语言含中文，替代无预转 IR 的 BGE-small-zh）+ ChromaDB Persistent 灌入 Phase 2 的 99 chunks，encode 138.6 ms/chunk，cos Top-K 召回相关性人工验证通过（见 `openvino/results/phase3/qa_run.md` Top-K 命中段）
- [x] **P0-2 LLM 生成**：`openvino_genai.LLMPipeline` 加载 `OpenVINO/Qwen3-1.7B-int4-ov`，RAG prompt 模板设计完成 + 输出带 `[doc_name p.页码]` 引用，`enable_thinking=False` + `/no_think` 双保险 + `<think>` 后处理剥离
- [x] **P0-3 端到端 `scripts/run_qa.py`**：跑通 5 条业务问题，3 条事实回答正确 + 2 条防幻觉拒答（1 条 retrieval miss，1 条保守拒答），平均 ~3.3s/题（CPU），耗时分解：embed 69ms / retrieve 1.7ms / LLM 3270ms
- [x] **P0-4 README/复现物料**：`openvino/README.md` 新增 Phase 3 章节（运行命令 + 输出示例表 + 性能数据 + 已知限制）
- [x] **P0-5 NPU 限制说明**：`openvino/README.md` 写了独立"Phase 3 NPU 限制说明"段，明确主线走 CPU/GPU/AUTO 的三条理由，并交叉引用 2026-05-22 决策记录

#### W3 必做评测（瘦身版 P1）

- [x] **P1-1 业务问题集精简**：从 `eval_questions.jsonl` 18 题挑 5 题（q008/q009/q011/q013/q015，覆盖 table_lookup×3 + cross_doc×1 + refusal×1），在 2 份 NVIDIA H100 PB（46 页 / 372 chunks）上端到端跑：1/5 业务 hit + 2/5 正确拒答 + 1/5 方向对 + 1/5 幻觉。失败案例集中暴露 small-embedder + last-token-pool 在表格行 chunk 上的系统性偏差（详见 `openvino/README.md § P1-1`），W4 加分项 BGE-reranker 设计为对此打补丁。本期顺手把 chunker 表格 caption 增强 + `--min_chars 60` 短 chunk 噪声过滤两处工程改进溢出
- [x] **P1-2 Tesseract vs PaddleOCR-VL 少量对比**：3 张代表性页面（GB/T 2423.1 p.5 公式 / p.7 纯文字 / p.14 复杂表格）对比完成。决定性结果：表格页 char 相似度仅 9.55%（Tesseract 完全丢失 rowspan/colspan 结构），公式页 Tesseract 0 个 LaTeX 标记 vs Paddle 12 个；速度 Tesseract 0.8-2.8s vs Paddle 11-27s（10-30× 快但结构丢失），主线选 Paddle 理由已被证据支持，详见 `openvino/README.md § 5 P1-2`

### W4 计划（2026-05-30 ~ 2026-06-05）— 最后冲刺，6/5 截止 ← 现在

> 主线优先级：**周报 PR + 邮件 → 演示视频 → README 终稿 → 提交 OpenVINO demo PR + 任务完成邮件**。
> 加分项时间允许再做，不阻塞 P0。

**P0 — 必做交付物（按建议执行顺序）**

- [x] **W3 周报 PR + 邮件**（≤ 0.5 day，本周内）
  - [x] 在 [PFCCLab/Camp](https://github.com/PFCCLab/Camp) 开 W3 PR，把 `docs/周报/W3_2026-06-05.md` 内容贴到 `WeeklyReports/Hackathon_10th/ERNIEPartner/W3_2026-06-05.md`（[PFCCLab/Camp#617](https://github.com/PFCCLab/Camp/pull/617)）
  - [ ] 发 W3 双周报邮件：按 `docs/周报/W3_2026-06-05.email.md` 模板，主题日期改成实际发送日 `0521`，正文里 PR 链接替换为 https://github.com/PFCCLab/Camp/pull/617
  - [x] 回头把"周报"段的 W3 状态打勾，附上 PR 编号
- [ ] **演示视频/录屏**（1 day）：terminal 跑通 `build_index.py` + `run_qa.py` 端到端 + 回答带 `[doc p.页]` 引用 + 性能输出截图，2~3 分钟（OBS / Windows 内置录屏均可）
- [x] **README 终稿**（0.5 day）：核对 `openvino/README.md`——Windows 环境变量段、IR 下载量（Qwen3-1.7B-int4 ≈ 1 GB / Qwen3-Embedding-0.6B-int8 ≈ 600 MB / PaddleOCR-VL-1.5 ≈ 2.7 GB）、磁盘需求 8 GB、模型选型决策（BGE → Qwen3-Embedding）；顶层 README.md 同步更新进度表和周报表
- [x] **提交 PR 到 OpenVINO demo 仓库**：[openvinotoolkit/openvino_build_deploy#552](https://github.com/openvinotoolkit/openvino_build_deploy/pull/552)
- [ ] **发送任务完成邮件**：草稿见 [`docs/任务完成提交.email.md`](docs/任务完成提交.email.md)，To `ext_paddle_oss@baidu.com` + Cc Intel 双邮箱

**P2 加分项（时间不够直接砍，按 ROI 排序）**

- [ ] **BGE-reranker-base 重排**（最高 ROI）——直接补 P1-1 暴露的表格行 chunk 召回问题：用 `BAAI/bge-reranker-base` cross-encoder 在 Top-30 候选上重新打分，把 `Memory size | 94 GB` 这种行 chunk 顶到 Top-5
- [ ] **NPU 进取路径**——把 Qwen3 LLM 或 Embedding 切到 NPU（NPU 明确支持的 workload），讲"OCR-GPU / LLM-NPU / 检索-CPU"AI PC 全家桶叙事

### 加分项 — P3（不阻塞主流程，时间盒严控）

> 以下项 6/2 前 W4 P0 没稳就完全砍掉，**主 Demo + PR > 一切**。

- [ ] OmniDocBench 子集 20 张图跑 Edit Distance / TEDS / CDM
- [ ] 接入 RAGAS（faithfulness / answer_relevancy / context_precision / context_recall）
- [ ] `eval_questions.jsonl` 扩到 30~50 题
- [ ] Gradio 演示 UI

### 评测体系准备（已就位 — 转入加分项后再用）

> 三层架构已设计 + 材料已就位，**但 W3 不主跑**。主 Demo 通后看时间从加分项里拣。

- [x] 评测体系三层架构定稿（见 [`docs/进阶方案.md` §七](docs/进阶方案.md)）
- [x] `eval_questions.jsonl` 骨架（18 题，5 类）+ schema 文档
- [x] `scripts/download_eval_materials.py` 自动下载脚本（5 份直链 + OmniDocBench 子集）
- [x] `scripts/run_omnidocbench_subset.py` OmniDocBench 子集对比骨架
- [x] **人工下载 2 份中文材料**（脚本检测到位后自动打勾，见 `manifest.json` 的 `manual[].present`）
  - [x] 昆仑芯 Product Brief → `K100_K200_spec.pdf` (204 KB)（实际为 K100/K200 一代 PB；官方无公开 P800 datasheet，功能等价用于评测）
  - [x] [GB/T 2423.1-2008](https://openstd.samr.gov.cn/bzgk/gb/newGbInfo?hcno=4B30041DEFB4D9283C1DC9592735F67E) → `gb_t_2423_1.pdf` (834 KB)

### 周报

> 在 [PFCCLab/Camp](https://github.com/PFCCLab/Camp/pull/584) 提交周报
> 目录：`WeeklyReports/Hackathon_10th/ERNIEPartner/`

- [x] Week 1 周报（已提交 [PFCCLab/Camp #598](https://github.com/PFCCLab/Camp/pull/598)）
- [x] Week 2 周报（已提交 [PFCCLab/Camp #609](https://github.com/PFCCLab/Camp/pull/609)）
- [x] Week 3 周报（已提交 [PFCCLab/Camp#617](https://github.com/PFCCLab/Camp/pull/617)）

> 周报是按双周（W1 5/8 / W2 5/22 / W3 6/5）提交，本期 W3 为收官周报。若 W4 主线 PR 提交后官方仍要求继续周报，再补 Week 4 / Week 5。

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
| `openvino/scripts/build_index.py` | Phase 3 · chunks.jsonl → ChromaDB 入库 |
| `openvino/scripts/run_qa.py` | Phase 3 · 端到端 RAG 问答 Demo |
| `openvino/src/embedding.py` | Phase 3 · Qwen3-Embedding OpenVINO 封装 |
| `openvino/src/vector_store.py` | Phase 3 · ChromaDB 持久化封装 |
| `openvino/src/llm.py` | Phase 3 · Qwen3-1.7B-int4 LLMPipeline 封装 |
| `openvino/src/rag.py` | Phase 3 · embedder + store + LLM 编排 |
| `openvino/scripts/download_eval_materials.py` | Phase 3/4 · 评测材料自动下载（PDF + OmniDocBench） |
| `openvino/scripts/run_omnidocbench_subset.py` | Phase 4 · OmniDocBench 子集对比骨架 |

### 数据

| 路径 | 用途 |
|------|------|
| `openvino/data/test_images/` | Phase 1 测试图 10 张（git 跟踪） |
| `openvino/data/test_documents/` | Phase 2 测试 PDF 3 份（git 跟踪） |
| `openvino/data/eval_questions.jsonl` | Phase 3/4 评测题集（git 跟踪） |
| `openvino/data/eval_questions.schema.md` | 评测题集 schema |
| `openvino/data/demo_questions.txt` | Phase 3 端到端 Demo 的 5 条业务问题（git 跟踪） |
| `openvino/data/p1_2_pages/` | P1-2 OCR 对比的 3 张代表性页面（git 跟踪，复现入口） |
| `openvino/data/eval_documents/` | Phase 3/4 真实评测 PDF（`.gitignore`，自动下载） |
| `openvino/data/omnidocbench/` | OmniDocBench 标注 + 采样图（`.gitignore`） |
| `openvino/chroma_db/` | Phase 3 · Demo 集 ChromaDB 持久化库（`.gitignore`） |
| `openvino/chroma_db_eval/` | Phase 3 · P1-1 eval 集 ChromaDB（`.gitignore`） |
| `openvino/results/phase3/` | Phase 3 · QA 运行结果（脚本生成） |
| `openvino/results/phase2_eval/` | P1-1 真实评测 PDF 切片输出（`.gitignore`） |
| `openvino/results/phase1_p1_2/` | P1-2 Tesseract vs Paddle 对比输出（`.gitignore`） |
