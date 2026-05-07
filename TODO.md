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

- [ ] **Phase 1 (Week 1)**: 环境搭建 + 模型验证 + 推理 Benchmark
- [ ] **Phase 2 (Week 2)**: 文档解析 + 表格感知切片模块
- [ ] **Phase 3 (Week 3)**: RAG 问答链路（Embedding + ChromaDB + LLM）
- [ ] **Phase 4 (Week 4)**: Tesseract vs PaddleOCR-VL 对比评测 + 优化
- [ ] **Phase 5 (Week 5)**: 整理 Notebook + README + requirements + 提交

### 周报

> 在 [PFCCLab/Camp](https://github.com/PFCCLab/Camp/pull/584) 提交周报
> 目录：`WeeklyReports/Hackathon_10th/ERNIEPartner/`

- [ ] Week 1 周报
- [ ] Week 2 周报
- [ ] Week 3 周报
- [ ] Week 4 周报
- [ ] Week 5 周报

---

## 高通赛题｜基于 QNN 部署 PaddleOCR-VL 模型

> 技术标签：PaddleOCR-VL，高通 QNN SDK，Hexagon NPU，Paddle2ONNX
> 截止日期：2026-06-05

### 提交内容

- [x] 模型转换脚本（Paddle → ONNX → QNN 全链路）
- [x] 转换说明文档
- [x] 端侧推理服务代码
- [x] 文档解析 pipeline 代码（参考 doc_parser）
- [x] 精度对比评测脚本
- [x] 性能测试脚本
- [x] Dockerfile + docker-compose

### 验证阶段（待 SDK 就绪后执行）

- [ ] Docker 环境构建成功
- [ ] 布局检测模型：Paddle → ONNX → QNN 转换跑通
- [ ] VL 模型：ONNX 导出成功（验证算子兼容性）
- [ ] VL 模型：QNN 转换成功
- [ ] HTP-simulator 上推理可运行
- [ ] 精度损失 ≤ 5%
- [ ] 性能报告填充实测数据

> 代码目录：[`qnn/`](qnn/)

---

## 文件索引

| 文件 | 用途 |
|------|------|
| `docs/进阶方案.md` | 已提交的进阶任务方案（OpenVINO） |
| `docs/开发手册.md` | Phase 1-5 执行手册（按周推进） |
| `docs/技术介绍.md` | 技术科普（OpenVINO / PaddleOCR-VL） |
| `docs/差异化分析.md` | 与现有 notebooks 的差异论证 |
| `docs/打卡记录.md` | 打卡任务 #1 提交记录 |
| `assets/` | 截图（打卡运行结果） |
| `qnn/` | **高通 QNN 赛题完整代码和文档** |
| `qnn/README.md` | QNN 赛题说明 |
| `qnn/docs/模型转换指南.md` | Paddle → ONNX → QNN 全链路文档 |
| `qnn/scripts/` | 转换/量化/评测脚本 |
| `qnn/src/` | 推理服务 + 文档解析 Pipeline |
