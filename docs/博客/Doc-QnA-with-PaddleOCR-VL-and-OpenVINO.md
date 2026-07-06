# 全链路本地化：用 PaddleOCR-VL + OpenVINO 打造带来源引用的文档问答系统

> 飞桨黑客松第 10 期 · 进阶任务 #13「基于 OpenVINO 的多模态文档理解与智能应用开发」参赛项目复盘。
> 代码已合入 PR：[openvinotoolkit/openvino_build_deploy#552](https://github.com/openvinotoolkit/openvino_build_deploy/pull/552)
> 项目仓库：[bob798/doc-qna-openvino](https://github.com/bob798/doc-qna-openvino)

## 一、为什么做这个项目

企业里最常见的知识载体不是数据库，而是 PDF：产品手册、规格表、扫描件、国标文档。想对它们提问，通常要么把文档上传到云端大模型（数据出域，敏感场景不可接受），要么自建 RAG（检索增强生成）系统——而自建系统最容易死在第一步：**文档解析**。

- 纯文本抽取（pdfplumber 之类）对扫描件无能为力；
- 传统 OCR（Tesseract）能认字，但表格结构全丢——我们实测一张规格表页，Tesseract 输出与真值的字符相似度只有 **9.55%**，rowspan/colspan 信息完全丢失；
- 表格恰恰是产品手册里最有价值的部分："A300 的额定功率是多少"这种问题，答案就藏在表格单元格里。

这个项目要回答的问题是：**能不能用一套完全跑在本地 CPU 上的开源方案，把"扫描件进、可引用的答案出"这条链路走通？**

答案是可以。四个模型、全部 OpenVINO 推理、无需任何云端 API：

| 角色 | 模型 | 大小 | 作用 |
|------|------|------|------|
| 文档解析 | [PaddleOCR-VL-1.5](https://huggingface.co/zhaohb/PaddleOCR-VL-1.5-ov)（0.9B VLM） | ~2.7 GB | PDF/扫描件 → 结构化 Markdown（含表格） |
| 向量化 | [Qwen3-Embedding-0.6B-int8](https://huggingface.co/OpenVINO/Qwen3-Embedding-0.6B-int8-ov) | ~600 MB | 1024 维多语言向量 |
| 重排 | [bge-reranker-base-int8](https://huggingface.co/OpenVINO/bge-reranker-base-int8-ov)（cross-encoder） | ~300 MB | 精排候选 + 抗"域外实体串台" |
| 生成 | [Qwen3-1.7B-int4](https://huggingface.co/OpenVINO/Qwen3-1.7B-int4-ov) | ~1 GB | RAG 问答 + 来源引用 |

## 二、总体架构与功能模块

```
PDF / 扫描件 / 规格表
        ↓
① PaddleOCR-VL（OpenVINO）      —— 视觉语言 OCR，输出结构化 Markdown
        ↓
② 表格感知切片（table-aware chunking）
        ↓
③ Qwen3-Embedding-0.6B-int8（OpenVINO）→ ChromaDB 向量库（召回 Top-20）
        ↓
④ bge-reranker-base-int8（OpenVINO, cross-encoder）—— 精排 + 三级抗串台守卫
        ↓
⑤ Qwen3-1.7B-int4（OpenVINO GenAI）—— RAG 生成，带 [文档名 p.页码] 引用
```

### 模块拆解

**① 文档解析（`src/doc_parser.py` + `src/inference.py`）**
对有文字层的 PDF 直接走文字层；对扫描页自动降级到 PaddleOCR-VL。VLM 式 OCR 的关键优势是输出**结构化 Markdown**——表格保留为 Markdown 表格，而不是一串乱序文字。

**② 表格感知切片（`src/chunker.py`）**
普通按段落切片会把表格拦腰截断。我们的做法：**表格每一行单独成 chunk，且携带表头作为上下文**。比如规格表的一行会被切成：

```
| 型号 | 额定功率 | 工作温度 |
| A300 | 500W | -10~55℃ |
```

这样"A300 额定功率"这种单元格级问题，向量检索能直接命中含答案的行。段落文本则按语义边界切分，并过滤页眉/页脚噪声（短于 60 字符的 boilerplate 会把真正的事实挤出 Top-K）。

**③ 向量化入库（`src/embedding.py` + `src/vector_store.py`）**
Qwen3-Embedding 经 OpenVINO int8 量化后在 CPU 上约 139 ms/chunk，99 个 chunk 全量入库约 14 秒。ChromaDB 持久化，二次运行直接复用索引。

**④ Cross-encoder 精排 + 四级抗幻觉守卫（`src/reranker.py` + `src/entity_gate.py` + `src/answer_grounding.py`）**
bi-encoder 把整句压成一个向量，"火星探测器的额定功率"里"额定功率"主导相似度，会骗过 `--min_score` 阈值。cross-encoder 把 (query, chunk) 拼在一起联合编码，能"看见"主体不一致——同一条串台题 bi-encoder cosine=0.465、reranker sigmoid 只有 0.043。四级守卫层层兜底：①**实体一致性守卫**拦 `A500`/未知标准号这类查得出的伪实体;②**cross-encoder 重排**拦语义主体串台;③**主体接地守卫**拦"特斯拉 Model 3 的工作温度"这种极强字段匹配下 reranker 仍给高分的反转盲区（主体词不在证据里→拒答）;④**答案数值接地守卫**（生成后）拦"检索不到正确内容、小模型硬答一个日期/数字"的幻觉——回答里的日期/数字必须能在证据中核实，否则改判拒答。前三级实测 5 域内 + 7 域外串台题:域内 5/5 作答、域外 7/7 拒答（旧单阈值方案仅 2/7）;第四级把 Q5 那种编造实施日期的 silent failure 也兜成了干净拒答。

**⑤ RAG 问答（`src/rag.py` + `src/llm.py`）**
Qwen3-1.7B-int4 通过 `openvino_genai.LLMPipeline` 加载，关闭 thinking 模式保证输出紧凑。再叠加两道防线：

1. **严格 system prompt**：只允许使用检索到的参考资料，答案必须带 `[文档名 p.页码]` 引用，找不到就答"文档中未提及"；
2. **来源引用可回溯**：每条答案都能定位到原文页码，人工可验证。

## 三、一步步复现

### 环境要求

- Python 3.10 / 3.11 / 3.12，Windows / Linux / macOS 均可
- 纯 CPU 即可运行（Intel i5 实测），有 Intel GPU 可 `--device GPU` 加速
- 磁盘约 8 GB（模型 + venv + 向量库）

### 1. 一键运行（推荐）

```bash
git clone https://github.com/openvinotoolkit/openvino_build_deploy
cd openvino_build_deploy/demos/doc_qna_demo
python main.py
```

就这一条命令。`main.py` 会自动检测并安装缺失依赖、首次运行时自动从 HuggingFace 下载 OpenVINO IR 模型、从仓库自带的解析结果构建 ChromaDB 索引，然后跑默认的 5 题 demo。

Windows 用户也可以直接双击 `setup/install.bat`，脚本会完成 clone → venv → 依赖 → 运行全流程。

### 2. 自定义玩法

```bash
# 单条提问（结果写入 results/demo_run_single.*，不会覆盖 5 题报告）
python main.py --question "A100 工作温度是多少？"

# 切换设备
python main.py --device GPU

# 换更大的 LLM（如果内存充足）
python main.py --llm_model_id OpenVINO/Qwen3-8B-int4-ov

# 调整重排拒答阈值
python main.py --rerank_min_score 0.30

# 四级守卫可逐个开关做消融对照（默认全开）：
python main.py --no_reranker        # 关重排（min_score 缺省自动转 0.35 兜底）
python main.py --no_entity_check    # 关实体码守卫（A500/未知标准号）
python main.py --no_subject_check   # 关主体接地守卫（特斯拉 Model 3 类串台）
python main.py --no_answer_check    # 关答案数值接地守卫（Q5 编造日期类幻觉）

# 复现抗串台守卫的分离度评测（5 域内 + 7 域外题，不加载 LLM，很快）
python scripts/eval_guard.py
```

### 3. Windows 踩坑记录（都已在代码里兜底）

- **控制台编码**：cmd 默认 cp1252，中文直接崩——代码里强制 `sys.stdout.reconfigure(encoding="utf-8")`；
- **symlink 权限**：无开发者模式时 huggingface_hub 建符号链接报错——设置 `HF_HUB_DISABLE_SYMLINKS=1`；
- **jinja2 缺失**：transformers 5.x 不再传递依赖 jinja2，而 `apply_chat_template` 必需它——`requirements.txt` 已显式声明 `jinja2>=3.1`。

## 四、运行效果

5 题端到端问答（CPU）：

| # | 问题 | 答案 |
|---|------|------|
| 1 | A100 型号的工作温度范围是多少？ | -20~70℃ `[spec_with_tables p.1]` |
| 2 | A300 型号的额定功率是多少瓦？ | 500W `[spec_with_tables p.1]` |
| 3 | 故障代码 E01/E02 如何处理？ | 应立即联系售后处理 `[spec_with_tables p.1]` |
| 4 | GB/T 2423.1—2008 对应的国际标准编号？ | IEC 60068-2-1:2007 `[gb_t_2423_1 p.11]` |
| 5 | GB/T 2423.1—2008 的实施日期？ | 拒答（答案数值接地守卫拦下模型编造的"2008 年 1 月 1 日"） |

Q1–Q4 均答对且引用正确。Q5 则展示了一道最隐蔽的 silent failure 是怎么被兜住的：真正含"2009-10-01 实施"的 chunk 因整段被英文标题占主导，既进不了 bi-encoder 召回前列、cross-encoder 也把它打成 0 分；而只含标准号"GB/T 2423.1—2008"的封面页因主体强匹配排到最前——这时 1.7B 模型会把标准年份"2008"误当成实施日期，硬答"2008 年 1 月 1 日"。这类"检索不到正确内容 + 小模型硬答"最难防，前面的实体/重排/主体接地守卫都拦不住（主体"GB/T 2423.1"确实在证据里）。**答案数值接地守卫**在生成后再核一遍：回答里的每个日期/数字都必须能在检索证据里找到，"2008 年 1 月 1 日"查无实据 → 改判"文档中未提及"。**对文档问答来说，"知之为知之"比"什么都敢答"更重要。**

抗串台专项测试（`scripts/eval_guard.py`，5 域内 + 7 域外/串台题）：**域内 5/5 正确作答，域外 7/7 正确拒答**（旧单阈值方案仅 2/7）。最难的一类"域外实体 + 域内字段"串台——如"火星探测器的额定功率""特斯拉 Model 3 的工作温度"——也被拦下：cross-encoder 把火星探测器题的相关度压到 0.043，主体接地守卫再兜住 reranker 给了 0.77 高分的特斯拉题（"Model"/"特斯拉"不在任何证据里 → 拒答）。

**① 5 题端到端问答**（真实终端截图）：Q1–Q4 表格/标准事实全部正确作答并附 `[文档名 p.页码]` 引用；Q5 的实施日期问题被**答案数值接地守卫**拦下——模型想拿标准年份"2008"硬答"2008 年 1 月 1 日"，因证据中无法核实而改判"文档中未提及"。

![5 题端到端问答：Q1–Q4 正确作答附引用，Q5 编造的实施日期被答案数值接地守卫改判拒答](https://raw.githubusercontent.com/bob798/doc-qna-openvino/main/assets/demo_5q.png)

**② 域外实体串台实测**：问"特斯拉 Model 3 的工作温度"，reranker 虽因字段强匹配给到 0.770，但**主体接地守卫**发现"Model"不在任何证据里 → 直接拒答（`llm=0.0ms`，未进 LLM）。

![域外实体串台：特斯拉 Model 3 的工作温度被主体接地守卫拒答，llm=0ms 未进 LLM](https://raw.githubusercontent.com/bob798/doc-qna-openvino/main/assets/demo_crosstalk.png)

**③ 抗串台分离度评测**（`python scripts/eval_guard.py`）：域内 **5/5 正确作答**、域外 **7/7 正确拒答**（旧单阈值仅 2/7），逐题标注由哪一级守卫拦下，并给出 bi-encoder 与 reranker 的分离区间对比。

![分离度评测表：域内 5/5、域外 7/7，逐题标注拦截守卫与 bi-encoder/reranker 分离区间](https://raw.githubusercontent.com/bob798/doc-qna-openvino/main/assets/demo_eval.png)

### 性能（Intel i5，纯 CPU）

| 阶段 | 平均耗时 |
|------|----------|
| 查询向量化 | ~140 ms（含首查冷启动，稳态 ~55 ms） |
| ChromaDB 检索 | 2 ms |
| Cross-encoder 重排（Top-20） | ~660 ms |
| LLM 生成 | ~2,440 ms（~19 tok/s） |
| **端到端 / 题** | **~3.2 s** |

瓶颈仍在 LLM 生成；新增的 cross-encoder 重排每题只多花约 0.66 s，却换来"域外实体串台"从拦不住到 7/7 拦下，且顺带把一道 retrieval-miss 题救了回来，这笔开销很划算。OpenVINO 的价值在于：1.7B int4 模型在没有独显的普通办公机上做到 ~19 tok/s，问答体验已经可用；换 Intel GPU 实测还有约 1.47× 加速。

## 五、已知限制（诚实清单）

1. **简短指引类答案的保守拒答（已缓解）**：早期严格 prompt 下，1.7B 模型会把"联系售后"判为"没有处理方法"而拒答；加入重排后证据更干净，该题现已能正确作答（"应立即联系售后处理"），但这类"简短指引 vs 抗幻觉"的边界取舍在更刁钻的表述上仍可能出现，7B+ 模型更稳；
2. **语义稀释 chunk 的漏召回（Q5，幻觉已拦，召回仍是短板）**：真答案所在 chunk（"2009-10-01 实施"）整段被英文标准标题占主导，bi-encoder 召回不进前列、cross-encoder 也打成 0 分；反而只含标准号的封面页排到最前，1.7B 模型据此把"2008"误当实施日期。这类"检索不到正确内容 + 小模型硬答"和 #3 的实体串台是不同问题，已由**答案数值接地守卫**兜住：回答里的日期/数字必须能在证据中核实，否则改判拒答——所以 Q5 现在是干净的拒答而非幻觉。但要真正**答对**它，还需改善召回（更细的 chunk、多向量/父子块检索或更强 embedding），这属下一步方向；
3. **半域外问题的实体串台（已解决）**：最初"火星探测器的额定功率"这种"域外实体 + 域内字段"组合会以 0.465 的较高 bi-encoder 相似度命中"额定功率"chunk（域内题最低才 0.722，二者区间重叠，单一 `--min_score` 阈值拦不干净）。现在用**三级守卫**在进 LLM 前拦截:①**实体一致性守卫**确定性拦住 `A500`/`GB/T 9999` 这类查得出的伪实体;②**cross-encoder 重排**(`OpenVINO/bge-reranker-base-int8-ov`)把 (query, chunk) 联合编码,火星探测器题重排分只有 0.043、珠峰 0.006,远低于域内 0.98+;③**主体接地守卫**兜住"特斯拉 Model 3 的工作温度"这种极强字段匹配下 reranker 仍给 0.77 的反转盲区(主体词不在任何证据里→拒答)。

## 六、小结

这个项目验证了一条完全本地化的文档问答路线：**PaddleOCR-VL 负责"看懂"文档（尤其是表格），OpenVINO 负责让三个模型都跑得动，表格感知切片 + 严格 grounding 负责让答案可信可溯源**。全程无 GPU 依赖、无云端 API、数据不出本机——对手册问答、规格核对、标准检索这类场景，已经是一个可以直接落地的起点。

欢迎试用并提 issue：

- Demo PR：https://github.com/openvinotoolkit/openvino_build_deploy/pull/552
- 项目仓库（含 benchmark 与评测材料）：https://github.com/bob798/doc-qna-openvino

---

*本文是飞桨黑客松第 10 期（文心伙伴赛道 · Intel）进阶任务 #13 的参赛复盘，感谢 PaddlePaddle 与 Intel OpenVINO 团队的评审与反馈。*
