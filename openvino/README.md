# 进阶任务 #13 · OpenVINO 子项目代码

> 飞桨黑客松第 10 期 · 文心伙伴赛道
> 项目：基于 PaddleOCR-VL + OpenVINO 的产品文档智能入库与客服问答系统

本目录是 OpenVINO 路径的全部代码和实验产物。Phase 1（推理 Benchmark）、Phase 2（文档解析 + 切片）、**Phase 3（RAG 端到端问答）** 均已完成代码实现，运行所需的模型 / 测试数据由开发者本机准备（IR 通过 huggingface_hub 自动拉取，PDF 见 `data/`）。

```
openvino/
├── README.md                       本文档
├── requirements.txt                依赖
├── configs/models.json             模型/路径配置
├── data/
│   ├── test_images/README.md       Phase 1 的 10 张测试图片说明
│   ├── test_documents/README.md    Phase 2 的 3 份测试 PDF 说明
│   ├── demo_questions.txt          Phase 3 端到端 Demo 的 5 条业务问题
│   └── eval_questions.jsonl        评测题集骨架（W4 加分项用）
├── src/
│   ├── inference.py                OpenVINO / PyTorch / Tesseract 推理封装
│   ├── pdf_preprocessor.py         pdfplumber 可读性判断 + pdf2image 渲染
│   ├── doc_parser.py               PaddleOCR-VL 全图解析 → 结构化 Markdown
│   ├── chunker.py                  表格感知切片 + 元数据
│   ├── pipeline.py                 Phase 2 端到端管线
│   ├── embedding.py                Phase 3 · Qwen3-Embedding-0.6B-int8 OpenVINO 封装
│   ├── vector_store.py             Phase 3 · ChromaDB 持久化封装
│   ├── llm.py                      Phase 3 · Qwen3-1.7B-int4 openvino_genai 封装
│   └── rag.py                      Phase 3 · embedder + store + LLM 编排
├── scripts/
│   ├── benchmark_inference.py      Phase 1 · PyTorch vs OpenVINO 速度
│   ├── benchmark_ocr_quality.py    Phase 1 · Tesseract vs PaddleOCR-VL 质量
│   ├── run_phase2_pipeline.py     Phase 2 · CLI 入口
│   ├── build_index.py              Phase 3 · 把 chunks.jsonl 灌入 ChromaDB
│   └── run_qa.py                   Phase 3 · 端到端 RAG 问答 Demo
└── results/
    ├── phase1/                     Benchmark 输出（脚本运行后自动生成）
    ├── phase2/                     切片输出（脚本运行后自动生成）
    └── phase3/                     问答结果（脚本运行后自动生成）
```

---

## 准备工作

### 1. 安装依赖

```bash
cd openvino
python -m venv .venv && source .venv/bin/activate    # 或 conda
pip install -r requirements.txt
# 可选：OCR 质量基线对比（仅评测脚本需要，默认 demo 不依赖）
#   pip install pytesseract
#   Tesseract 需另装二进制（macOS：`brew install tesseract tesseract-lang`）
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

## Phase 3 · RAG 端到端问答

### 模型选择

| 角色 | 模型 ID | 设备 | 下载量 | 备注 |
|------|---------|------|--------|------|
| OCR | `zhaohb/PaddleOCR-VL-1.5-ov` | CPU / GPU | ~2.7 GB | INT4 LLM + INT8 Vision + PP-DocLayoutV3 |
| Embedding | `OpenVINO/Qwen3-Embedding-0.6B-int8-ov` | CPU / GPU | ~600 MB | 1024 维，多语言含中文，官方预转 INT8 IR |
| LLM | `OpenVINO/Qwen3-1.7B-int4-ov` | CPU / GPU | ~1 GB | 通过 `openvino_genai.LLMPipeline` 加载，`enable_thinking=False` |
| 向量库 | ChromaDB Persistent | — | — | cosine 距离，`chroma_db/` 目录 |

> **磁盘需求**：三个模型 IR 合计约 4.3 GB（首次运行自动下载到 HF cache）。加上
> Python 虚拟环境和 ChromaDB 持久化目录，建议预留 **8 GB** 可用空间。

> **关于"BGE-small-zh"**：原计划用 BAAI/bge-small-zh-v1.5 + OV 转换。`OpenVINO/`
> 官方仓库当前没有该模型的预转 INT8 IR，本项目为了与 Phase 1/2 保持"全程官方
> 预转 IR"的复现性，改用 `OpenVINO/Qwen3-Embedding-0.6B-int8-ov`（多语言含中文，
> INT8）。若用户想坚持 BGE，可用 optimum-intel 自行转换后通过 `--embed_local_dir`
> 传入。

### 一次性准备（自动下载 IR 到 HF cache）

```powershell
# Windows 默认 cp1252 控制台需要 UTF-8，否则中文 print 直接崩溃
$env:PYTHONIOENCODING = "utf-8"
# Windows 无开发者模式时，huggingface_hub 的 symlink 会报权限错误
$env:HF_HUB_DISABLE_SYMLINKS = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
```

### 1. 灌库：Phase 2 chunks → ChromaDB

```bash
python scripts/build_index.py \
    --chunks_dir results/phase2 \
    --persist_dir chroma_db \
    --device CPU \
    --reset
```

输出：

- `chroma_db/` —— ChromaDB 持久化目录
- `build_index.summary.json` —— 编码耗时、维度、入库总数

实测（CPU，99 个 chunk）：embedder 编译 ~4s，encode `138.6 ms/chunk`，总入库 ~14s。

### 2. 端到端问答

```bash
python scripts/run_qa.py \
    --questions_file data/demo_questions.txt \
    --persist_dir chroma_db \
    --top_k 5 --max_new_tokens 384 \
    --embed_device CPU --llm_device CPU \
    --out results/phase3/qa_run.json \
    --out_md results/phase3/qa_run.md
```

可选问题源：

- `--question "你的问题"` —— 单条 CLI 输入
- `--questions_file data/demo_questions.txt` —— 每行一题的纯文本
- `--eval_jsonl data/eval_questions.jsonl --question_ids q003,q008` —— 从评测集挑题

如果 `chroma_db/` 为空，加 `--auto_build_index` 让 `run_qa` 自动从
`--chunks_dir` 灌一遍。

### 3. Demo 输出示例（CPU，5 题）

| # | 问题 | 答案 |
|---|------|------|
| 1 | A100 型号的工作温度范围是多少？ | A100 型号的工作温度范围是 -20~70℃ `[spec_with_tables p.1]` |
| 2 | A300 型号的额定功率是多少瓦？ | A300 型号的额定功率是 500W `[spec_with_tables p.1]` |
| 3 | 故障代码 E01 或 E02 出现时应该如何处理？ | 文档中未提及故障代码 E01 或 E02 的处理方法。`[spec_with_tables p.1]`  （注：原文有"请立即联系售后"，模型判定为不构成"处理方法"而保守拒答——避免幻觉的取舍） |
| 4 | GB/T 2423.1—2008 这份标准对应的国际标准编号是多少？ | GB/T 2423.1—2008 对应的国际标准编号是 IEC 60068-2-1:2007 `[gb_t_2423_1 p.3]` |
| 5 | GB/T 2423.1—2008 标准的实施日期是哪一天？ | 文档中未提及 GB/T 2423.1—2008 标准的实施日期。`[gb_t_2423_1 p.1]`  （注：原文 p.1 有"2009-10-01 实施"，但该 chunk 因英文标题占主导未被 Top-5 召回——典型 retrieval miss） |

性能（CPU，i5 系列，Qwen3-1.7B-int4）：

| 阶段 | 平均耗时 (ms) |
|------|---------------|
| embed query | 69 |
| ChromaDB retrieve | 1.7 |
| LLM 生成（含 token 解码） | 3270 |
| **end-to-end / 题** | **3340** |
| LLM 吞吐 | ~10.7 tok/s |

完整运行日志见 `results/phase3/qa_run.md` 与 `qa_run.json`。

### Phase 3 已知限制

1. **Q5 风格的 retrieval miss**：当目标事实所在的 chunk 被其他主题（如英文标题、
   章节名）"语义稀释"时，1024 维小模型可能漏召回。短期缓解 = 调大 `--top_k` 或
   切更小的 chunk；W4 加分项里会试用 BGE-reranker-base 重排候选。
2. **Q3 风格的保守拒答**：严格 system prompt 下，模型会把"联系售后"这种简短指引
   判为"未提供处理方法"，这是抗幻觉的代价——保留这种取舍。
3. **语义相近的半域外问题串台**：明显域外问题（如"珠峰高度"）能靠严格 prompt +
   `--min_score` 相似度阈值正确拒答；但"域外实体 + 域内字段"的组合（如"火星探测器
   的额定功率"）会以较高相似度命中"额定功率"所在 chunk，1.7B 模型可能忽略主体不一致
   直接用本机参数作答。实测分数分布（Qwen3-Embedding-0.6B-int8，99 chunks 语料）：
   域内 5 题 top-1 ∈ [0.722, 0.840]，域外 5 题 top-1 ∈ [0.237, 0.465]（火星探测器题
   = 0.465 为域外最高）。默认 `--min_score 0.35` 只拦明显域外题；本语料上调到
   `--min_score 0.5` 可连该串台题一起在进 LLM 前拦截且不误伤域内题，但阈值随语料
   分布变化，通用解法仍是 reranker 或实体一致性校验（加分项方向）。

### 4. P1-1 真实评测 PDF 业务问题集（2 份 NVIDIA H100 Product Brief）

把 Demo 主证据从 Phase 2 已有 4 份 PDF 升级到 `data/eval_documents/` 真实评测 PDF。
为保持 Phase 2 OCR 时间可控，本期只灌入 2 份 NVIDIA H100 PB（H100 PCIe + H100 NVL，
共 46 页 / 372 chunks 灌入 `chroma_db_eval/`，含 chunker 表格 caption 增强 + 短
chunk 噪声过滤 `--min_chars 60`）。从 `eval_questions.jsonl` 18 题里挑 5 题代表
4 类查询：

| # | id | 类型 | 问题 | 期望关键词 | 实测回答 | 结论 |
|---|----|------|------|-----------|---------|------|
| 1 | q008 | table_lookup | H100 PCIe HBM 显存容量？ | 80 GB | "文档中未提及" | ❌ retrieval miss → 正确拒答 |
| 2 | q009 | table_lookup | H100 PCIe TDP？ | 350W | 350 W | ✅ |
| 3 | q011 | table_lookup | H100 NVL 显存容量？ | 94 GB | "16 GB" | ❌ 幻觉（候选里没有 94 GB 行） |
| 4 | q013 | cross_doc | 对比 H100 PCIe / NVL 显存差异 | 80 GB + 94 GB | "NVL 更高"（无具体数字） | ⚠️ 方向对但没拿到具体数字 |
| 5 | q015 | refusal | H100 NVL 游戏渲染？ | must_refuse | "不支持光线追踪游戏渲染" | ✅ 正确拒答 |

**1/5 业务 hit + 2/5 正确拒答 + 1/5 方向对 + 1/5 幻觉**。运行命令：

```bash
# 灌库（带 caption 增强 + 噪声过滤）
python scripts/build_index.py \
    --chunks_files results/phase2_eval/h100_pcie_pb.chunks.jsonl results/phase2_eval/h100_nvl_pb.chunks.jsonl \
    --persist_dir chroma_db_eval --collection eval_chunks \
    --device CPU --batch_size 8 --min_chars 60 --reset

# 5 题选答
python scripts/run_qa.py \
    --eval_jsonl data/eval_questions.jsonl --question_ids q008,q009,q011,q013,q015 \
    --persist_dir chroma_db_eval --collection eval_chunks \
    --top_k 8 --max_new_tokens 384 \
    --out results/phase3/eval_qa_run.json --out_md results/phase3/eval_qa_run.md
```

性能：avg total 3528 ms / 题（与 Demo 集相当），retrieve 1.6 ms（无瓶颈），LLM 3454 ms。

#### P1-1 关键发现：small-embedder 在表格行 chunk 上的系统性偏差

q008 / q011 / q013 都是同一类失败：要查表格里某一行的具体数值，但答案行 chunk
（如 `Table 2. Memory Specifications | Memory size | 94 GB`）的余弦相似度只有
**0.38**，而通用表头 chunk（如 `Specifications | NVIDIA H100 NVL`）能拿到 **0.75+**。
即使把 `--top_k` 调到 30，含 `94 GB` 的行 chunk 仍然进不了候选池。

原因可定位到两点：

1. **last-token-pool 偏向尾部**：Qwen3-Embedding 用最后有效 token 的 hidden state
   做句嵌入；表格行 chunk 以 `| --- |` 分隔符或 `| 94 GB |` 这种短串结尾，嵌入被
   分隔符语义稀释，整体偏离 "memory size capacity" 这种概念。
2. **小模型对多列表格列名的辨别力不足**：Table 1 表头是 `Specification | NVIDIA H100 NVL`，
   Table 2 表头是 `Specification | Description`，两张表的内容主题完全不同，但
   embedding 距离差异小；用户查询里出现 "NVIDIA H100" → Table 1 全行通杀。

**这正是 W4 P2 加分项 BGE-reranker-base 设计要解决的问题**——把召回扩到 Top-30，
让 reranker 用 cross-encoder 在 query × chunk 上重新打分，能把 "Memory size | 94 GB"
顶起来。本期已经做了两项治标改进作为铺垫：

- **chunker 表格 caption 增强**：每个表格行 chunk 在文本头注入紧贴表格的上文标题
  （如 `Table 2. Memory Specifications`），让"Memory" 类关键词能间接命中；
- **`--min_chars` 短 chunk 噪声过滤**：直接丢掉页脚 `"PB-11133-001_v02 | 6"` 这类
  N 次重复短串，从 397 chunks 缩到 372；q009 / q015 召回质量明显改善。

q009 / q015 能成功是因为：q009 的答案 chunk 里同时含 "Total board power" 和 "350 W default"
两个强语义锚，q015 是 refusal 类（只要候选里没有 "gaming" 就该拒答），不依赖
精确行召回。

---

### 5. P1-2 Tesseract vs PaddleOCR-VL 少量页对比

为支撑 Phase 3 OCR 路径选型，选 3 张代表性页面（同一份 PDF，避免风格差异引入
噪声），分别覆盖"中文 + LaTeX 公式 / 中文长文档 / 中文 + 复杂表格"三种典型场景：

| 文件 | 类型 | 来源 |
|------|------|------|
| `formula_gb2423_p05.png` | 中文 + LaTeX 公式 | GB/T 2423.1-2008 p.5（含 `$5\,K$`、`$0.5\,m/s$` 等行内公式） |
| `text_gb2423_p07.png` | 纯中文长文 | GB/T 2423.1-2008 p.7（多级标题 + 段落） |
| `table_gb2423_p14.png` | 中文 + HTML 表格 | GB/T 2423.1-2008 p.14（附录 NB 表 NB.1，含 rowspan/colspan） |

运行命令（复用 Phase 1 已有 `benchmark_ocr_quality.py`）：

```bash
# Windows: 需要 Tesseract + chi_sim 语言包
# 1) 安装 Tesseract for Windows（默认 C:\Program Files\Tesseract-OCR\）
# 2) 下载 chi_sim.traineddata 放到任意可写目录，环境变量 TESSDATA_PREFIX 指过去
$env:TESSDATA_PREFIX = "$HOME\.tessdata"
$env:PATH = "C:\Program Files\Tesseract-OCR;" + $env:PATH

python scripts/benchmark_ocr_quality.py \
    --image_dir data/p1_2_pages \
    --ir_dir ./models \
    --tesseract_lang chi_sim+eng \
    --output_dir results/phase1_p1_2 \
    --device CPU
```

**结果总览**（CPU，Intel AI PC，Tesseract 5.x + chi_sim+eng / PaddleOCR-VL-1.5-ov）：

| 页面 | 类型 | Tesseract 耗时 | PaddleOCR-VL 耗时 | Tesseract 字数 | Paddle 字数 | 字符相似度 |
|------|------|----------------:|------------------:|----------------:|------------:|-----------:|
| formula_gb2423_p05 | 公式 | **2.8 s** | 26.9 s | 1416 | 1189 | 77.7% |
| text_gb2423_p07 | 纯文字 | **2.6 s** | 26.9 s | 1239 | 983 | 74.8% |
| table_gb2423_p14 | 表格 | **0.8 s** | 11.2 s | 518 | 2309 | **9.6%** |

#### 关键定性差异

1. **表格结构（决定性）**——Tesseract 把附录 NB.1 表（rowspan/colspan）拍扁成无
   结构纯文本，仅截到表头几个词；PaddleOCR-VL 输出带 `<table>` HTML（rowspan/
   colspan 完整保留），可被 chunker 的 `_normalize_html_tables` 转 Markdown 后
   逐行切片入库。**这是 9.55% 字符相似度的成因**——结构信息完全不在 Tesseract 输出里。
   对 RAG 场景，没结构就没法做 "Memory size | 80 GB" 这种精确行级召回。

2. **公式标记**——Tesseract 0 个 LaTeX 标记，PaddleOCR-VL 输出 12 个 `$...$` 行内
   公式。Tesseract 把 `$5\,K$` 退化成纯文本 `5 K`（人眼能看，但 Markdown 渲染丢
   公式格式，下游若想做"按公式查文档"的查询会缺失关键元数据）。

3. **段落布局**——两者在纯文字页都拿到了主要内容，但 Tesseract 把 `## 3.1`、
   `## 4.2` 这类章节号识别为段落首字符（无 Markdown 标题语义），且偶发字符替换
   （把 `Ae` 识为 `Ac`）。PaddleOCR-VL 输出已经是结构化 Markdown，标题/段落/表格
   层级清晰。

#### 速度 vs 质量取舍

| | Tesseract | PaddleOCR-VL |
|--|-----------|--------------|
| 速度（CPU） | **0.8–2.8 s/页**（10–30× 快） | 11–27 s/页 |
| 表格结构 | 散落纯文本 | HTML 表格 + rowspan/colspan ✅ |
| LaTeX 公式 | 退化为纯文本 | `$...$` 行内公式 ✅ |
| 标题层级 | 平铺 | Markdown `##` 层级 ✅ |
| 中文准确率 | 字符级 OK，偶有替换 | 字符级 OK，复杂版式更稳 |
| 适用场景 | 纯文字、对速度敏感的批量预筛 | RAG 入库、结构化 Markdown 交付 |

**Phase 3 主线选 PaddleOCR-VL 的理由（被本对比证据支持）**：RAG 检索需要表格行
级和章节标题元信息支撑精确召回与 `[doc p.页]` 引用——Tesseract 缺这两类结构
就无法支撑下游业务。速度差距（10–30×）则用"批处理一次性入库 + 后续查询毫秒级
检索"摊掉。

输出物：`results/phase1_p1_2/quality_compare.{md,json}` + 每张图单独的
`quality_compare/<img>/` 目录（含 Tesseract `*.txt` / PaddleOCR-VL `*.md` /
diff 文件 / 原图副本，便于在最终 README/Notebook 引用截图）。

---

### 6. Phase 3 → P1-1 工程改进溢出（提取自实战）

本期 P1-1 把 demo 跑在真实业务 PDF 上时暴露了两个 chunker / 索引层的真实坑，已
在 chunker 和 build_index 里修复：

| 问题 | 现象 | 解决 |
|------|------|------|
| 页眉页脚噪声充斥 Top-K | NVIDIA PB 每页有 `"PB-11133-001_v02 \| <pageno>"` 短脚注，pdfplumber 拆成独立 chunk → 23 页产 23 个几乎一样的 noise chunk → 检索时反复占据头部 | `iter_chunks_jsonl(min_chars=60)` + `build_index.py --min_chars 60` 入参 |
| 通用表头 vs 答案行不可区分 | Table 1 / Table 2 同一页，表头都是 `\| Specification \| ... \|`，仅 caption 不同；表格行 chunk 丢了 caption 信息 | `chunker._table_to_chunks` 加 `caption` 参数，遍历表格上文找最近一行非空非表格内容作 caption，注入 chunk 文本头 + 元数据 `table_caption` |

---

## Phase 3 NPU 限制说明

**主线 Demo 使用 CPU / GPU / AUTO，不在 NPU 上跑完整链路**，原因：

1. **PaddleOCR-VL 的 PP-DocLayoutV3 IR 含 NPU 暂不支持的算子**——Phase 2 的 OCR
   路径上 NPU 会回落到 CPU，反而比直接走 CPU 慢；
2. Qwen3-1.7B-int4 与 Qwen3-Embedding-0.6B-int8 在 NPU 上理论可跑，但 NPU 共享
   系统内存且对 dynamic-shape 支持不全，2026.1 plugin 上仍有概率编译失败；
3. W4 加分项（P2）会专门尝试 "OCR-CPU / LLM-NPU / Embedding-NPU" 的拆分方案
   做 AI PC 全家桶叙事，时间不够就砍。

不要在 W3 主线上花时间硬试 V3 NPU——这是导师 2026-05-22 反馈后明确的方向收敛
（见 `docs/决策记录/2026-05-22_导师反馈方向校准.md`）。

---

## 已完成 Phase 汇总

| Phase | 内容 | 状态 |
|-------|------|:----:|
| 1 | 环境搭建 + 模型验证 + 推理 Benchmark（OV CPU/GPU 1.47×） | ✅ |
| 2 | 文档解析 + 表格感知切片（GB/T 2423.1 14 页 → 87 chunks） | ✅ |
| 3 | RAG 端到端问答（5 题 CPU ~3.3s/题，带 [doc p.页] 引用） | ✅ |
| P1-1 | eval 集 5 题真实业务 PDF 验证 + small-embedder 偏差分析 | ✅ |
| P1-2 | Tesseract vs PaddleOCR-VL 3 页对比（表格页相似度仅 9.55%） | ✅ |

## 提交

- OpenVINO demo 仓库 PR：[openvinotoolkit/openvino_build_deploy#552](https://github.com/openvinotoolkit/openvino_build_deploy/pull/552)
- 任务完成提交邮件草稿：[`../docs/任务完成提交.email.md`](../docs/任务完成提交.email.md)

## 加分项（时间允许）

- BGE-reranker 重排、NPU 路径尝试、Gradio UI、OmniDocBench 子集

详见 [`../docs/项目计划.md`](../docs/项目计划.md) 与 [`../TODO.md`](../TODO.md)。
