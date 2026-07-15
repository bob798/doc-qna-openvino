# doc-qna-openvino

> Ask questions of your product manuals — and get a refusal when the answer isn't in them.

[中文文档](./README_CN.md)

Turns product manuals, spec sheets, and scanned PDFs into a knowledge base you can query in
natural language. Runs entirely on CPU/iGPU via OpenVINO — no GPU server, no API calls,
nothing leaves the machine.

---

## The pipeline

```
PDF / image
   │
   ▼
[ PDF preprocessing ]   text-layer detection (pdfplumber) + rendering (pypdfium2)
   │
   ▼
[ PaddleOCR-VL ]        OpenVINO-accelerated inference → structured Markdown
   │
   ▼
[ table-aware chunking ] header+row chunks · 200–500 char paragraphs · {doc, page, section} metadata
   │
   ▼
[ embed + retrieve ]    Qwen3-Embedding-0.6B INT8 (OpenVINO) → ChromaDB
   │
   ▼
[ 4-stage guard ]       entity gate → cross-encoder rerank → subject grounding → numeric grounding
   │
   ▼
[ generate ]            Qwen3 INT4 (OpenVINO GenAI) → answer with [doc p.N] citations
```

## The hard part isn't answering — it's refusing

A document Q&A system that answers everything is worse than useless: ask it about a Mars rover
and it will confidently answer using fields from your HVAC spec sheet. Naive similarity
thresholds don't catch this — the out-of-domain questions scored `0.237–0.465` against an
in-domain floor of `0.722`, close enough that no single cutoff separates them.

So refusal is enforced by four independent guards:

| # | Guard | Stage | What it catches |
|---|---|---|---|
| 1 | **Entity gate** (deterministic) | pre-LLM | Plausible-looking entities absent from the corpus (model `A500`, standard `GB/T 9999`) → refuse |
| 2 | **Cross-encoder rerank** (`bge-reranker-base-int8-ov`) | pre-LLM | Joint (query, chunk) encoding sees subject mismatch: Mars rover `0.043`, Everest `0.006` vs in-domain `0.98+` |
| 3 | **Subject grounding** | pre-LLM | Reranker's blind spot — strong field matches still score high (Tesla Model 3 `0.770`, above a weakly-retrieved in-domain question at `0.641`). If the query's subject appears in no evidence chunk → refuse |
| 4 | **Numeric grounding** | post-generation | Small models inventing a plausible date/number when retrieval found nothing. Every date and figure in the answer must be verifiable against retrieved evidence |

**Measured** (`scripts/eval_guard.py`, 5 in-domain + 7 out-of-domain/cross-talk questions):

| | In-domain answered correctly | Out-of-domain correctly refused |
|---|---|---|
| Similarity threshold only (`--min_score 0.35`) | 5/5 | **2/7** |
| 4-stage guard | 5/5 | **7/7** |

Reranking widened in-domain vs. out-of-domain separation from an overlapping
`[0.722, 0.840]` vs `[0.237, 0.465]` to `[0.641, 0.997]` vs `[0.000, 0.770]`; guards 1, 3, and 4
cover the reversals reranking alone still gets wrong. Ablate with `--no_reranker` / `--no_entity_check`.

## Results

| What | Measured on |
|---|---|
| GPU/CPU inference speedup **1.47×** | OpenVINO CPU vs iGPU benchmark |
| **87 chunks** from a real 14-page national standard | GB/T 2423.1 |
| **372 chunks** from 46 pages of real spec sheets | NVIDIA H100 PCIe + H100 NVL Product Briefs |
| **~3.3 s/question** on CPU, answers carry `[doc p.N]` citations | Phase 3 end-to-end |
| OCR quality comparison vs. Tesseract | 3 representative pages + 5-question eval set |

## Quick start

```bash
git clone https://github.com/bob798/doc-qna-openvino
cd doc-qna-openvino/openvino
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Prepare the PaddleOCR-VL OpenVINO IR — see openvino/README.md § 准备工作
# 2. Benchmark inference
python scripts/benchmark_inference.py --image_dir data/test_images --device CPU
# 3. Parse documents + table-aware chunking
python scripts/run_phase2_pipeline.py --pdf_dir data/test_documents --out results/phase2
# 4. Build the index and run RAG Q&A
python scripts/build_index.py --chunks_dir results/phase2 --persist_dir chroma_db --device CPU --reset
python scripts/run_qa.py --questions_file data/demo_questions.txt --persist_dir chroma_db --out results/phase3/qa_run.json

# Evaluate the guards
python scripts/eval_guard.py
```

Full setup notes (including Windows environment variables) in [`openvino/README.md`](openvino/README.md).

## Stack

Python · OpenVINO / OpenVINO GenAI · PaddleOCR-VL · Qwen3 INT4 · Qwen3-Embedding-0.6B INT8 ·
bge-reranker-base INT8 · ChromaDB · pdfplumber · pypdfium2

## Context

Built for the **PaddlePaddle Hackathon #10** (ERNIE partner track, advanced task #13:
multimodal document understanding with OpenVINO). Upstream contribution:
[openvinotoolkit/openvino_build_deploy#552](https://github.com/openvinotoolkit/openvino_build_deploy/pull/552).

The guard layer was hardened through code review against 10 confirmed defects — see
[`README_CN.md`](./README_CN.md) for the full development log, weekly reports, and Chinese documentation.

## Links

- Track overview: https://github.com/PaddlePaddle/Paddle/issues/78485
- PaddleOCR-VL: https://huggingface.co/PaddlePaddle/PaddleOCR-VL
- OpenVINO notebook (PaddleOCR-VL): https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/paddleocr_vl
