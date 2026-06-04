#!/usr/bin/env bash
# ============================================================
#  一键运行脚本 · 基于 PaddleOCR-VL + OpenVINO 的产品文档智能问答
#  飞桨黑客松第10期 · 进阶任务 #13
#
#  用法：
#    cd openvino
#    bash run_demo.sh              （默认：安装依赖 + 构建索引 + 问答）
#    bash run_demo.sh --skip-install （跳过 pip install）
# ============================================================

set -e
cd "$(dirname "$0")"

SKIP_INSTALL=0
[[ "$1" == "--skip-install" ]] && SKIP_INSTALL=1

export PYTHONIOENCODING=utf-8

echo
echo "============================================================"
echo "  飞桨黑客松第10期 · 进阶任务 #13"
echo "  基于 PaddleOCR-VL + OpenVINO 的产品文档智能问答系统"
echo "============================================================"
echo
echo "  技术栈: PaddleOCR-VL (OpenVINO) + Qwen3-Embedding-0.6B-int8"
echo "          + ChromaDB + Qwen3-1.7B-int4 (OpenVINO GenAI)"
echo "  全链路: PDF 解析 → 表格感知切片 → 向量检索 → LLM 生成（带引用）"
echo

# Step 1
if [[ $SKIP_INSTALL -eq 1 ]]; then
    echo "[1/3] 跳过依赖安装 (--skip-install)"
else
    echo "[1/3] 安装依赖..."
    pip install -r requirements.txt -q
    echo "[1/3] 依赖安装完成"
fi
echo

# Step 2
echo "[2/3] 构建向量索引（4 份 PDF / 99 chunks → ChromaDB）..."
echo "     模型: Qwen3-Embedding-0.6B-int8 (首次运行自动下载 ~600MB)"
echo
python scripts/build_index.py \
    --chunks_dir results/phase2 \
    --persist_dir chroma_db \
    --device CPU --reset
echo
echo "[2/3] 索引构建完成"
echo

# Step 3
echo "[3/3] 端到端 RAG 问答（5 条业务问题）..."
echo "     模型: Qwen3-1.7B-int4 (首次运行自动下载 ~1GB)"
echo
python scripts/run_qa.py \
    --questions_file data/demo_questions.txt \
    --persist_dir chroma_db \
    --out results/phase3/demo_run.json \
    --out_md results/phase3/demo_run.md
echo

# Results
echo "============================================================"
echo "  结果报告"
echo "============================================================"
echo
cat results/phase3/demo_run.md
echo
echo "============================================================"
echo "  演示完成"
echo "  结果文件: results/phase3/demo_run.json"
echo "  报告文件: results/phase3/demo_run.md"
echo "============================================================"
