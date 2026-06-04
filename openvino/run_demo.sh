#!/usr/bin/env bash
# 一键运行：安装依赖 + 端到端问答（单文件 main.py）
# 用法：cd openvino && bash run_demo.sh
set -e
cd "$(dirname "$0")"

export PYTHONIOENCODING=utf-8

echo
echo "============================================================"
echo "  飞桨黑客松第10期 · 进阶任务 #13"
echo "  基于 PaddleOCR-VL + OpenVINO 的产品文档智能问答系统"
echo "============================================================"
echo

echo "[1/2] 安装依赖..."
pip install -r requirements.txt -q
echo "[1/2] 依赖安装完成"
echo

echo "[2/2] 运行端到端 Demo..."
echo
python main.py
