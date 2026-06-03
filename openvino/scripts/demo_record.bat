@echo off
chcp 65001 >nul
title Doc-QnA-OpenVINO Demo

REM ============================================================
REM  飞桨黑客松第10期 · 进阶任务 #13 演示脚本
REM  项目：基于 PaddleOCR-VL + OpenVINO 的产品文档智能问答系统
REM  用法：在 openvino 目录下运行，录屏软件先开，再双击此脚本
REM ============================================================

cd /d "%~dp0\.."

echo.
echo ============================================================
echo   飞桨黑客松第10期 · 进阶任务 #13
echo   基于 PaddleOCR-VL + OpenVINO 的产品文档智能问答系统
echo ============================================================
echo.
echo   技术栈：PaddleOCR-VL (OpenVINO) + Qwen3-Embedding-0.6B-int8
echo           + ChromaDB + Qwen3-1.7B-int4 (OpenVINO GenAI)
echo   平台：  CPU-only, Intel OpenVINO 加速
echo   GitHub: bob798
echo.
echo ============================================================
echo.
pause

REM ---------- 场景 1：构建向量索引 ----------
echo.
echo ============================================================
echo   场景 1/3：构建向量索引（4 份 PDF / 99 chunks）
echo ============================================================
echo.
echo ^> python scripts/build_index.py --chunks_dir results/phase2 --persist_dir chroma_db --device CPU --reset
echo.
python scripts/build_index.py --chunks_dir results/phase2 --persist_dir chroma_db --device CPU --reset
echo.
echo [索引构建完成]
echo.
pause

REM ---------- 场景 2：端到端问答（5 题 Demo） ----------
echo.
echo ============================================================
echo   场景 2/3：端到端 RAG 问答（5 条业务问题）
echo ============================================================
echo.
echo   问题覆盖：表格查询 / 额定功率 / 抗幻觉拒答 / 跨文档事实
echo.
echo ^> python scripts/run_qa.py --questions_file data/demo_questions.txt --persist_dir chroma_db --out results/phase3/demo_run.json --out_md results/phase3/demo_run.md
echo.
python scripts/run_qa.py --questions_file data/demo_questions.txt --persist_dir chroma_db --out results/phase3/demo_run.json --out_md results/phase3/demo_run.md
echo.
echo [问答完成]
echo.
pause

REM ---------- 场景 3：查看结构化结果报告 ----------
echo.
echo ============================================================
echo   场景 3/3：结构化结果报告 + 性能数据
echo ============================================================
echo.
type results\phase3\demo_run.md
echo.
echo ============================================================
echo   演示结束 · 感谢观看
echo ============================================================
echo.
pause
