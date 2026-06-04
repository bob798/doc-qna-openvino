@echo off
chcp 65001 >nul
setlocal
title Doc-QnA-OpenVINO · 一键演示

REM ============================================================
REM  一键运行脚本 · 基于 PaddleOCR-VL + OpenVINO 的产品文档智能问答
REM  飞桨黑客松第10期 · 进阶任务 #13
REM
REM  用法：
REM    cd openvino
REM    run_demo.bat              （默认：安装依赖 + 构建索引 + 问答）
REM    run_demo.bat --skip-install （跳过 pip install，适合已装好依赖）
REM ============================================================

cd /d "%~dp0"

set SKIP_INSTALL=0
if "%1"=="--skip-install" set SKIP_INSTALL=1

REM ---------- 环境变量 ----------
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo.
echo ============================================================
echo   飞桨黑客松第10期 · 进阶任务 #13
echo   基于 PaddleOCR-VL + OpenVINO 的产品文档智能问答系统
echo ============================================================
echo.
echo   技术栈: PaddleOCR-VL (OpenVINO) + Qwen3-Embedding-0.6B-int8
echo           + ChromaDB + Qwen3-1.7B-int4 (OpenVINO GenAI)
echo   全链路: PDF 解析 → 表格感知切片 → 向量检索 → LLM 生成（带引用）
echo   GitHub: bob798
echo.

REM ---------- Step 1: 安装依赖 ----------
if %SKIP_INSTALL%==1 (
    echo [1/3] 跳过依赖安装 (--skip-install)
) else (
    echo [1/3] 安装依赖...
    pip install -r requirements.txt -q
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请检查 Python 环境
        goto :end
    )
    echo [1/3] 依赖安装完成
)
echo.

REM ---------- Step 2: 构建索引 ----------
echo [2/3] 构建向量索引（4 份 PDF / 99 chunks → ChromaDB）...
echo      模型: Qwen3-Embedding-0.6B-int8 (首次运行自动下载 ~600MB)
echo.
python scripts/build_index.py --chunks_dir results/phase2 --persist_dir chroma_db --device CPU --reset
if errorlevel 1 (
    echo [错误] 索引构建失败
    goto :end
)
echo.
echo [2/3] 索引构建完成
echo.

REM ---------- Step 3: 端到端问答 ----------
echo [3/3] 端到端 RAG 问答（5 条业务问题）...
echo      模型: Qwen3-1.7B-int4 (首次运行自动下载 ~1GB)
echo      问题覆盖: 表格查询 / 额定功率 / 抗幻觉拒答 / 跨文档事实
echo.
python scripts/run_qa.py --questions_file data/demo_questions.txt --persist_dir chroma_db --out results/phase3/demo_run.json --out_md results/phase3/demo_run.md
if errorlevel 1 (
    echo [错误] 问答执行失败
    goto :end
)
echo.

REM ---------- 结果展示 ----------
echo ============================================================
echo   结果报告
echo ============================================================
echo.
type results\phase3\demo_run.md
echo.
echo ============================================================
echo   演示完成
echo   结果文件: results\phase3\demo_run.json
echo   报告文件: results\phase3\demo_run.md
echo ============================================================

:end
echo.
pause
endlocal
