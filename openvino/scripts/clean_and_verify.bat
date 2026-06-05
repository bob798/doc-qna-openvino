@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM ============================================================
REM  清理本地环境 + 模拟评委复现流程
REM  用法：cd openvino && scripts\clean_and_verify.bat
REM ============================================================

cd /d "%~dp0\.."

echo.
echo ============================================================
echo   模拟评委复现流程（清理 → 一键运行）
echo ============================================================
echo.

REM ---------- Step 1: 清理生成物 ----------
echo [1/3] 清理本地生成物...

if exist "chroma_db" (
    rmdir /s /q chroma_db
    echo   已删除 chroma_db/
)

if exist "results\phase3" (
    rmdir /s /q results\phase3
    echo   已删除 results/phase3/
)

if exist "results\demo_run.json" (
    del /q results\demo_run.json
    echo   已删除 results/demo_run.json
)

if exist "results\demo_run.md" (
    del /q results\demo_run.md
    echo   已删除 results/demo_run.md
)

if exist "build_index.summary.json" (
    del /q build_index.summary.json
    echo   已删除 build_index.summary.json
)

REM 清理 HuggingFace 模型缓存（Embedding + LLM）
echo   清理 HuggingFace 模型缓存...
set "HF_CACHE=%USERPROFILE%\.cache\huggingface\hub"
if exist "%HF_CACHE%\models--OpenVINO--Qwen3-Embedding-0.6B-int8-ov" (
    rmdir /s /q "%HF_CACHE%\models--OpenVINO--Qwen3-Embedding-0.6B-int8-ov"
    echo   已删除 Qwen3-Embedding-0.6B-int8-ov 缓存
)
if exist "%HF_CACHE%\models--OpenVINO--Qwen3-1.7B-int4-ov" (
    rmdir /s /q "%HF_CACHE%\models--OpenVINO--Qwen3-1.7B-int4-ov"
    echo   已删除 Qwen3-1.7B-int4-ov 缓存
)

echo   清理完成
echo.

REM ---------- Step 2: 确认源数据存在 ----------
echo [2/3] 检查源数据（不应被清理）...

set OK=1
if not exist "results\phase2\spec_with_tables.chunks.jsonl" (
    echo   [FAIL] results\phase2\spec_with_tables.chunks.jsonl 缺失
    set OK=0
)
if not exist "data\demo_questions.txt" (
    echo   [FAIL] data\demo_questions.txt 缺失
    set OK=0
)
if not exist "main.py" (
    echo   [FAIL] main.py 缺失
    set OK=0
)
if not exist "requirements.txt" (
    echo   [FAIL] requirements.txt 缺失
    set OK=0
)

if !OK!==0 (
    echo   源数据不完整，请检查 git 仓库
    goto :end
)
echo   源数据完整
echo.

REM ---------- Step 3: 一键运行（模拟评委） ----------
echo [3/3] 模拟评委操作: python main.py
echo.
echo ============================================================

python main.py

echo.
echo ============================================================
echo   复现完成！检查上方输出是否正常：
echo   - 5 题问答全部有回答
echo   - 每题有 [doc_name p.page] 引用
echo   - 性能数据（embed/retrieve/llm/total/tps）
echo ============================================================

:end
echo.
pause
endlocal
