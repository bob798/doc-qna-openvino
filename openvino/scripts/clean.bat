@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM ============================================================
REM  清理所有生成物和模型缓存
REM  用法：cd openvino && scripts\clean.bat
REM ============================================================

cd /d "%~dp0\.."

echo.
echo ============================================================
echo   清理生成物 + 模型缓存
echo ============================================================
echo.

if exist "chroma_db" (
    rmdir /s /q chroma_db
    echo   [OK] chroma_db/
)

if exist "results\phase3" (
    rmdir /s /q results\phase3
    echo   [OK] results/phase3/
)

if exist "results\demo_run.json" (
    del /q results\demo_run.json
    echo   [OK] results/demo_run.json
)

if exist "results\demo_run.md" (
    del /q results\demo_run.md
    echo   [OK] results/demo_run.md
)

if exist "build_index.summary.json" (
    del /q build_index.summary.json
    echo   [OK] build_index.summary.json
)

set "HF_CACHE=%USERPROFILE%\.cache\huggingface\hub"
if exist "%HF_CACHE%\models--OpenVINO--Qwen3-Embedding-0.6B-int8-ov" (
    rmdir /s /q "%HF_CACHE%\models--OpenVINO--Qwen3-Embedding-0.6B-int8-ov"
    echo   [OK] Qwen3-Embedding-0.6B-int8-ov 模型缓存
)
if exist "%HF_CACHE%\models--OpenVINO--Qwen3-1.7B-int4-ov" (
    rmdir /s /q "%HF_CACHE%\models--OpenVINO--Qwen3-1.7B-int4-ov"
    echo   [OK] Qwen3-1.7B-int4-ov 模型缓存
)

echo.
echo   清理完成
echo.
pause
endlocal
