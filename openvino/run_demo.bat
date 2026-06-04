@echo off
chcp 65001 >nul
setlocal
title Doc-QnA-OpenVINO · 一键演示

REM ============================================================
REM  一键运行：安装依赖 + 端到端问答（单文件 main.py）
REM  用法：cd openvino && run_demo.bat
REM ============================================================

cd /d "%~dp0"

set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo.
echo ============================================================
echo   飞桨黑客松第10期 · 进阶任务 #13
echo   基于 PaddleOCR-VL + OpenVINO 的产品文档智能问答系统
echo ============================================================
echo.

REM ---------- 安装依赖 ----------
echo [1/2] 安装依赖...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查 Python 环境
    goto :end
)
echo [1/2] 依赖安装完成
echo.

REM ---------- 运行 main.py ----------
echo [2/2] 运行端到端 Demo...
echo.
python main.py
if errorlevel 1 (
    echo [错误] Demo 运行失败
    goto :end
)

:end
echo.
pause
endlocal
