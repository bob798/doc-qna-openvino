@echo off
chcp 65001 >nul

REM ============================================================
REM  提交 Phase 2 切片结果到 git
REM  用法：在项目根目录 (doc-qna-openvino) 下运行
REM ============================================================

cd /d "%~dp0\..\.."

echo.
echo [1/5] 修改 .gitignore，允许 chunks.jsonl 和 summary.json 被跟踪...
powershell -Command "$c = Get-Content .gitignore; $c = $c -replace '^openvino/results/phase2/\*\.chunks\.jsonl$', '#openvino/results/phase2/*.chunks.jsonl'; $c = $c -replace '^openvino/results/phase2/\*\.summary\.json$', '#openvino/results/phase2/*.summary.json'; Set-Content .gitignore $c"
echo    done.

echo.
echo [2/5] 暂存文件...
git add .gitignore
git add -f openvino/results/phase2/*.chunks.jsonl
git add -f openvino/results/phase2/*.summary.json
echo    done.

echo.
echo [3/5] 查看暂存状态...
git status

echo.
echo [4/5] 提交...
git commit -m "提交 Phase 2 切片结果：评委可跳过模型准备直接跑 Phase 3 演示"

echo.
echo [5/5] 推送到远程...
git push

echo.
echo ============================================================
echo   完成！Phase 2 结果已推送到 GitHub
echo ============================================================
pause
