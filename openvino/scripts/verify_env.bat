@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM ============================================================
REM  环境验证脚本 · 模拟评委从 git clone 到端到端问答的完整流程
REM  用法：在 openvino 目录下，激活 venv 后运行此脚本
REM ============================================================

cd /d "%~dp0\.."

set PASS=0
set FAIL=0
set WARN=0

echo.
echo ============================================================
echo   环境验证 · doc-qna-openvino
echo   验证时间: %date% %time%
echo ============================================================
echo.

REM ---------- Step 1: Python 环境 ----------
echo [1/8] 检查 Python 环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo   [FAIL] Python 未安装或不在 PATH 中
    set /a FAIL+=1
    goto :end
) else (
    for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYVER=%%i
    echo   [PASS] !PYVER!
    set /a PASS+=1
)
echo.

REM ---------- Step 2: 关键依赖 ----------
echo [2/8] 检查关键 Python 依赖...
set DEPS_OK=1

for %%d in (openvino openvino_genai chromadb huggingface_hub transformers pypdfium2 pdfplumber) do (
    python -c "import %%d" >nul 2>&1
    if errorlevel 1 (
        echo   [FAIL] %%d 未安装
        set DEPS_OK=0
        set /a FAIL+=1
    ) else (
        echo   [PASS] %%d
        set /a PASS+=1
    )
)

if !DEPS_OK!==0 (
    echo.
    echo   修复方法: pip install -r requirements.txt
)
echo.

REM ---------- Step 3: Windows 环境变量 ----------
echo [3/8] 检查 Windows 环境变量...
if not defined PYTHONIOENCODING (
    echo   [WARN] PYTHONIOENCODING 未设置，中文输出可能乱码
    echo         修复: set PYTHONIOENCODING=utf-8
    set /a WARN+=1
) else (
    echo   [PASS] PYTHONIOENCODING=!PYTHONIOENCODING!
    set /a PASS+=1
)
echo.

REM ---------- Step 4: PaddleOCR-VL 模型 ----------
echo [4/8] 检查 PaddleOCR-VL OpenVINO IR...
if exist "models\paddleocr_vl_ov\openvino_model.xml" (
    echo   [PASS] models\paddleocr_vl_ov\openvino_model.xml 存在
    set /a PASS+=1
) else (
    echo   [WARN] PaddleOCR-VL IR 未找到（Phase 2 文档解析需要）
    echo         参考 README.md 第 2 节准备模型
    echo         注意：如果 results\phase2 已有 chunks 文件，可跳过此步直接跑 Phase 3
    set /a WARN+=1
)
echo.

REM ---------- Step 5: 测试数据 ----------
echo [5/8] 检查测试数据...
set DATA_OK=1

if exist "data\test_documents\text_pdf.pdf" (
    echo   [PASS] data\test_documents\text_pdf.pdf
    set /a PASS+=1
) else (
    echo   [FAIL] data\test_documents\text_pdf.pdf 缺失
    set DATA_OK=0
    set /a FAIL+=1
)

if exist "data\test_documents\scanned.pdf" (
    echo   [PASS] data\test_documents\scanned.pdf
    set /a PASS+=1
) else (
    echo   [FAIL] data\test_documents\scanned.pdf 缺失
    set DATA_OK=0
    set /a FAIL+=1
)

if exist "data\test_documents\spec_with_tables.pdf" (
    echo   [PASS] data\test_documents\spec_with_tables.pdf
    set /a PASS+=1
) else (
    echo   [FAIL] data\test_documents\spec_with_tables.pdf 缺失
    set DATA_OK=0
    set /a FAIL+=1
)

if exist "data\demo_questions.txt" (
    echo   [PASS] data\demo_questions.txt
    set /a PASS+=1
) else (
    echo   [FAIL] data\demo_questions.txt 缺失
    set /a FAIL+=1
)
echo.

REM ---------- Step 6: Phase 2 结果（chunks） ----------
echo [6/8] 检查 Phase 2 切片结果...
set CHUNKS_FOUND=0
for %%f in (results\phase2\*.chunks.jsonl) do (
    set /a CHUNKS_FOUND+=1
)

if !CHUNKS_FOUND! GEQ 1 (
    echo   [PASS] 找到 !CHUNKS_FOUND! 个 chunks.jsonl 文件
    set /a PASS+=1
) else (
    echo   [WARN] results\phase2\ 下无 chunks.jsonl 文件
    echo         需先运行 Phase 2:
    echo         python scripts/run_phase2_pipeline.py --pdf_dir data/test_documents --out results/phase2
    set /a WARN+=1
)
echo.

REM ---------- Step 7: Phase 3 端到端验证 ----------
echo [7/8] Phase 3 端到端验证（构建索引 + 问答）...

if !CHUNKS_FOUND! GEQ 1 (
    echo.
    echo   --- 7a: 构建 ChromaDB 索引 ---
    set PYTHONIOENCODING=utf-8
    set HF_HUB_DISABLE_SYMLINKS=1
    set HF_HUB_DISABLE_SYMLINKS_WARNING=1

    python scripts/build_index.py --chunks_dir results/phase2 --persist_dir chroma_db_verify --device CPU --reset
    if errorlevel 1 (
        echo   [FAIL] build_index.py 执行失败
        set /a FAIL+=1
    ) else (
        echo   [PASS] 索引构建成功
        set /a PASS+=1

        echo.
        echo   --- 7b: 端到端问答（5 题） ---
        python scripts/run_qa.py --questions_file data/demo_questions.txt --persist_dir chroma_db_verify --out results/phase3/verify_run.json --out_md results/phase3/verify_run.md
        if errorlevel 1 (
            echo   [FAIL] run_qa.py 执行失败
            set /a FAIL+=1
        ) else (
            echo   [PASS] 端到端问答成功
            set /a PASS+=1

            echo.
            echo   --- 验证输出文件 ---
            if exist "results\phase3\verify_run.json" (
                echo   [PASS] results\phase3\verify_run.json 已生成
                set /a PASS+=1
            ) else (
                echo   [FAIL] verify_run.json 未生成
                set /a FAIL+=1
            )
            if exist "results\phase3\verify_run.md" (
                echo   [PASS] results\phase3\verify_run.md 已生成
                set /a PASS+=1
            ) else (
                echo   [FAIL] verify_run.md 未生成
                set /a FAIL+=1
            )
        )
    )

    REM 清理验证用的 chroma_db
    if exist "chroma_db_verify" (
        rmdir /s /q chroma_db_verify
        echo   [INFO] 已清理验证用 chroma_db_verify 目录
    )
) else (
    echo   [SKIP] 无 Phase 2 chunks，跳过端到端验证
    echo         请先完成 Phase 2 或准备 PaddleOCR-VL 模型
)
echo.

REM ---------- Step 8: 磁盘空间 ----------
echo [8/8] 检查磁盘空间...
for /f "tokens=3" %%a in ('dir /-c "%cd%" ^| findstr "bytes free"') do (
    set FREE_BYTES=%%a
)
echo   可用空间: !FREE_BYTES! bytes
echo   建议预留: 8 GB（模型 4.3 GB + venv + ChromaDB）
echo.

REM ---------- 汇总 ----------
echo ============================================================
echo   验证汇总
echo ============================================================
echo   PASS: !PASS!   FAIL: !FAIL!   WARN: !WARN!
echo.

if !FAIL! GTR 0 (
    echo   [结果] 存在 !FAIL! 项失败，请修复后重试
) else if !WARN! GTR 0 (
    echo   [结果] 全部通过，!WARN! 项警告（不阻塞演示）
) else (
    echo   [结果] 全部通过，环境就绪！
)
echo ============================================================
echo.

:end
pause
endlocal
