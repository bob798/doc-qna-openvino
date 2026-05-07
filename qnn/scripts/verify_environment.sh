#!/bin/bash
# 环境验证脚本 — 进入容器后首先运行
# 用法: bash scripts/verify_environment.sh

set -e
PASS=0
FAIL=0
WARN=0

check() {
    local name="$1"
    shift
    if "$@" > /dev/null 2>&1; then
        echo "  [PASS] $name"
        ((PASS++))
    else
        echo "  [FAIL] $name"
        ((FAIL++))
    fi
}

warn_check() {
    local name="$1"
    shift
    if "$@" > /dev/null 2>&1; then
        echo "  [PASS] $name"
        ((PASS++))
    else
        echo "  [WARN] $name (可选，不影响核心功能)"
        ((WARN++))
    fi
}

echo "============================================"
echo " PaddleOCR-VL QNN 部署环境验证"
echo "============================================"
echo ""

# ── 1. QNN SDK ──
echo "[1/5] QNN SDK"
echo "  QNN_SDK_ROOT=$QNN_SDK_ROOT"

check "SDK 目录存在" test -d "$QNN_SDK_ROOT"
check "qnn-onnx-converter" test -f "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-onnx-converter"
check "qnn-model-lib-generator" test -f "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator"
check "qnn-context-binary-generator" test -f "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator"
check "qnn-net-run" test -f "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-net-run"
check "libQnnHtp.so" test -f "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so"
check "libQnnCpu.so" test -f "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnCpu.so"
check "libQnnSystem.so" test -f "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnSystem.so"
echo ""

# ── 2. Python 环境 ──
echo "[2/5] Python 环境"
check "Python3" python3 --version
check "pip" pip --version
check "numpy" python3 -c "import numpy; print(f'  numpy {numpy.__version__}')"
check "Pillow" python3 -c "from PIL import Image; import PIL; print(f'  Pillow {PIL.__version__}')"
check "onnx" python3 -c "import onnx; print(f'  onnx {onnx.__version__}')"
check "paddle2onnx" python3 -c "import paddle2onnx; print(f'  paddle2onnx {paddle2onnx.__version__}')"
echo ""

# ── 3. PaddlePaddle + PaddleOCR ──
echo "[3/5] PaddlePaddle + PaddleOCR"
check "paddlepaddle" python3 -c "import paddle; print(f'  paddle {paddle.__version__}')"
check "paddleocr" python3 -c "import paddleocr; print(f'  paddleocr {paddleocr.__version__}')"
warn_check "transformers" python3 -c "import transformers; print(f'  transformers {transformers.__version__}')"
warn_check "optimum" python3 -c "import optimum; print(f'  optimum {optimum.__version__}')"
echo ""

# ── 4. QNN 工具链可执行 ──
echo "[4/5] QNN 工具链可执行性"
# qnn-onnx-converter 是 Python 脚本，测试 --help
if "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-onnx-converter" --help > /dev/null 2>&1; then
    echo "  [PASS] qnn-onnx-converter --help"
    ((PASS++))
else
    echo "  [FAIL] qnn-onnx-converter --help (可能缺少 Python 依赖)"
    ((FAIL++))
fi

# qnn-net-run 是二进制，测试可执行
if "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-net-run" --help > /dev/null 2>&1; then
    echo "  [PASS] qnn-net-run --help"
    ((PASS++))
elif ldd "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-net-run" > /dev/null 2>&1; then
    echo "  [WARN] qnn-net-run 二进制存在但可能缺少动态库"
    ((WARN++))
else
    echo "  [FAIL] qnn-net-run"
    ((FAIL++))
fi
echo ""

# ── 5. 项目文件 ──
echo "[5/5] 项目文件完整性"
check "export_paddle_to_onnx.py" test -f scripts/export_paddle_to_onnx.py
check "convert_onnx_to_qnn.py" test -f scripts/convert_onnx_to_qnn.py
check "quantize_models.py" test -f scripts/quantize_models.py
check "evaluate_accuracy.py" test -f scripts/evaluate_accuracy.py
check "benchmark_performance.py" test -f scripts/benchmark_performance.py
check "qnn_inference.py" test -f src/qnn_inference.py
check "doc_parser.py" test -f src/doc_parser.py
echo ""

# ── 汇总 ──
echo "============================================"
echo " 结果: $PASS 通过 / $FAIL 失败 / $WARN 警告"
echo "============================================"

if [ $FAIL -eq 0 ]; then
    echo ""
    echo " 环境就绪！下一步:"
    echo "   1. python scripts/export_paddle_to_onnx.py --download"
    echo "   2. python scripts/export_paddle_to_onnx.py --model_type layout"
    echo "   3. python scripts/convert_onnx_to_qnn.py --check_sdk"
    echo ""
else
    echo ""
    echo " 有 $FAIL 项检查未通过，请先修复后再继续"
    echo ""
    exit 1
fi
