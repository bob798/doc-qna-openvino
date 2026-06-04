#!/usr/bin/env bash
# Doc-QnA Demo - Linux/macOS Runner
set -e

INSTALL_DIR="$PWD/doc-qna-openvino/openvino"

cd "$INSTALL_DIR"

# Check venv
if [ ! -f "venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found! Please run install.sh first."
    exit 1
fi

# Activate
echo "Activating virtual environment..."
source venv/bin/activate

export PYTHONIOENCODING=utf-8

# Run demo
echo "Running Doc-QnA Demo..."
python main.py
