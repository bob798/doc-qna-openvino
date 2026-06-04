#!/usr/bin/env bash
# Doc-QnA Demo - Linux/macOS 一键安装并运行
set -e

INSTALL_DIR="$PWD/doc-qna-openvino"

command -v git >/dev/null 2>&1 || { echo "ERROR: Git not installed."; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: Python3 not installed."; exit 1; }

# Clone
if [ ! -d "$INSTALL_DIR" ]; then
    echo "Cloning repository..."
    git clone https://github.com/bob798/doc-qna-openvino.git "$INSTALL_DIR"
else
    echo "Repository exists. Pulling latest..."
    cd "$INSTALL_DIR" && git pull
fi

cd "$INSTALL_DIR/openvino"

# Venv
if [ ! -f "venv/bin/activate" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# Install + Run
export PYTHONIOENCODING=utf-8
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo
echo "Running Doc-QnA Demo..."
python main.py
