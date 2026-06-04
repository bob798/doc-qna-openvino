#!/usr/bin/env bash
# Doc-QnA Demo - Linux/macOS Installer
set -e

INSTALL_DIR="$PWD/doc-qna-openvino"

# Check dependencies
command -v git >/dev/null 2>&1 || { echo "ERROR: Git is not installed."; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: Python3 is not installed."; exit 1; }

# Clone repository
if [ ! -d "$INSTALL_DIR" ]; then
    echo "Cloning repository..."
    git clone https://github.com/bob798/doc-qna-openvino.git "$INSTALL_DIR"
else
    echo "Repository already exists. Pulling latest..."
    cd "$INSTALL_DIR" && git pull
fi

# Navigate to demo directory
cd "$INSTALL_DIR/openvino"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo
echo "========================================"
echo "All requirements installed for Doc-QnA Demo."
echo "You can now run the demo!"
echo "========================================"
