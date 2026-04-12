# 飞桨黑客松 - OpenVINO Notebooks macOS 安装打卡

## 任务说明

按照 [macOS 安装文档](https://github.com/openvinotoolkit/openvino_notebooks/wiki/macOS) 要求，在 macOS 上安装 openvino_notebooks 并成功运行 `paddleocr_vl` notebook。

---

## 环境信息

| 项目 | 版本 |
|------|------|
| OS | macOS 12.0 (Darwin 21.1.0) x86_64 |
| Python | 3.12.4 |
| OpenVINO | 2025.4.1 |
| torch | 2.2.2 |
| transformers | 4.54.0 |
| nncf | 2.14.1 |
| JupyterLab | 4.5.6 |

---

## 安装步骤

### 1. 检查前置依赖

```bash
xcode-select -p       # /Library/Developer/CommandLineTools
python3 --version     # Python 3.12.4
brew --version        # Homebrew 5.0.11
```

### 2. 安装系统依赖

```bash
brew install protobuf
brew install ffmpeg
```

### 3. 克隆仓库

```bash
git clone --depth=1 https://github.com/openvinotoolkit/openvino_notebooks.git
cd openvino_notebooks
```

### 4. 创建虚拟环境

```bash
python3 -m venv openvino_env
source openvino_env/bin/activate
```

### 5. 安装依赖

```bash
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install openvino
python -m ipykernel install --user --name openvino_env
```

### 6. 验证安装

```bash
python check_install.py
# Everything looks good!
```

### 7. 安装 notebook 专用依赖（macOS x86_64 适配）

```bash
# torch 2.8.0 不支持 macOS x86_64，使用 2.2.2
pip install torch torchvision torchaudio
pip install "numpy<2.0"   # macOS 兼容性
pip install "opencv-python-headless==4.10.0.84"
pip install "transformers==4.54.0" "nncf==2.14.1" "gradio==4.19" \
    "modelscope" "huggingface-hub" "sentencepiece" "einops" \
    "nncf==2.14.1" "datasets" "protobuf" "fastapi" "uvicorn" \
    "httpx[socks]" "socksio"
```

---

## 运行结果

### Notebook: `paddleocr_vl/paddleocr_vl.ipynb`

**模型**：PaddleOCR-VL-1.5（0.9B 超轻量视觉语言模型）

**推理任务**：OCR 文字识别

**推理结果**：

```
============================================================
📄 CPU OpenVINO ocr result:
============================================================
PaddleOCR-VL-1.5 is an advanced next-generation model of PaddleOCR-VL,
achieving a new state-of-the-art accuracy of 94.5% on OmniDocBench v1.5.
To rigorously evaluate robustness against real-world physical distortions—
including scanning artifacts, skew, warping, screen photography, and
illumination—we propose the Real5-OmniDocBench benchmark...
============================================================
```

> 截图：[待补充]

---

## 打卡提交方式

### 发送邮件至：
- ext_paddle_oss@baidu.com
- zhuo.wu@intel.com
- ethan.yang@intel.com

### 邮件主题格式：
```
文心伙伴赛道-intel-打卡-【你的GithubID】
```
例：`文心伙伴赛道-intel-打卡-onecatcn`

### 邮件正文需包含：
1. GitHub ID
2. Notebook 路径：`notebooks/paddleocr_vl/paddleocr_vl.ipynb`
3. 环境信息（已整理在上方表格）
4. 截图（见下方待办）
5. 可选：30-60 秒演示视频

---

## 打卡邮件正文

**收件人：** ext_paddle_oss@baidu.com; zhuo.wu@intel.com; ethan.yang@intel.com

**主题：** `文心伙伴赛道-intel-打卡-bob798`

**正文：**

```
Hi，

我已完成 OpenVINO Notebook 快速上手打卡任务，信息如下：

GitHub ID：bob798
Notebook：notebooks/paddleocr_vl/paddleocr_vl.ipynb

环境信息：
- OS：macOS 12.0 (Darwin 21.1.0) x86_64
- CPU：Intel x86_64
- Python：3.12.4
- OpenVINO：2025.4.1
- JupyterLab：4.5.6

附截图：
1. 依赖安装成功（✅ Dependencies installed.）
2. 推理运行日志及 OCR 识别结果（📄 CPU OpenVINO ocr result）

感谢！
bob798
```

## 后续待办

- [x] 截图1：依赖安装成功（`✅ Dependencies installed.`）
- [x] 截图2+3：推理日志及 OCR 识别结果（`📄 CPU OpenVINO ocr result`）
- [ ] 按上方格式发送打卡邮件（附截图）
- [ ] 等待回复确认
