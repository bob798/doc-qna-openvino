# 基于 Qualcomm AI Engine Direct (QNN) 部署 PaddleOCR-VL 模型

> 飞桨黑客松第 10 期 · 高通赛题
> GitHub ID: bob798

实现端侧页面级文档解析：将 PaddleOCR-VL pipeline 中的关键子模型（布局检测模型、VL 识别模型）通过 QNN 工具链转换并部署至高通 Hexagon NPU 进行端侧推理。

## 项目结构

```
qnn/
├── README.md                           # 本文件
├── docs/
│   └── 模型转换指南.md                  # Paddle → ONNX → QNN 全链路说明
├── scripts/
│   ├── export_paddle_to_onnx.py        # Step 1: Paddle → ONNX 导出
│   ├── convert_onnx_to_qnn.py          # Step 2: ONNX → QNN 转换 (全链路)
│   ├── quantize_models.py              # Step 3: 模型量化 (INT8/FP16)
│   ├── evaluate_accuracy.py            # 精度对比评测
│   └── benchmark_performance.py        # 性能耗时测试
├── src/
│   ├── qnn_inference.py                # QNN 端侧推理服务
│   └── doc_parser.py                   # 页面级文档解析 Pipeline
├── configs/                            # HTP 后端配置
├── data/
│   └── calibration/                    # 量化校准数据
└── reports/                            # 评测报告
```

## 环境要求

### 硬件
- **开发机**: x86_64 Linux (Ubuntu 20.04+)
- **推理设备**: 高通骁龙平台 (Hexagon NPU) 或 HTP-simulator

### 软件
| 依赖 | 版本 | 用途 |
|------|------|------|
| QNN SDK (QAIRT) | 2.x+ | 模型转换和推理 |
| Python | 3.8+ | 脚本运行 |
| PaddlePaddle | 2.6+ | 基线推理 |
| PaddleOCR | 3.x | 模型下载和基线对比 |
| Paddle2ONNX | 1.2+ | Paddle → ONNX 转换 |
| ONNX | 1.14+ | 模型验证 |
| NumPy | 1.24+ | 数据处理 |
| Pillow | 10.0+ | 图片处理 |
| transformers | 4.40+ | VL 模型 HuggingFace 导出 (路径 B) |

## 快速开始

### 1. 环境安装

```bash
# 安装 Python 依赖
pip install paddlepaddle paddleocr paddle2onnx onnx numpy Pillow

# 安装 QNN SDK (需先从 https://aihub.qualcomm.com 下载)
export QNN_SDK_ROOT=/path/to/qairt/<version>
source $QNN_SDK_ROOT/bin/envsetup.sh
```

### 2. 下载模型

```bash
# 查看下载指引
python scripts/export_paddle_to_onnx.py --download
```

### 3. 模型转换全链路

```bash
# Step 1: Paddle → ONNX
python scripts/export_paddle_to_onnx.py --model_type all --model_dir ./models/paddle --output_dir ./models/onnx

# Step 2: ONNX → QNN (含 HTP 上下文二进制生成)
python scripts/convert_onnx_to_qnn.py --input_dir ./models/onnx --output_dir ./models/qnn --backend htp

# Step 3: 量化 (可选，提升 HTP 性能)
python scripts/quantize_models.py calibrate --image_dir ./data/calibration_images --output_dir ./data/calibration
python scripts/quantize_models.py quantize --model all --onnx_dir ./models/onnx --output_dir ./models/qnn
```

### 4. 运行文档解析

```bash
# 单页解析
python src/doc_parser.py --image test.png --backend htp

# 批量解析
python src/doc_parser.py --image_dir ./data/test_images --output_dir ./results --backend htp
```

### 5. 评测

```bash
# 基线推理
python scripts/evaluate_accuracy.py baseline --image_dir ./data/test_images

# QNN 推理
python scripts/evaluate_accuracy.py qnn --image_dir ./data/test_images

# 生成精度对比报告
python scripts/evaluate_accuracy.py report --results_dir ./results

# 性能测试
python scripts/benchmark_performance.py --image_dir ./data/test_images --output ./reports/性能测试报告.md
```

## 模型转换流程

```
PaddleOCR-VL 子模型
├── 布局检测 (PP-StructureV3, RT-DETR based)
│   └── Paddle (.pdmodel) → ONNX → QNN C++ → .so → HTP context binary
│       量化: INT8 (CNN 结构，量化友好)
│
└── VL 识别 (PP-DocBee2-3B, Qwen2.5-VL based)
    ├── 视觉编码器 (SigLip-400M)
    │   └── Paddle/HF → ONNX → QNN → HTP context binary
    │       量化: FP16 (ViT 结构，保精度)
    │
    └── 文本解码器 (Qwen2.5-3B)
        ├── Prefill 子图 → ONNX → QNN → HTP context binary
        └── Decode 子图  → ONNX → QNN → HTP context binary
            量化: FP16 (LLM 对量化敏感)
```

## 文档解析 Pipeline

```
输入: 单页文档图片
      ↓
 ┌─ 布局检测 (QNN/HTP) ─┐
 │  识别版面元素:          │
 │  文本块/标题/表格/      │
 │  公式/图表              │
 └────────┬───────────────┘
          ↓
    按阅读顺序排序
          ↓
 ┌─ 逐区域 VL 识别 (QNN/HTP) ─┐
 │  文本块 → 段落文字            │
 │  表格   → Markdown 表格      │
 │  公式   → LaTeX 格式         │
 └────────┬─────────────────────┘
          ↓
    组装 Markdown 输出
          ↓
输出: 结构化 Markdown
```

## 验收要求

| 要求 | 状态 | 说明 |
|------|------|------|
| 模型转换完整 | 待验证 | Paddle → ONNX → QNN 全链路脚本已提供 |
| 端侧推理可运行 | 待验证 | 支持 HTP-simulator 和实际设备 |
| 文档解析 pipeline 可用 | 待验证 | 覆盖文本块、表格、公式、图表 |
| 精度损失 ≤ 5% | 待验证 | 评测脚本和报告模板已提供 |

## 参考文档

- [PaddleOCR-VL GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [Paddle2ONNX 文档](https://github.com/PaddlePaddle/Paddle2ONNX)
- [Qualcomm QNN SDK 文档](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html)
- [高通 HTP 后端优化指南](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/htp_backend.html)
