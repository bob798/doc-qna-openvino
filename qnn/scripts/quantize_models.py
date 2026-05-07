#!/usr/bin/env python3
"""
QNN 模型量化脚本

对 PaddleOCR-VL 子模型进行量化，以适配高通 HTP 高效推理：
- INT8：最高性能，适用于布局检测模型
- INT16：性能与精度平衡
- FP16：最高精度，适用于 VL 识别模型

使用方式：
    # 生成校准数据
    python quantize_models.py --generate_calibration --image_dir ./data/calibration_images

    # INT8 量化布局检测模型
    python quantize_models.py --model layout --precision int8

    # FP16 量化 VL 模型（推荐，精度损失最小）
    python quantize_models.py --model vl --precision fp16

    # 全量化（推荐配置：布局检测 INT8 + VL 模型 FP16）
    python quantize_models.py --model all --auto
"""

import os
import sys
import argparse
import json
import logging
import struct
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 量化策略配置 ──────────────────────────────────────────────────────────────

QUANTIZATION_STRATEGIES = {
    "layout_detection": {
        "recommended_precision": "int8",
        "description": "布局检测模型 — CNN 结构，INT8 量化精度损失极小",
        "act_bw": 8,
        "weight_bw": 8,
        "input_shape": [1, 3, 800, 608],
        "use_symmetric": True,
        "calibration_samples": 100,
    },
    "vl_vision_encoder": {
        "recommended_precision": "fp16",
        "description": "视觉编码器 — ViT 结构，FP16 保精度",
        "float_bw": 16,
        "input_shape": [1, 3, 384, 384],
        "calibration_samples": 50,
    },
    "vl_text_decoder_prefill": {
        "recommended_precision": "fp16",
        "description": "文本解码器(prefill) — Transformer，FP16 保精度",
        "float_bw": 16,
        "calibration_samples": 20,
        "note": "LLM 解码器对量化敏感，建议 FP16",
    },
    "vl_text_decoder_decode": {
        "recommended_precision": "fp16",
        "description": "文本解码器(decode) — 带 KV-cache，FP16",
        "float_bw": 16,
        "calibration_samples": 20,
    },
}


# ── 校准数据生成 ──────────────────────────────────────────────────────────────

def generate_calibration_data(
    image_dir: str,
    output_dir: str,
    num_samples: int = 100,
):
    """
    生成量化校准数据（input_list.txt + raw 文件）

    QNN 量化器需要校准数据来统计各层的激活分布，
    格式为 input_list.txt 指向一系列 .raw 文件。
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        from PIL import Image
    except ImportError:
        logger.error("需要 Pillow: pip install Pillow")
        sys.exit(1)

    # 收集图片
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = []
    for f in sorted(Path(image_dir).iterdir()):
        if f.suffix.lower() in image_extensions:
            images.append(str(f))
    images = images[:num_samples]

    if not images:
        logger.error(f"未在 {image_dir} 找到图片，请准备校准数据集")
        logger.info("建议使用 50-100 张代表性文档图片作为校准数据")
        sys.exit(1)

    logger.info(f"使用 {len(images)} 张图片生成校准数据")

    # ── 布局检测模型校准数据 ──
    _generate_layout_calibration(images, output_dir)

    # ── 视觉编码器校准数据 ──
    _generate_vision_calibration(images, output_dir)

    # ── 文本解码器校准数据（使用随机 token）──
    _generate_decoder_calibration(output_dir)

    logger.info(f"校准数据生成完成: {output_dir}")


def _generate_layout_calibration(images: list, output_dir: str):
    """生成布局检测模型的校准数据"""
    from PIL import Image

    raw_dir = os.path.join(output_dir, "layout_raw")
    os.makedirs(raw_dir, exist_ok=True)
    input_list = []

    target_h, target_w = 800, 608
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for i, img_path in enumerate(images):
        img = Image.open(img_path).convert("RGB").resize((target_w, target_h))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        # HWC → CHW → NCHW
        arr = arr.transpose(2, 0, 1)[np.newaxis, ...]

        raw_path = os.path.join(raw_dir, f"layout_{i:04d}.raw")
        arr.astype(np.float32).tofile(raw_path)
        input_list.append(raw_path)

    list_path = os.path.join(output_dir, "layout_detection_input_list.txt")
    with open(list_path, "w") as f:
        f.write("\n".join(input_list))

    logger.info(f"  布局检测校准数据: {len(input_list)} 个样本 → {list_path}")


def _generate_vision_calibration(images: list, output_dir: str):
    """生成视觉编码器的校准数据"""
    from PIL import Image

    raw_dir = os.path.join(output_dir, "vision_raw")
    os.makedirs(raw_dir, exist_ok=True)
    input_list = []

    target_size = 384
    # SigLip 的预处理参数
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    for i, img_path in enumerate(images):
        img = Image.open(img_path).convert("RGB").resize((target_size, target_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)[np.newaxis, ...]

        raw_path = os.path.join(raw_dir, f"vision_{i:04d}.raw")
        arr.astype(np.float32).tofile(raw_path)
        input_list.append(raw_path)

    list_path = os.path.join(output_dir, "vl_vision_encoder_input_list.txt")
    with open(list_path, "w") as f:
        f.write("\n".join(input_list))

    logger.info(f"  视觉编码器校准数据: {len(input_list)} 个样本 → {list_path}")


def _generate_decoder_calibration(output_dir: str, num_samples: int = 20):
    """生成文本解码器的校准数据（随机 token 序列）"""
    raw_dir = os.path.join(output_dir, "decoder_raw")
    os.makedirs(raw_dir, exist_ok=True)

    vocab_size = 151936  # Qwen2.5 词表大小
    max_seq_len = 512

    for phase in ["prefill", "decode"]:
        input_list = []
        for i in range(num_samples):
            if phase == "prefill":
                seq_len = np.random.randint(64, max_seq_len)
                input_ids = np.random.randint(0, vocab_size, (1, seq_len)).astype(np.int64)
            else:
                input_ids = np.random.randint(0, vocab_size, (1, 1)).astype(np.int64)

            raw_path = os.path.join(raw_dir, f"decoder_{phase}_{i:04d}.raw")
            input_ids.tofile(raw_path)
            input_list.append(raw_path)

        list_path = os.path.join(output_dir, f"vl_text_decoder_{phase}_input_list.txt")
        with open(list_path, "w") as f:
            f.write("\n".join(input_list))

        logger.info(f"  文本解码器({phase})校准数据: {num_samples} 个样本 → {list_path}")


# ── 量化执行 ──────────────────────────────────────────────────────────────────

def quantize_model(
    model_name: str,
    onnx_dir: str,
    output_dir: str,
    precision: str,
    calibration_dir: str,
):
    """对指定模型执行量化转换"""
    from convert_onnx_to_qnn import check_qnn_sdk, convert_onnx_to_qnn_model

    strategy = QUANTIZATION_STRATEGIES.get(model_name)
    if not strategy:
        logger.error(f"未知模型: {model_name}")
        return

    logger.info(f"量化 {model_name}: {strategy['description']}")
    logger.info(f"  精度: {precision}")

    # 构建 ONNX 文件名
    onnx_files = {
        "layout_detection": "layout_detection.onnx",
        "vl_vision_encoder": "vl_vision_encoder.onnx",
        "vl_text_decoder_prefill": "vl_text_decoder_prefill.onnx",
        "vl_text_decoder_decode": "vl_text_decoder_decode.onnx",
    }

    onnx_path = os.path.join(onnx_dir, onnx_files[model_name])
    if not os.path.exists(onnx_path):
        logger.warning(f"未找到 ONNX 模型: {onnx_path}，跳过")
        return

    sdk_path = check_qnn_sdk()
    quantized_output = os.path.join(output_dir, f"quantized_{precision}")

    convert_onnx_to_qnn_model(
        sdk_path=sdk_path,
        onnx_path=onnx_path,
        output_dir=quantized_output,
        model_name=f"{model_name}_{precision}",
        input_layout=strategy.get("input_layout", "NCHW") if "input_shape" in strategy else None,
        quantization=precision,
        input_list=os.path.join(calibration_dir, f"{model_name}_input_list.txt"),
    )

    logger.info(f"  ✓ {model_name} {precision} 量化完成")


def auto_quantize(onnx_dir: str, output_dir: str, calibration_dir: str):
    """自动量化：按推荐策略量化所有模型"""
    logger.info("=" * 60)
    logger.info("自动量化模式：使用推荐精度配置")
    logger.info("=" * 60)
    logger.info("  布局检测: INT8 (CNN, 量化友好)")
    logger.info("  视觉编码器: FP16 (ViT, 保精度)")
    logger.info("  文本解码器: FP16 (LLM, 量化敏感)")
    logger.info("")

    for model_name, strategy in QUANTIZATION_STRATEGIES.items():
        precision = strategy["recommended_precision"]
        quantize_model(model_name, onnx_dir, output_dir, precision, calibration_dir)


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QNN 模型量化")
    subparsers = parser.add_subparsers(dest="command")

    # 生成校准数据
    cal_parser = subparsers.add_parser("calibrate", help="生成量化校准数据")
    cal_parser.add_argument("--image_dir", type=str, required=True,
                            help="校准图片目录")
    cal_parser.add_argument("--output_dir", type=str, default="./data/calibration",
                            help="校准数据输出目录")
    cal_parser.add_argument("--num_samples", type=int, default=100,
                            help="校准样本数")

    # 量化
    q_parser = subparsers.add_parser("quantize", help="量化模型")
    q_parser.add_argument("--model", choices=list(QUANTIZATION_STRATEGIES.keys()) + ["all"],
                          default="all", help="要量化的模型")
    q_parser.add_argument("--precision", choices=["int8", "int16", "fp16"],
                          help="量化精度（--model all 时忽略，使用推荐配置）")
    q_parser.add_argument("--onnx_dir", type=str, default="./models/onnx",
                          help="ONNX 模型目录")
    q_parser.add_argument("--output_dir", type=str, default="./models/qnn",
                          help="量化模型输出目录")
    q_parser.add_argument("--calibration_dir", type=str, default="./data/calibration",
                          help="校准数据目录")

    # 兼容旧命令行接口
    parser.add_argument("--generate_calibration", action="store_true",
                        help="生成校准数据")
    parser.add_argument("--image_dir", type=str, default="./data/calibration_images")
    parser.add_argument("--model", choices=list(QUANTIZATION_STRATEGIES.keys()) + ["all"])
    parser.add_argument("--precision", choices=["int8", "int16", "fp16"])
    parser.add_argument("--auto", action="store_true", help="使用推荐量化配置")

    args = parser.parse_args()

    if args.command == "calibrate":
        generate_calibration_data(args.image_dir, args.output_dir, args.num_samples)
    elif args.command == "quantize":
        if args.model == "all":
            auto_quantize(args.onnx_dir, args.output_dir, args.calibration_dir)
        else:
            precision = args.precision or QUANTIZATION_STRATEGIES[args.model]["recommended_precision"]
            quantize_model(args.model, args.onnx_dir, args.output_dir, precision, args.calibration_dir)
    elif args.generate_calibration:
        generate_calibration_data(args.image_dir, "./data/calibration")
    elif args.auto:
        auto_quantize("./models/onnx", "./models/qnn", "./data/calibration")
    elif args.model:
        precision = args.precision or QUANTIZATION_STRATEGIES[args.model]["recommended_precision"]
        quantize_model(args.model, "./models/onnx", "./models/qnn", precision, "./data/calibration")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
