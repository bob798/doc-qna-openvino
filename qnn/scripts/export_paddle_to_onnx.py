#!/usr/bin/env python3
"""
PaddleOCR-VL 子模型导出：Paddle → ONNX

将布局检测模型和 VL 识别模型分别导出为 ONNX 格式，
作为后续 QNN 转换的输入。

使用方式：
    python export_paddle_to_onnx.py --model_type layout --model_dir ./models/paddle/layout --output_dir ./models/onnx
    python export_paddle_to_onnx.py --model_type vl --model_dir ./models/paddle/vl --output_dir ./models/onnx
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 模型配置 ──────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "layout": {
        "description": "PP-StructureV3 布局检测模型（RT-DETR based）",
        "input_shape": [1, 3, 800, 608],  # batch, channels, height, width
        "input_names": ["image"],
        "output_names": ["boxes", "scores", "labels"],
        "opset_version": 14,
        "paddle_model_prefix": "model",  # model.pdmodel + model.pdiparams
    },
    "vl": {
        "description": "PP-DocBee2-3B 视觉语言识别模型",
        "submodels": {
            "vision_encoder": {
                "description": "视觉编码器（SigLip-400M）",
                "input_shape": [1, 3, 384, 384],
                "input_names": ["pixel_values"],
                "output_names": ["image_features"],
                "opset_version": 14,
            },
            "text_decoder": {
                "description": "文本解码器（Qwen2.5-3B）",
                "input_names": ["input_ids", "attention_mask", "position_ids", "past_key_values"],
                "output_names": ["logits", "present_key_values"],
                "opset_version": 14,
                "note": "解码器需按 KV-cache 拆分为 prefill 和 decode 两个子图",
            },
        },
    },
}


# ── 布局检测模型导出 ──────────────────────────────────────────────────────────

def export_layout_model(model_dir: str, output_dir: str, opset_version: int = 14):
    """导出布局检测模型 Paddle → ONNX"""
    try:
        import paddle2onnx
    except ImportError:
        logger.error("请安装 paddle2onnx: pip install paddle2onnx")
        sys.exit(1)

    config = MODEL_CONFIGS["layout"]
    model_prefix = os.path.join(model_dir, config["paddle_model_prefix"])
    pdmodel_path = f"{model_prefix}.pdmodel"
    pdiparams_path = f"{model_prefix}.pdiparams"

    if not os.path.exists(pdmodel_path):
        logger.error(f"未找到模型文件: {pdmodel_path}")
        logger.info("请先下载布局检测模型：")
        logger.info("  paddleocr model -t structure -d PP-StructureV3")
        logger.info("  或从 HuggingFace 下载: PaddlePaddle/PP-StructureV3")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "layout_detection.onnx")

    logger.info(f"导出布局检测模型: {pdmodel_path} → {output_path}")
    logger.info(f"  输入形状: {config['input_shape']}")
    logger.info(f"  ONNX opset: {opset_version}")

    # Paddle2ONNX 命令行等效调用
    paddle2onnx.export(
        model_file=pdmodel_path,
        params_file=pdiparams_path,
        save_file=output_path,
        opset_version=opset_version,
        enable_onnx_checker=True,
        auto_update_opset=True,
    )

    logger.info(f"布局检测模型导出成功: {output_path}")
    _validate_onnx(output_path)
    return output_path


# ── VL 识别模型导出 ────────────────────────────────────────────────────────────

def export_vl_model(model_dir: str, output_dir: str, opset_version: int = 14):
    """
    导出 PP-DocBee2-3B 视觉语言模型。

    PP-DocBee2-3B 基于 Qwen2.5-VL 架构，包含：
    1. 视觉编码器 (SigLip-400M) → 独立导出
    2. 文本解码器 (Qwen2.5-3B) → 拆分为 prefill + decode 两个子图导出

    对于 3B 规模的解码器，直接 Paddle2ONNX 导出可能遇到算子兼容性问题，
    因此提供两条路径：
    - 路径 A：直接 Paddle2ONNX 导出（优先尝试）
    - 路径 B：通过 HuggingFace transformers + torch.onnx.export 导出
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 路径 A：尝试 Paddle2ONNX 直接导出 ──
    logger.info("=" * 60)
    logger.info("尝试路径 A：Paddle2ONNX 直接导出 VL 模型子组件")
    logger.info("=" * 60)

    vl_submodels = {
        "vision_encoder": {
            "paddle_dir": os.path.join(model_dir, "vision_encoder"),
            "output_name": "vl_vision_encoder.onnx",
        },
        "text_decoder_prefill": {
            "paddle_dir": os.path.join(model_dir, "text_decoder"),
            "output_name": "vl_text_decoder_prefill.onnx",
        },
        "text_decoder_decode": {
            "paddle_dir": os.path.join(model_dir, "text_decoder"),
            "output_name": "vl_text_decoder_decode.onnx",
        },
    }

    exported_paths = {}

    for name, info in vl_submodels.items():
        paddle_dir = info["paddle_dir"]
        output_path = os.path.join(output_dir, info["output_name"])

        pdmodel = os.path.join(paddle_dir, "model.pdmodel")
        pdiparams = os.path.join(paddle_dir, "model.pdiparams")

        if os.path.exists(pdmodel) and os.path.exists(pdiparams):
            logger.info(f"导出 {name}: {pdmodel} → {output_path}")
            try:
                import paddle2onnx
                paddle2onnx.export(
                    model_file=pdmodel,
                    params_file=pdiparams,
                    save_file=output_path,
                    opset_version=opset_version,
                    enable_onnx_checker=True,
                    auto_update_opset=True,
                )
                exported_paths[name] = output_path
                logger.info(f"  ✓ {name} 导出成功")
            except Exception as e:
                logger.warning(f"  ✗ {name} Paddle2ONNX 导出失败: {e}")
                logger.info(f"  → 将尝试路径 B（HuggingFace 导出）")
        else:
            logger.warning(f"  未找到 Paddle 模型: {paddle_dir}")
            logger.info(f"  → 将尝试路径 B（HuggingFace 导出）")

    # ── 路径 B：HuggingFace transformers 导出 ──
    missing = set(vl_submodels.keys()) - set(exported_paths.keys())
    if missing:
        logger.info("=" * 60)
        logger.info("路径 B：通过 HuggingFace transformers 导出缺失组件")
        logger.info("=" * 60)
        _export_vl_via_transformers(model_dir, output_dir, missing, opset_version)

    return exported_paths


def _export_vl_via_transformers(model_dir: str, output_dir: str, components: set, opset_version: int):
    """
    通过 HuggingFace transformers 加载 PP-DocBee2-3B 并导出 ONNX。
    PP-DocBee2-3B 兼容 Qwen2.5-VL 架构，可用 transformers AutoModel 加载。
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer, AutoProcessor
    except ImportError:
        logger.error("路径 B 需要 torch + transformers: pip install torch transformers")
        logger.info("或使用 optimum-cli 导出:")
        logger.info("  optimum-cli export onnx --model PaddlePaddle/PP-DocBee2-3B ./models/onnx/vl/")
        return

    hf_model_id = "PaddlePaddle/PP-DocBee2-3B"
    local_hf_dir = os.path.join(model_dir, "hf")

    # 优先用 optimum 导出（更稳定）
    logger.info("推荐使用 optimum-cli 导出（自动处理动态轴和子图拆分）：")
    logger.info(f"  optimum-cli export onnx --model {hf_model_id} {output_dir}/vl/ --task image-text-to-text")
    logger.info("")
    logger.info("如 optimum 不支持该模型架构，手动导出流程如下：")

    if "vision_encoder" in components:
        logger.info("手动导出视觉编码器：")
        logger.info(f"""
    from transformers import AutoModel
    import torch

    model = AutoModel.from_pretrained("{hf_model_id}", trust_remote_code=True)
    vision = model.visual  # SigLip 视觉编码器

    dummy_pixel = torch.randn(1, 3, 384, 384)
    torch.onnx.export(
        vision, dummy_pixel,
        "{output_dir}/vl_vision_encoder.onnx",
        input_names=["pixel_values"],
        output_names=["image_features"],
        dynamic_axes={{"pixel_values": {{0: "batch"}}, "image_features": {{0: "batch"}}}},
        opset_version={opset_version},
    )
""")

    if "text_decoder_prefill" in components or "text_decoder_decode" in components:
        logger.info("手动导出文本解码器（需拆分 prefill/decode）：")
        logger.info(f"""
    # Prefill 阶段（首次前向，无 KV-cache）
    torch.onnx.export(
        model.model,  # Qwen2.5 decoder
        (input_ids, attention_mask, position_ids),
        "{output_dir}/vl_text_decoder_prefill.onnx",
        input_names=["input_ids", "attention_mask", "position_ids"],
        output_names=["logits"] + [f"present_key_{{i}}" for i in range(num_layers)],
        dynamic_axes={{
            "input_ids": {{0: "batch", 1: "seq_len"}},
            "attention_mask": {{0: "batch", 1: "seq_len"}},
        }},
        opset_version={opset_version},
    )

    # Decode 阶段（逐 token 生成，带 KV-cache）
    torch.onnx.export(
        model.model,
        (input_ids_single, attention_mask, position_ids, past_key_values),
        "{output_dir}/vl_text_decoder_decode.onnx",
        input_names=["input_ids", "attention_mask", "position_ids"]
                   + [f"past_key_{{i}}" for i in range(num_layers)],
        output_names=["logits"] + [f"present_key_{{i}}" for i in range(num_layers)],
        opset_version={opset_version},
    )
""")


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _validate_onnx(onnx_path: str):
    """验证导出的 ONNX 模型"""
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        logger.info(f"  ONNX 验证通过: {onnx_path}")

        # 打印模型信息
        graph = model.graph
        logger.info(f"  输入: {[inp.name for inp in graph.input]}")
        logger.info(f"  输出: {[out.name for out in graph.output]}")
        logger.info(f"  节点数: {len(graph.node)}")
    except ImportError:
        logger.warning("  未安装 onnx 包，跳过验证: pip install onnx")
    except Exception as e:
        logger.warning(f"  ONNX 验证失败: {e}")


def download_models(output_dir: str):
    """下载 PaddleOCR-VL 所需的 Paddle 模型"""
    logger.info("下载 PaddleOCR-VL 模型...")
    logger.info("")
    logger.info("方式 1：使用 PaddleOCR CLI（推荐）")
    logger.info("  pip install paddleocr")
    logger.info("  # 下载布局检测模型")
    logger.info("  paddleocr model -t structure -d PP-StructureV3")
    logger.info("  # 下载 VL 识别模型")
    logger.info("  paddleocr model -t ocr_vl -d PP-DocBee2-3B")
    logger.info("")
    logger.info("方式 2：从 HuggingFace 下载")
    logger.info("  # 布局检测")
    logger.info("  huggingface-cli download PaddlePaddle/PP-StructureV3 --local-dir ./models/paddle/layout")
    logger.info("  # VL 识别")
    logger.info("  huggingface-cli download PaddlePaddle/PP-DocBee2-3B --local-dir ./models/paddle/vl")
    logger.info("")
    logger.info("方式 3：使用 PaddleOCR Python API 自动下载")
    logger.info("""
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False)
    # 模型将自动下载到 ~/.paddleocr/models/
""")


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PaddleOCR-VL 模型导出：Paddle → ONNX")
    parser.add_argument("--model_type", choices=["layout", "vl", "all"], default="all",
                        help="要导出的模型类型")
    parser.add_argument("--model_dir", type=str, default="./models/paddle",
                        help="Paddle 模型根目录")
    parser.add_argument("--output_dir", type=str, default="./models/onnx",
                        help="ONNX 模型输出目录")
    parser.add_argument("--opset", type=int, default=14,
                        help="ONNX opset version")
    parser.add_argument("--download", action="store_true",
                        help="显示模型下载指引")

    args = parser.parse_args()

    if args.download:
        download_models(args.model_dir)
        return

    if args.model_type in ("layout", "all"):
        layout_dir = os.path.join(args.model_dir, "layout")
        export_layout_model(layout_dir, args.output_dir, args.opset)

    if args.model_type in ("vl", "all"):
        vl_dir = os.path.join(args.model_dir, "vl")
        export_vl_model(vl_dir, args.output_dir, args.opset)

    logger.info("=" * 60)
    logger.info("导出完成！下一步：运行 ONNX → QNN 转换")
    logger.info("  python convert_onnx_to_qnn.py --input_dir ./models/onnx --output_dir ./models/qnn")


if __name__ == "__main__":
    main()
