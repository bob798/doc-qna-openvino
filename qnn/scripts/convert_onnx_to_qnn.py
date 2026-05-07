#!/usr/bin/env python3
"""
ONNX → QNN 全链路转换脚本

使用 QNN SDK 工具链将 ONNX 模型转换为 QNN 格式并针对 HTP 后端优化：
    1. qnn-onnx-converter   → ONNX 转 QNN 模型 (.cpp / .bin)
    2. qnn-model-lib-generator → 生成 QNN 模型共享库 (.so)
    3. qnn-context-binary-generator → 生成 HTP 上下文二进制 (.bin)

使用方式：
    python convert_onnx_to_qnn.py --input_dir ./models/onnx --output_dir ./models/qnn
    python convert_onnx_to_qnn.py --model ./models/onnx/layout_detection.onnx --backend htp
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── QNN SDK 环境检测 ──────────────────────────────────────────────────────────

def check_qnn_sdk() -> str:
    """检测 QNN SDK 安装路径"""
    qnn_root = os.environ.get("QNN_SDK_ROOT")
    if not qnn_root:
        # 常见安装路径
        candidates = [
            os.path.expanduser("~/Qualcomm/AIStack/qairt"),
            os.path.expanduser("~/qairt"),
            "/opt/qcom/aistack/qairt",
            os.path.expanduser("~/Qualcomm/AIStack/QNN"),
            "/opt/qcom/aistack/qnn",
        ]
        for path in candidates:
            if os.path.exists(path):
                qnn_root = path
                break

    if not qnn_root or not os.path.exists(qnn_root):
        logger.error("未找到 QNN SDK，请设置环境变量 QNN_SDK_ROOT")
        logger.info("下载地址: https://aihub.qualcomm.com/ (需注册)")
        logger.info("安装后: export QNN_SDK_ROOT=/path/to/qairt/<version>")
        sys.exit(1)

    # 查找最新版本目录
    versions = sorted(Path(qnn_root).glob("*"), reverse=True)
    sdk_path = str(versions[0]) if versions else qnn_root

    logger.info(f"QNN SDK 路径: {sdk_path}")
    return sdk_path


def get_tool_path(sdk_path: str, tool_name: str) -> str:
    """获取 QNN 工具的完整路径"""
    # 优先在 bin 目录查找
    for search_dir in ["bin/x86_64-linux-clang", "bin", "tools"]:
        tool_path = os.path.join(sdk_path, search_dir, tool_name)
        if os.path.exists(tool_path):
            return tool_path

    # 尝试直接调用（假设已在 PATH 中）
    return tool_name


def run_command(cmd: list, description: str):
    """执行命令并检查结果"""
    logger.info(f"执行: {description}")
    logger.info(f"  命令: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout:
            logger.info(result.stdout[-500:])  # 只打印最后 500 字符
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"命令执行失败: {e}")
        if e.stderr:
            logger.error(f"  stderr: {e.stderr[-500:]}")
        raise


# ── 转换配置 ──────────────────────────────────────────────────────────────────

MODELS = {
    "layout_detection": {
        "onnx_file": "layout_detection.onnx",
        "description": "PP-StructureV3 布局检测模型",
        "input_layout": "NCHW",
    },
    "vl_vision_encoder": {
        "onnx_file": "vl_vision_encoder.onnx",
        "description": "PP-DocBee2 视觉编码器",
        "input_layout": "NCHW",
    },
    "vl_text_decoder_prefill": {
        "onnx_file": "vl_text_decoder_prefill.onnx",
        "description": "PP-DocBee2 文本解码器 (prefill)",
        "input_layout": None,  # 非图像输入
    },
    "vl_text_decoder_decode": {
        "onnx_file": "vl_text_decoder_decode.onnx",
        "description": "PP-DocBee2 文本解码器 (decode)",
        "input_layout": None,
    },
}


# ── Step 1: ONNX → QNN 模型转换 ──────────────────────────────────────────────

def convert_onnx_to_qnn_model(
    sdk_path: str,
    onnx_path: str,
    output_dir: str,
    model_name: str,
    input_layout: Optional[str] = "NCHW",
    quantization: Optional[str] = None,
    input_list: Optional[str] = None,
):
    """
    Step 1: qnn-onnx-converter
    将 ONNX 模型转换为 QNN C++ 模型表示
    """
    converter = get_tool_path(sdk_path, "qnn-onnx-converter")
    output_path = os.path.join(output_dir, f"{model_name}.cpp")

    cmd = [
        converter,
        "--input_network", onnx_path,
        "--output_path", output_path,
    ]

    # 输入布局转换（NCHW → NHWC，HTP 更高效）
    if input_layout:
        cmd.extend(["--input_layout", input_layout, "NHWC"])

    # 量化选项
    if quantization and input_list:
        cmd.extend(["--input_list", input_list])
        if quantization == "int8":
            cmd.extend(["--act_bw", "8", "--weight_bw", "8"])
        elif quantization == "int16":
            cmd.extend(["--act_bw", "16", "--weight_bw", "8"])
        elif quantization == "fp16":
            cmd.extend(["--float_bw", "16"])

    os.makedirs(output_dir, exist_ok=True)
    run_command(cmd, f"ONNX → QNN 转换: {model_name}")

    logger.info(f"  ✓ QNN 模型生成: {output_path}")
    return output_path


# ── Step 2: 生成模型共享库 ────────────────────────────────────────────────────

def generate_model_lib(
    sdk_path: str,
    model_cpp: str,
    output_dir: str,
    model_name: str,
    target: str = "x86_64-linux-clang",
):
    """
    Step 2: qnn-model-lib-generator
    将 QNN C++ 模型编译为共享库 (.so)
    """
    lib_generator = get_tool_path(sdk_path, "qnn-model-lib-generator")

    cmd = [
        lib_generator,
        "-c", model_cpp,
        "-b", os.path.join(os.path.dirname(model_cpp), f"{model_name}.bin"),
        "-o", output_dir,
        "-t", target,
    ]

    run_command(cmd, f"生成模型库: {model_name}")

    lib_path = os.path.join(output_dir, target, f"lib{model_name}.so")
    logger.info(f"  ✓ 模型库生成: {lib_path}")
    return lib_path


# ── Step 3: 生成 HTP 上下文二进制 ─────────────────────────────────────────────

def generate_context_binary(
    sdk_path: str,
    model_lib_or_bin: str,
    output_dir: str,
    model_name: str,
    backend: str = "htp",
):
    """
    Step 3: qnn-context-binary-generator
    为 HTP 后端生成优化的上下文二进制文件
    """
    ctx_generator = get_tool_path(sdk_path, "qnn-context-binary-generator")

    # 后端库映射
    backend_libs = {
        "htp": "libQnnHtp.so",
        "gpu": "libQnnGpu.so",
        "cpu": "libQnnCpu.so",
    }
    backend_lib = os.path.join(sdk_path, "lib/x86_64-linux-clang", backend_libs.get(backend, "libQnnHtp.so"))

    output_path = os.path.join(output_dir, f"{model_name}_ctx.bin")

    cmd = [
        ctx_generator,
        "--model", model_lib_or_bin,
        "--backend", backend_lib,
        "--binary_file", output_path,
    ]

    # HTP 特定优化选项
    if backend == "htp":
        cmd.extend([
            "--config_file", _generate_htp_config(output_dir, model_name),
        ])

    os.makedirs(output_dir, exist_ok=True)
    run_command(cmd, f"生成 HTP 上下文二进制: {model_name}")

    logger.info(f"  ✓ 上下文二进制生成: {output_path}")
    return output_path


def _generate_htp_config(output_dir: str, model_name: str) -> str:
    """生成 HTP 后端优化配置"""
    config_path = os.path.join(output_dir, f"{model_name}_htp_config.json")
    config = """{
    "graphs": {
        "vtcm_mb": 8,
        "O": 3,
        "fp16_relaxed_precision": 1
    },
    "devices": [
        {
            "soc_model": 43,
            "cores": [
                {"core_id": 0, "perf_profile": "burst"}
            ]
        }
    ]
}"""
    with open(config_path, "w") as f:
        f.write(config)
    return config_path


# ── 全链路转换 ────────────────────────────────────────────────────────────────

def convert_full_pipeline(
    input_dir: str,
    output_dir: str,
    backend: str = "htp",
    quantization: Optional[str] = None,
    input_list_dir: Optional[str] = None,
):
    """执行全链路转换：ONNX → QNN C++ → 共享库 → HTP 上下文二进制"""
    sdk_path = check_qnn_sdk()

    qnn_model_dir = os.path.join(output_dir, "models")
    qnn_lib_dir = os.path.join(output_dir, "libs")
    qnn_ctx_dir = os.path.join(output_dir, "context_binaries")

    results = {}

    for model_name, config in MODELS.items():
        onnx_path = os.path.join(input_dir, config["onnx_file"])
        if not os.path.exists(onnx_path):
            logger.warning(f"跳过 {model_name}: 未找到 {onnx_path}")
            continue

        logger.info("=" * 60)
        logger.info(f"转换 {model_name}: {config['description']}")
        logger.info("=" * 60)

        # 量化数据列表
        input_list = None
        if quantization and input_list_dir:
            input_list = os.path.join(input_list_dir, f"{model_name}_input_list.txt")

        # Step 1: ONNX → QNN
        model_cpp = convert_onnx_to_qnn_model(
            sdk_path, onnx_path, qnn_model_dir, model_name,
            input_layout=config.get("input_layout"),
            quantization=quantization,
            input_list=input_list,
        )

        # Step 2: 编译共享库
        model_lib = generate_model_lib(
            sdk_path, model_cpp, qnn_lib_dir, model_name,
        )

        # Step 3: 生成 HTP 上下文二进制
        ctx_binary = generate_context_binary(
            sdk_path, model_lib, qnn_ctx_dir, model_name, backend,
        )

        results[model_name] = {
            "model_cpp": model_cpp,
            "model_lib": model_lib,
            "context_binary": ctx_binary,
        }

    # 打印汇总
    logger.info("=" * 60)
    logger.info("转换汇总")
    logger.info("=" * 60)
    for name, paths in results.items():
        logger.info(f"  {name}:")
        for key, path in paths.items():
            logger.info(f"    {key}: {path}")

    return results


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ONNX → QNN 全链路转换")
    parser.add_argument("--input_dir", type=str, default="./models/onnx",
                        help="ONNX 模型输入目录")
    parser.add_argument("--output_dir", type=str, default="./models/qnn",
                        help="QNN 模型输出目录")
    parser.add_argument("--model", type=str, default=None,
                        help="单独转换指定 ONNX 模型文件")
    parser.add_argument("--model_name", type=str, default=None,
                        help="模型名称（与 --model 配合使用）")
    parser.add_argument("--backend", choices=["htp", "gpu", "cpu"], default="htp",
                        help="目标后端")
    parser.add_argument("--quantization", choices=["int8", "int16", "fp16", None], default=None,
                        help="量化精度")
    parser.add_argument("--input_list_dir", type=str, default=None,
                        help="量化校准数据列表目录")
    parser.add_argument("--check_sdk", action="store_true",
                        help="仅检查 QNN SDK 环境")

    args = parser.parse_args()

    if args.check_sdk:
        check_qnn_sdk()
        return

    if args.model:
        # 单模型转换
        sdk_path = check_qnn_sdk()
        model_name = args.model_name or Path(args.model).stem
        qnn_model_dir = os.path.join(args.output_dir, "models")

        convert_onnx_to_qnn_model(
            sdk_path, args.model, qnn_model_dir, model_name,
            quantization=args.quantization,
            input_list=os.path.join(args.input_list_dir, f"{model_name}_input_list.txt") if args.input_list_dir else None,
        )
        generate_model_lib(
            sdk_path,
            os.path.join(qnn_model_dir, f"{model_name}.cpp"),
            os.path.join(args.output_dir, "libs"),
            model_name,
        )
        generate_context_binary(
            sdk_path,
            os.path.join(args.output_dir, "libs", "x86_64-linux-clang", f"lib{model_name}.so"),
            os.path.join(args.output_dir, "context_binaries"),
            model_name,
            args.backend,
        )
    else:
        # 全链路转换
        convert_full_pipeline(
            args.input_dir, args.output_dir,
            backend=args.backend,
            quantization=args.quantization,
            input_list_dir=args.input_list_dir,
        )


if __name__ == "__main__":
    main()
