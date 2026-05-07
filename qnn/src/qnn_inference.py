#!/usr/bin/env python3
"""
高通 QNN 端侧推理服务

加载转换后的 QNN 模型（context binary），在 HTP 后端执行推理。
支持布局检测和 VL 识别两个子模型的独立推理和组合调用。

使用方式：
    from qnn_inference import QNNInferenceService

    service = QNNInferenceService("./models/qnn/context_binaries")
    # 布局检测
    boxes, scores, labels = service.detect_layout(image)
    # VL 识别
    text = service.recognize_vl(image, prompt="识别图中所有文字和表格")
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── QNN Runtime 封装 ──────────────────────────────────────────────────────────

class QNNRuntime:
    """
    QNN Runtime 推理引擎封装

    支持两种运行方式：
    1. QNN Python API（需要 qnn_wrapper_api，QNN SDK 2.x+）
    2. qnn-net-run 命令行调用（兼容所有 QNN 版本）
    """

    def __init__(self, sdk_path: Optional[str] = None, backend: str = "htp"):
        self.sdk_path = sdk_path or os.environ.get("QNN_SDK_ROOT", "")
        self.backend = backend
        self._backend_lib = self._resolve_backend_lib()
        self._use_python_api = self._check_python_api()

    def _resolve_backend_lib(self) -> str:
        backend_map = {
            "htp": "libQnnHtp.so",
            "gpu": "libQnnGpu.so",
            "cpu": "libQnnCpu.so",
            "htp_simulator": "libQnnHtpNetRunExtensions.so",
        }
        lib_name = backend_map.get(self.backend, "libQnnHtp.so")
        lib_path = os.path.join(self.sdk_path, "lib", "x86_64-linux-clang", lib_name)
        if os.path.exists(lib_path):
            return lib_path
        return lib_name

    def _check_python_api(self) -> bool:
        try:
            sys.path.insert(0, os.path.join(self.sdk_path, "lib", "python"))
            import qnn_wrapper_api  # noqa: F401
            return True
        except ImportError:
            return False

    def load_context(self, context_binary: str) -> "QNNContext":
        """加载 QNN context binary"""
        if not os.path.exists(context_binary):
            raise FileNotFoundError(f"QNN context binary 未找到: {context_binary}")

        if self._use_python_api:
            return QNNContextPythonAPI(context_binary, self._backend_lib, self.sdk_path)
        else:
            return QNNContextCLI(context_binary, self._backend_lib, self.sdk_path)


class QNNContext:
    """QNN 推理上下文基类"""

    def execute(self, inputs: dict) -> dict:
        raise NotImplementedError


class QNNContextPythonAPI(QNNContext):
    """通过 QNN Python API 执行推理"""

    def __init__(self, context_binary: str, backend_lib: str, sdk_path: str):
        self.context_binary = context_binary
        sys.path.insert(0, os.path.join(sdk_path, "lib", "python"))
        import qnn_wrapper_api as qnn_api

        self.qnn = qnn_api
        self.model = qnn_api.QnnModel(
            backend_lib_path=backend_lib,
            model_path=context_binary,
            system_lib_path=os.path.join(sdk_path, "lib", "x86_64-linux-clang", "libQnnSystem.so"),
        )
        self.model.init()
        logger.info(f"QNN 模型加载成功 (Python API): {context_binary}")

    def execute(self, inputs: dict) -> dict:
        """执行推理，inputs 为 {name: np.ndarray} 字典"""
        output = self.model.execute(inputs)
        return output


class QNNContextCLI(QNNContext):
    """通过 qnn-net-run 命令行执行推理"""

    def __init__(self, context_binary: str, backend_lib: str, sdk_path: str):
        self.context_binary = context_binary
        self.backend_lib = backend_lib
        self.sdk_path = sdk_path
        self.net_run = os.path.join(sdk_path, "bin", "x86_64-linux-clang", "qnn-net-run")
        if not os.path.exists(self.net_run):
            self.net_run = "qnn-net-run"
        self._tmp_dir = "/tmp/qnn_inference"
        os.makedirs(self._tmp_dir, exist_ok=True)
        logger.info(f"QNN 模型加载成功 (CLI): {context_binary}")

    def execute(self, inputs: dict) -> dict:
        """
        执行推理：
        1. 将输入写为 .raw 文件
        2. 生成 input_list.txt
        3. 调用 qnn-net-run
        4. 读取输出 .raw 文件
        """
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory(prefix="qnn_") as tmpdir:
            # 写入输入数据
            input_list_entries = []
            for name, data in inputs.items():
                raw_path = os.path.join(tmpdir, f"{name}.raw")
                data.astype(np.float32).tofile(raw_path)
                input_list_entries.append(raw_path)

            input_list = os.path.join(tmpdir, "input_list.txt")
            with open(input_list, "w") as f:
                f.write(" ".join(input_list_entries))

            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # 执行推理
            cmd = [
                self.net_run,
                "--backend", self.backend_lib,
                "--retrieve_context", self.context_binary,
                "--input_list", input_list,
                "--output_dir", output_dir,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"qnn-net-run 失败: {result.stderr}")

            # 读取输出
            outputs = {}
            result_dir = os.path.join(output_dir, "Result_0")
            if os.path.exists(result_dir):
                for raw_file in sorted(Path(result_dir).glob("*.raw")):
                    outputs[raw_file.stem] = np.fromfile(str(raw_file), dtype=np.float32)

            return outputs


# ── 推理服务 ──────────────────────────────────────────────────────────────────

class QNNInferenceService:
    """
    PaddleOCR-VL QNN 端侧推理服务

    加载所有子模型，提供高层推理接口。
    """

    def __init__(
        self,
        model_dir: str,
        sdk_path: Optional[str] = None,
        backend: str = "htp",
    ):
        self.model_dir = model_dir
        self.runtime = QNNRuntime(sdk_path, backend)
        self.models = {}
        self._load_models()

    def _load_models(self):
        """加载所有 QNN 子模型"""
        model_files = {
            "layout": "layout_detection_ctx.bin",
            "vision_encoder": "vl_vision_encoder_ctx.bin",
            "text_decoder_prefill": "vl_text_decoder_prefill_ctx.bin",
            "text_decoder_decode": "vl_text_decoder_decode_ctx.bin",
        }

        for name, filename in model_files.items():
            path = os.path.join(self.model_dir, filename)
            if os.path.exists(path):
                self.models[name] = self.runtime.load_context(path)
                logger.info(f"  加载 {name}: {path}")
            else:
                logger.warning(f"  未找到 {name}: {path}")

    def detect_layout(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        布局检测：识别文档中的版面元素

        Args:
            image: BGR 图片 (H, W, 3) uint8

        Returns:
            boxes: 检测框 (N, 4) [x1, y1, x2, y2]
            scores: 置信度 (N,)
            labels: 类别标签 (N,) — 0:text, 1:title, 2:table, 3:figure, 4:formula
        """
        if "layout" not in self.models:
            raise RuntimeError("布局检测模型未加载")

        # 预处理
        input_tensor = self._preprocess_layout(image)

        # 推理
        t0 = time.time()
        outputs = self.models["layout"].execute({"image": input_tensor})
        t1 = time.time()
        logger.info(f"布局检测推理耗时: {(t1-t0)*1000:.1f} ms")

        # 后处理
        boxes = outputs.get("boxes", np.array([]))
        scores = outputs.get("scores", np.array([]))
        labels = outputs.get("labels", np.array([]))

        return boxes, scores, labels

    def recognize_vl(
        self,
        image: np.ndarray,
        prompt: str = "识别图中所有文字内容，保持原始格式输出为 Markdown",
        max_new_tokens: int = 2048,
    ) -> str:
        """
        VL 识别：视觉语言模型解析文档内容

        Args:
            image: BGR 图片 (H, W, 3) uint8
            prompt: 识别指令
            max_new_tokens: 最大生成 token 数

        Returns:
            解析结果文本（Markdown 格式）
        """
        if "vision_encoder" not in self.models:
            raise RuntimeError("视觉编码器未加载")

        # Step 1: 视觉编码
        pixel_values = self._preprocess_vision(image)
        t0 = time.time()
        vision_outputs = self.models["vision_encoder"].execute({"pixel_values": pixel_values})
        t1 = time.time()
        logger.info(f"视觉编码耗时: {(t1-t0)*1000:.1f} ms")

        image_features = vision_outputs.get("image_features")

        # Step 2: 构建输入（prompt tokenize + 图像 token 拼接）
        input_ids, attention_mask, position_ids = self._prepare_decoder_input(
            prompt, image_features
        )

        # Step 3: Prefill
        if "text_decoder_prefill" not in self.models:
            raise RuntimeError("文本解码器(prefill)未加载")

        t0 = time.time()
        prefill_outputs = self.models["text_decoder_prefill"].execute({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        })
        t1 = time.time()
        logger.info(f"Prefill 耗时: {(t1-t0)*1000:.1f} ms")

        # Step 4: Autoregressive decode
        generated_tokens = self._autoregressive_decode(
            prefill_outputs, max_new_tokens
        )

        # Step 5: Decode tokens to text
        text = self._decode_tokens(generated_tokens)
        return text

    # ── 预处理 ────────────────────────────────────────────────────────────────

    def _preprocess_layout(self, image: np.ndarray) -> np.ndarray:
        """布局检测预处理：resize + normalize + CHW"""
        from PIL import Image

        target_h, target_w = 800, 608
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if isinstance(image, np.ndarray):
            img = Image.fromarray(image[:, :, ::-1])  # BGR → RGB
        else:
            img = image

        img = img.resize((target_w, target_h))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW
        return arr

    def _preprocess_vision(self, image: np.ndarray) -> np.ndarray:
        """视觉编码器预处理：resize + normalize (SigLip)"""
        from PIL import Image

        target_size = 384
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        if isinstance(image, np.ndarray):
            img = Image.fromarray(image[:, :, ::-1])
        else:
            img = image

        img = img.resize((target_size, target_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
        return arr

    def _prepare_decoder_input(
        self, prompt: str, image_features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """构建文本解码器输入：tokenize + 图像 token 拼接"""
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-3B", trust_remote_code=True
            )
        except ImportError:
            logger.warning("未安装 transformers，使用简单 tokenizer fallback")
            tokenizer = None

        if tokenizer:
            # 构建 chat 格式
            messages = [
                {"role": "system", "content": "你是一个文档解析助手。"},
                {"role": "user", "content": f"<image>\n{prompt}"},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            tokens = tokenizer(text, return_tensors="np")
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
        else:
            # Fallback：直接用 prompt 的 UTF-8 bytes 作为 token IDs
            input_ids = np.array([[ord(c) for c in prompt]], dtype=np.int64)
            attention_mask = np.ones_like(input_ids, dtype=np.int64)

        seq_len = input_ids.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]

        return input_ids, attention_mask, position_ids

    def _autoregressive_decode(
        self, prefill_outputs: dict, max_new_tokens: int
    ) -> list:
        """自回归解码：逐 token 生成"""
        generated = []
        eos_token_id = 151645  # Qwen2.5 的 <|im_end|>

        # 从 prefill 输出中提取 logits 和 KV-cache
        logits = prefill_outputs.get("logits")
        if logits is None:
            logger.warning("prefill 输出中未找到 logits")
            return generated

        # 取最后一个位置的 logits，贪心采样
        next_token = int(np.argmax(logits[..., -1, :]))
        generated.append(next_token)

        if "text_decoder_decode" not in self.models:
            logger.warning("文本解码器(decode)未加载，仅返回首 token")
            return generated

        # 构建 KV-cache（从 prefill 输出中提取 present_key_* ）
        past_kv = {k: v for k, v in prefill_outputs.items() if k.startswith("present_key")}

        t0 = time.time()
        for step in range(max_new_tokens - 1):
            input_ids = np.array([[next_token]], dtype=np.int64)
            attention_mask = np.ones((1, 1), dtype=np.int64)
            position_ids = np.array([[len(generated)]], dtype=np.int64)

            decode_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
            decode_inputs.update(past_kv)

            outputs = self.models["text_decoder_decode"].execute(decode_inputs)
            logits = outputs.get("logits")
            if logits is None:
                break

            next_token = int(np.argmax(logits[..., -1, :]))
            if next_token == eos_token_id:
                break

            generated.append(next_token)
            past_kv = {k: v for k, v in outputs.items() if k.startswith("present_key")}

        t1 = time.time()
        logger.info(f"Decode {len(generated)} tokens 耗时: {(t1-t0)*1000:.1f} ms")
        return generated

    def _decode_tokens(self, token_ids: list) -> str:
        """将 token IDs 解码为文本"""
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-3B", trust_remote_code=True
            )
            return tokenizer.decode(token_ids, skip_special_tokens=True)
        except ImportError:
            return "".join(chr(t) if t < 128 else "?" for t in token_ids)


# ── 便捷函数 ──────────────────────────────────────────────────────────────────

def create_service(
    model_dir: str = "./models/qnn/context_binaries",
    backend: str = "htp",
) -> QNNInferenceService:
    """创建推理服务实例"""
    return QNNInferenceService(model_dir=model_dir, backend=backend)


# ── CLI 测试 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QNN 端侧推理服务测试")
    parser.add_argument("--model_dir", type=str, default="./models/qnn/context_binaries",
                        help="QNN context binary 目录")
    parser.add_argument("--backend", choices=["htp", "htp_simulator", "cpu"], default="htp",
                        help="推理后端")
    parser.add_argument("--image", type=str, required=True, help="测试图片路径")
    parser.add_argument("--task", choices=["layout", "vl", "both"], default="both",
                        help="测试任务")
    parser.add_argument("--prompt", type=str,
                        default="识别图中所有文字内容，保持原始格式输出为 Markdown",
                        help="VL 识别指令")

    args = parser.parse_args()

    # 加载图片
    from PIL import Image
    image = np.array(Image.open(args.image).convert("RGB"))

    # 创建服务
    service = create_service(args.model_dir, args.backend)

    # 布局检测
    if args.task in ("layout", "both"):
        logger.info("=" * 40)
        logger.info("布局检测")
        boxes, scores, labels = service.detect_layout(image)
        label_names = ["text", "title", "table", "figure", "formula"]
        for i in range(len(scores)):
            label = label_names[int(labels[i])] if int(labels[i]) < len(label_names) else str(int(labels[i]))
            logger.info(f"  [{label}] score={scores[i]:.3f} box={boxes[i].tolist()}")

    # VL 识别
    if args.task in ("vl", "both"):
        logger.info("=" * 40)
        logger.info("VL 识别")
        result = service.recognize_vl(image, args.prompt)
        print("\n--- 识别结果 ---")
        print(result)
