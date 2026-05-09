"""
PaddleOCR-VL OpenVINO 推理封装

支持两种后端：
  - openvino_genai.VLMPipeline: 推荐，封装好的视觉语言推理 API
  - 原生 openvino.Core() 直接加载 IR: 备用路径，用于无 GenAI 环境

同时暴露 PyTorch 推理路径（HuggingFace transformers）作为 Benchmark 对照组。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "请完整识别这个文档图片中的所有内容（包括文字、表格、公式、图表标题），"
    "保留原始排版与阅读顺序，输出为结构化 Markdown 格式。"
    "表格请输出为 Markdown 表格语法，公式请输出为 LaTeX。"
)


@dataclass
class InferenceResult:
    """单次推理结果与计时"""

    text: str = ""
    backend: str = ""
    image_path: str = ""
    elapsed_ms: float = 0.0
    extras: dict = field(default_factory=dict)


# ── OpenVINO GenAI 后端 ──────────────────────────────────────────────────────


class OpenVINOPaddleOCRVL:
    """OpenVINO GenAI VLMPipeline 封装"""

    def __init__(
        self,
        ir_dir: str | Path,
        device: str = "CPU",
        max_new_tokens: int = 2048,
    ):
        self.ir_dir = Path(ir_dir)
        self.device = device
        self.max_new_tokens = max_new_tokens

        if not self.ir_dir.exists():
            raise FileNotFoundError(
                f"未找到 OpenVINO IR 目录: {self.ir_dir}\n"
                f"请先运行 paddleocr_vl notebook 完成 IR 转换。"
            )

        try:
            import openvino_genai as ov_genai
        except ImportError as e:
            raise ImportError(
                "openvino_genai 未安装，请 `pip install openvino-genai`"
            ) from e

        logger.info(f"加载 OpenVINO VLMPipeline: {self.ir_dir} on {device}")
        self.pipe = ov_genai.VLMPipeline(str(self.ir_dir), device)
        self.config = ov_genai.GenerationConfig()
        self.config.max_new_tokens = max_new_tokens

    @staticmethod
    def _image_to_tensor(image: Image.Image | np.ndarray):
        import openvino as ov

        if isinstance(image, np.ndarray):
            arr = image
        else:
            arr = np.array(image.convert("RGB"))
        # VLMPipeline 期望 NHWC uint8
        if arr.ndim == 3:
            arr = arr[None, ...]
        return ov.Tensor(arr.astype(np.uint8))

    def infer(
        self,
        image: str | Path | Image.Image | np.ndarray,
        prompt: str = DEFAULT_PROMPT,
    ) -> InferenceResult:
        if isinstance(image, (str, Path)):
            image_path = str(image)
            pil = Image.open(image_path).convert("RGB")
        else:
            image_path = ""
            pil = image if isinstance(image, Image.Image) else Image.fromarray(image)

        tensor = self._image_to_tensor(pil)

        t0 = time.perf_counter()
        out = self.pipe.generate(prompt, image=tensor, generation_config=self.config)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        text = out if isinstance(out, str) else getattr(out, "texts", [str(out)])[0]
        return InferenceResult(
            text=text,
            backend="openvino",
            image_path=image_path,
            elapsed_ms=elapsed_ms,
        )


# ── PyTorch 后端（Benchmark 对照） ─────────────────────────────────────────────


class PyTorchPaddleOCRVL:
    """HuggingFace transformers 加载 PaddleOCR-VL，用作速度对照组"""

    def __init__(
        self,
        hf_repo: str = "PaddlePaddle/PaddleOCR-VL",
        device: str = "cpu",
        max_new_tokens: int = 2048,
        torch_dtype: Optional[str] = None,
    ):
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "需要安装 torch + transformers: `pip install torch transformers`"
            ) from e
        import torch

        self.device = device
        self.max_new_tokens = max_new_tokens

        logger.info(f"加载 PyTorch PaddleOCR-VL: {hf_repo} on {device}")
        kwargs = {"trust_remote_code": True}
        if torch_dtype:
            kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        self.processor = AutoProcessor.from_pretrained(hf_repo, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(hf_repo, **kwargs).to(device)
        self.model.eval()

    def infer(
        self,
        image: str | Path | Image.Image | np.ndarray,
        prompt: str = DEFAULT_PROMPT,
    ) -> InferenceResult:
        import torch

        if isinstance(image, (str, Path)):
            image_path = str(image)
            pil = Image.open(image_path).convert("RGB")
        elif isinstance(image, np.ndarray):
            image_path = ""
            pil = Image.fromarray(image)
        else:
            image_path = ""
            pil = image

        inputs = self.processor(images=pil, text=prompt, return_tensors="pt").to(self.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return InferenceResult(
            text=text,
            backend="pytorch",
            image_path=image_path,
            elapsed_ms=elapsed_ms,
        )


# ── Tesseract 基线（解析质量对比） ────────────────────────────────────────────


class TesseractEngine:
    """pytesseract 封装，用作解析质量基线"""

    def __init__(self, lang: str = "chi_sim+eng", psm: int = 3):
        try:
            import pytesseract  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "未安装 pytesseract，请 `pip install pytesseract` 并安装 tesseract 二进制"
            ) from e
        self.lang = lang
        self.psm = psm

    def infer(
        self,
        image: str | Path | Image.Image | np.ndarray,
        prompt: str = "",  # 兼容统一接口
    ) -> InferenceResult:
        import pytesseract

        if isinstance(image, (str, Path)):
            image_path = str(image)
            pil = Image.open(image_path).convert("RGB")
        elif isinstance(image, np.ndarray):
            image_path = ""
            pil = Image.fromarray(image)
        else:
            image_path = ""
            pil = image

        config = f"--psm {self.psm}"
        t0 = time.perf_counter()
        text = pytesseract.image_to_string(pil, lang=self.lang, config=config)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return InferenceResult(
            text=text,
            backend="tesseract",
            image_path=image_path,
            elapsed_ms=elapsed_ms,
        )


# ── 工具：批量推理 ───────────────────────────────────────────────────────────


def iter_images(image_dir: str | Path, exts={".png", ".jpg", ".jpeg", ".bmp", ".tiff"}) -> List[Path]:
    return sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts)
