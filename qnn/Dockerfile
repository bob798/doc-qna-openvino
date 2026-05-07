FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 基础依赖
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    git wget curl unzip \
    build-essential cmake \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# Python 依赖（分层安装，利用 Docker 缓存）
RUN pip install --no-cache-dir \
    numpy>=1.24.0 \
    Pillow>=10.0.0 \
    onnx>=1.14.0 \
    paddle2onnx>=1.2.0

RUN pip install --no-cache-dir \
    paddlepaddle==2.6.2 \
    paddleocr>=2.9.0

RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    optimum>=1.19.0

# QNN SDK — 从 build context 复制并解压
COPY v2.41.0.251128.zip /tmp/qnn-sdk.zip
RUN unzip -q /tmp/qnn-sdk.zip -d /opt && rm /tmp/qnn-sdk.zip

ENV QNN_SDK_ROOT=/opt/qairt/2.41.0.251128
ENV PATH="${QNN_SDK_ROOT}/bin/x86_64-linux-clang:${PATH}"
ENV LD_LIBRARY_PATH="${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="${QNN_SDK_ROOT}/lib/python:${PYTHONPATH}"

WORKDIR /workspace/qnn
CMD ["/bin/bash"]
