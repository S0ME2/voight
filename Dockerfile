# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/voight \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    MODEL_DIR=/opt/voight/models \
    PYTHONPATH=/app
WORKDIR /app
RUN groupadd --system voight \
    && useradd --system --gid voight --home-dir "$HOME" --create-home voight \
    && mkdir -p /app/logs "$MODEL_DIR" \
    && chown -R voight:voight /app/logs "$HOME" "$MODEL_DIR" \
    && apt-get update \
    && apt-get install --no-install-recommends -y git libgl1 libglib2.0-0 libgomp1 libturbojpeg0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements/ /app/requirements/

FROM base AS cpu-deps
RUN pip install --no-cache-dir --no-deps -r requirements/cpu.lock

FROM base AS gpu-deps
RUN pip install --no-cache-dir --no-deps -r requirements/gpu.lock

FROM cpu-deps AS cpu-assets
COPY --chown=voight:voight app/ /app/app/
COPY --chown=voight:voight config/ /app/config/
COPY --chown=voight:voight scripts/models/prepare.py /app/scripts/models/prepare.py
ENV RUNTIME_TARGET=cpu PRELOAD=false
RUN python scripts/models/prepare.py && chown -R voight:voight "$MODEL_DIR"

FROM gpu-deps AS gpu-assets
COPY --chown=voight:voight app/ /app/app/
COPY --chown=voight:voight config/ /app/config/
# Paddle ships CUDA 11.8 packages in the shared ``nvidia`` namespace. Keep
# ONNX Runtime's CUDA 12 runtime isolated so the two stacks do not overwrite
# each other's shared libraries.
RUN pip install --no-cache-dir --no-deps --target /opt/cuda12 \
    nvidia-cublas-cu12==12.9.2.10 \
    nvidia-cuda-nvrtc-cu12==12.9.86 \
    nvidia-cuda-runtime-cu12==12.9.79 \
    nvidia-cudnn-cu12==9.25.1.1 \
    nvidia-cufft-cu12==11.4.1.4 \
    nvidia-curand-cu12==10.3.10.19 \
    nvidia-nvjitlink-cu12==12.9.86
ENV LD_LIBRARY_PATH=/opt/cuda12/nvidia/cublas/lib:/opt/cuda12/nvidia/cuda_nvrtc/lib:/opt/cuda12/nvidia/cuda_runtime/lib:/opt/cuda12/nvidia/cudnn/lib:/opt/cuda12/nvidia/cufft/lib:/opt/cuda12/nvidia/curand/lib:/opt/cuda12/nvidia/nvjitlink/lib
# Reuse CPU-prepared assets: importing the GPU Paddle wheel needs libcuda, but
# model preparation must remain GPU-free during image builds.
COPY --from=cpu-assets --chown=voight:voight /opt/voight/models /opt/voight/models
COPY --from=cpu-assets --chown=voight:voight /home/voight /home/voight
# These third-party assets are downloaded during preparation into package
# directories rather than MODEL_DIR.
COPY --from=cpu-assets --chown=voight:voight /usr/local/lib/python3.12/site-packages/capybara/vision/visualization/NotoSansMonoCJKtc-VF.ttf /usr/local/lib/python3.12/site-packages/capybara/vision/visualization/NotoSansMonoCJKtc-VF.ttf
COPY --from=cpu-assets --chown=voight:voight /usr/local/lib/python3.12/site-packages/docaligner/heatmap_reg/ckpt/fastvit_sa24_h_e_bifpn_256_fp32.onnx /usr/local/lib/python3.12/site-packages/docaligner/heatmap_reg/ckpt/fastvit_sa24_h_e_bifpn_256_fp32.onnx
COPY --from=cpu-assets --chown=voight:voight /usr/local/lib/python3.12/site-packages/mrzscanner/det/ckpt/mrz_detection_20250222_fp32.onnx /usr/local/lib/python3.12/site-packages/mrzscanner/det/ckpt/mrz_detection_20250222_fp32.onnx
COPY --from=cpu-assets --chown=voight:voight /usr/local/lib/python3.12/site-packages/mrzscanner/rec/ckpt/mrz_recognition_20250221_fp32.onnx /usr/local/lib/python3.12/site-packages/mrzscanner/rec/ckpt/mrz_recognition_20250221_fp32.onnx

FROM cpu-assets AS cpu
USER voight
ENV RUNTIME_TARGET=cpu PRELOAD=false
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/v1/health/ready').read()"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM cpu AS cpu-test
USER root
COPY scripts/ /app/scripts/
COPY tests/ /app/tests/
COPY README.md .env.example .gitignore setup.sh /app/
COPY docs/ /app/docs/
COPY .git/ /app/.git/
COPY annotation_input/ /app/annotation_input/
COPY annotations/ /app/annotations/
COPY benchmarks/ /app/benchmarks/
COPY archive/ /app/archive/
COPY experiments/ /app/experiments/
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]

FROM gpu-assets AS gpu
USER voight
ENV RUNTIME_TARGET=gpu GPU_ID=0 PRELOAD=false
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/v1/health/ready').read()"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
