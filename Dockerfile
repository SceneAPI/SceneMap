FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    SFMAPI_INSTANTSFM_SERVICE_PORT=8096 \
    SFMAPI_PLUGIN_WORKDIR=/sfmapi/work \
    SFMAPI_PLUGIN_CACHE=/sfmapi/cache \
    TORCH_HOME=/sfmapi/cache/torch \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda

ENV PATH="/opt/venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        colmap \
        git \
        libgl1 \
        libcudss0-dev-cuda-12 \
        libglib2.0-0 \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

ARG SCENEAPI_REF=main
ARG INSTANTSFM_REPO=https://github.com/cre185/InstantSfM.git
ARG INSTANTSFM_REF=main
ARG TORCH_DEVICE=cuda
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_PACKAGES="torch torchvision torchaudio"
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
ENV TORCH_DEVICE=${TORCH_DEVICE} \
    TORCH_INDEX_URL=${TORCH_INDEX_URL} \
    TORCH_CPU_INDEX_URL=${TORCH_CPU_INDEX_URL} \
    TORCH_PACKAGES=${TORCH_PACKAGES} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN uv venv "${VIRTUAL_ENV}"
RUN if [ -f third_party/instantsfm/pyproject.toml ]; then \
        echo "using bundled InstantSfM source"; \
    elif [ -d .git ]; then \
        git submodule update --init --recursive third_party/instantsfm; \
    else \
        mkdir -p third_party \
        && git clone --depth 1 --branch "${INSTANTSFM_REF}" "${INSTANTSFM_REPO}" third_party/instantsfm; \
    fi
RUN uv pip install "sceneapi @ git+https://github.com/SceneAPI/SceneAPI.git@${SCENEAPI_REF}"
RUN uv pip install --no-sources ".[server]"
RUN python -c "from scenemap.instantsfm.provisioning import provision; import json; print(json.dumps(provision(force=True), indent=2))"

EXPOSE 8096
CMD ["python", "-m", "scenemap.instantsfm.container_service"]
