FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3 AS decoder
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils && rm -rf /var/lib/apt/lists/*
RUN curl --fail --location --retry 3 https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-10-29-19-17/ffmpeg-n9.0.2-37-gae19868753-linux64-gpl-9.0.tar.xz -o /tmp/ffmpeg.tar.xz \
    && echo '83537b0f6a97905f325080f6f5916e70ec252d1ef3e4ce4746f755b88720cbf5  /tmp/ffmpeg.tar.xz' | sha256sum --check - \
    && mkdir -p /decoder /unpack \
    && tar -xJf /tmp/ffmpeg.tar.xz -C /unpack --strip-components=1 \
    && cp /unpack/bin/ffmpeg /unpack/bin/ffprobe /decoder/ \
    && /decoder/ffmpeg -version | head -1 | grep -F 'ffmpeg version n9.0.2-37-gae19868753-20261029 ' \
    && /decoder/ffprobe -version | head -1 | grep -F 'ffprobe version n9.0.2-37-gae19868753-20261029 '

FROM node:22.22.2-bookworm-slim@sha256:9f6d5975c7dca860947d3915877f85607946403fc55349f39b4bc3688448bb6e AS web
WORKDIR /web
COPY serving/web/package.json serving/web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY serving/web/ ./
RUN npm run build

FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3 AS builder
WORKDIR /app
COPY env/serving.lock /tmp/serving.lock
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --require-hashes --only-binary=:all: --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/serving.lock \
    && /opt/venv/bin/pip check
COPY src/ /app/src/
COPY serving/__init__.py serving/app.py serving/jobs.py serving/index.html /app/serving/
COPY --from=web /web/dist/ /app/serving/web/dist/
COPY artifacts/serving/ /app/artifacts/serving/
RUN /opt/venv/bin/python -c "from pathlib import Path; import torch; from src.serving_bundle import verify_bundle; assert torch.version.cuda is None and not torch.cuda.is_available(); verify_bundle(Path('artifacts/serving/manifest.json'))"

FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=decoder /decoder/ /usr/local/bin/
COPY --from=builder /opt/venv/ /opt/venv/
COPY --chown=10001:10001 --from=builder /app/ /app/
RUN chmod -R a+rX /app
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "serving.app:create_app", "--factory", "--workers", "1", "--host", "0.0.0.0", "--port", "8000"]
