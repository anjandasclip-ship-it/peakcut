# PeakCut: one image with the web server, the clip pipeline and ffmpeg.
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates bzip2 fonts-noto-core fonts-noto-cjk \
 && rm -rf /var/lib/apt/lists/*
# yt-dlp needs a JavaScript runtime for YouTube
COPY --from=denoland/deno:bin /deno /usr/local/bin/deno

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY studio studio
COPY web web

ENV PEAKCUT_DATA=/data \
    PEAKCUT_MODELS=/app/models \
    PEAKCUT_HOST=0.0.0.0 \
    PEAKCUT_PORT=8765 \
    PEAKCUT_TRUST_PROXY=1 \
    PYTHONUNBUFFERED=1 \
    PEAKCUT_MIN_FREE_DISK_GB=0.05

RUN python -m studio.fetch_model

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import os, urllib.request; p = os.environ.get('PORT') or os.environ.get('PEAKCUT_PORT') or '8765'; urllib.request.urlopen(f'http://127.0.0.1:{p}/healthz', timeout=4)"
CMD ["python", "-m", "studio.server"]
