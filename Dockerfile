FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU wheels only. A GPU is optional and is not baked into this image.
RUN printf '%s\n' 'torch==2.14.1+cpu' 'torchvision==0.29.1+cpu' > /tmp/cpu-torch.txt \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu -c /tmp/cpu-torch.txt torch torchvision \
    && pip install -r requirements.txt -c /tmp/cpu-torch.txt

COPY . .
RUN useradd --create-home --uid 1000 cineguard \
    && mkdir -p /app/data /app/evidence /app/reports /app/models /app/demo_clips \
    && chown -R cineguard:cineguard /app

USER cineguard
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
