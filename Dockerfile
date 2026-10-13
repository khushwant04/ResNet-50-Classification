FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only PyTorch wheels keep the image much smaller than the default CUDA build.
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt gunicorn

RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser app.py model.pth ./
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser templates/ ./templates/

USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/', timeout=3)" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "app:app"]
