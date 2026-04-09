# Dockerfile — PROJECT ROOT
# ─────────────────────────────────────────────────────────────────────────────
# FLIDIE OpenEnv Container
#
# Port 7860 is HF Spaces convention. When testing locally, map it:
#   docker build -t flidie-env .
#   docker run --rm -p 8000:7860 flidie-env
#   curl http://localhost:8000/health
# ─────────────────────────────────────────────────────────────────────────────

# python:3.11-slim: small base image (~55MB), no dev tools, production safe.
# Do NOT use python:3.11 (full image is 1.2GB — HF Spaces build timeout risk).
# Do NOT use python:3.12+ (openenv-core tested on 3.11).
FROM python:3.11-slim

# Set working directory. All paths inside the container are relative to /app.
WORKDIR /app

# Install uv — the fast package manager used by the OpenEnv ecosystem.
# --no-cache-dir keeps the image small by not storing pip's download cache.
# uv is significantly faster than pip for dependency resolution.
RUN pip install --no-cache-dir uv



# Copy dependency files FIRST (before source code).
# Docker layer caching: if pyproject.toml and uv.lock don't change,
# the expensive `uv sync` step is skipped on subsequent builds.
# This turns a 60-second rebuild into a 3-second rebuild.
COPY pyproject.toml uv.lock ./

# Install all dependencies into /app/.venv.
# --frozen: use uv.lock exactly, don't resolve — ensures reproducibility.
# --no-dev: exclude test dependencies (pytest, httpx) from production image.
RUN uv sync --frozen --no-dev

# Copy the entire project into the container.
# .dockerignore controls what is excluded (see §07).
COPY . .

# 7860 is HF Spaces convention. MUST match the uvicorn port in CMD.
EXPOSE 7860

# Add the uv-managed venv to PATH so uvicorn is found.
# Without this, CMD would fail with "uvicorn: not found".
ENV PATH="/app/.venv/bin:$PATH"

ENV ENABLE_WEB_INTERFACE=true

# Healthcheck: HF Spaces probes /health every 30s.
# If this fails 3 times, the Space is marked unhealthy and restarts.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Start the server.
# host 0.0.0.0: listen on all interfaces (required for Docker networking).
# workers 1: single worker is fine for evaluation — keeps memory low.
# Do NOT use --reload in production (file watching uses extra CPU/memory).
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
