# futsalcut-py: Python reference implementation + evaluation tools (CPU only; must run without AVX).
# Build on the test host with the python/ folder as context:
#   docker build -f docker/py.Dockerfile -t futsalcut-py python/
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ffmpeg/ffprobe CLI only for cross-checking outputs; decoding and remuxing go through PyAV.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
 && pip install --no-deps dvr-scan==1.8.2.1

WORKDIR /work
