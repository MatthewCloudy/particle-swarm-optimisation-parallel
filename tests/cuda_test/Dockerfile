FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app
ENV PYTHONPATH=/app

COPY . .

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir \
        numpy \
        cupy-cuda12x

CMD ["python", "tests/cuda_test/cuda_test.py"]
